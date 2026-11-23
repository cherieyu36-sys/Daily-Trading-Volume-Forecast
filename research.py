"""Walk-forward intraday volume forecasting and execution research."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

SLOTS = pd.date_range('2000-01-01 09:30', periods=26, freq='15min').strftime('%H:%M').tolist()
CUT = 6  # Six completed bars at 11:00 ET; 20 remain.


def load_yahoo(ticker):
    import yfinance as yf
    raw = yf.Ticker(ticker).history(period='1mo', interval='15m', auto_adjust=False, prepost=False)
    if raw.empty:
        raise RuntimeError('Yahoo returned no data; retry later or use --synthetic.')
    raw = raw.tz_convert('America/New_York')
    raw['date'] = raw.index.date
    raw['slot'] = raw.index.strftime('%H:%M')
    raw = raw[raw.slot.isin(SLOTS)]
    return raw.reset_index(drop=True)


def synthetic(seed=42, days=80):
    rng = np.random.default_rng(seed)
    rows = []
    shape = 1 + 2.5 * np.linspace(-1, 1, 26)**2
    for day in pd.bdate_range('2025-01-02', periods=days):
        activity = rng.lognormal(0, .35)
        volume = np.round(120000 * activity * shape * rng.lognormal(0, .18, 26))
        price = 100 * np.exp(np.cumsum(rng.normal(0, .0015, 26)))
        for i, slot in enumerate(SLOTS):
            rows.append((day.date(), slot, price[i-1] if i else 100., price[i], volume[i]))
    return pd.DataFrame(rows, columns=['date','slot','Open','Close','Volume'])


def panels(raw):
    raw = raw.copy()
    raw['date'] = pd.to_datetime(raw['date']).dt.date
    result = []
    for day, frame in raw.groupby('date', sort=True):
        if len(frame) != 26 or set(frame.slot) != set(SLOTS):
            continue  # Exclude half days, missing bars and partial sessions.
        f = frame.set_index('slot').loc[SLOTS]
        a = f[['Open','Close','Volume']].to_numpy(float)
        if np.isfinite(a).all() and (a > 0).all():
            result.append((day, a))
    return result


def features(a):
    morning = a[:CUT]
    r = np.diff(np.log(np.r_[morning[0,0], morning[:,1]]))
    return np.array([np.log(morning[:,2].sum()), r.sum(), np.sqrt((r*r).sum()), morning[-2:,2].sum()/morning[:,2].sum()])


def forecasts(history, today):
    h = np.stack(history)
    avg = h[:,CUT:,2].mean(axis=0)
    ratio = today[:CUT,2].sum()/h[:,:CUT,2].sum(axis=1).mean()
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(np.stack([features(x) for x in h]), np.log1p(h[:,CUT:,2]))
    ridge = np.maximum(np.expm1(model.predict(features(today)[None,:])[0]), 1.)
    return {'historical':avg, 'scaled':avg*ratio, 'ridge':ridge}


def allocate(v, q, risk=0.):
    """Continuous-share optimisation, quadratic impact and inventory penalty."""
    w = v/v.sum()
    if risk == 0:
        return q*w
    def objective(x):
        return np.sum(x*x/w) + risk*np.sum((1-np.cumsum(x))**2)
    res = minimize(objective, w, method='SLSQP', bounds=[(0,1)]*len(w),
                   constraints={'type':'eq','fun':lambda x:x.sum()-1}, options={'ftol':1e-10})
    if not res.success:
        raise RuntimeError(res.message)
    return q*np.maximum(res.x,0)/np.maximum(res.x,0).sum()


def backtest(days, order=10000, min_train=10, lookback=30, impact_bps=20., half_spread_bps=1., risk=.1):
    metrics, schedules, predictions = [], [], []
    for i in range(min_train, len(days)):
        day, a = days[i]
        history = [x for _,x in days[max(0,i-lookback):i]]
        f = forecasts(history,a)
        v, p = a[CUT:,2], a[CUT:,1]
        arrival = a[CUT-1,1]
        market_vwap = np.average(p, weights=v)  # bar-close proxy
        for name, pred in f.items():
            predictions.append({'date':day,'model':name,'mae_shares':np.abs(v-pred).mean(), 'share_l1':np.abs(v/v.sum()-pred/pred.sum()).sum()})
        strategies = {'twap':np.full(len(v), order/len(v)), **{name+'_vwap':allocate(pred,order) for name,pred in f.items()}, 'ridge_optimised':allocate(f['ridge'],order,risk)}
        for name, q in strategies.items():
            participation = q/v
            cost = half_spread_bps + impact_bps*participation
            fill = p*(1-cost/10000)
            execution = np.average(fill, weights=q)
            metrics.append({'date':day,'strategy':name,'is_bps':10000*(arrival-execution)/arrival,'vwap_slippage_bps':10000*(market_vwap-execution)/market_vwap,'modelled_cost_bps':np.average(cost,weights=q),'max_participation':participation.max(),'completion':q.sum()/order})
            pred = f.get(name.removesuffix('_vwap'), f['ridge'] if name=='ridge_optimised' else None)
            for j, slot in enumerate(SLOTS[CUT:]):
                schedules.append({'date':day,'strategy':name,'slot_et':slot,'shares':q[j],'forecast_volume':pred[j] if pred is not None else np.nan,'actual_volume':v[j]})
    return tuple(pd.DataFrame(x) for x in (metrics,schedules,predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ticker',default='AAPL')
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--order',type=int,default=10000)
    parser.add_argument('--output',default='outputs')
    args = parser.parse_args()
    if args.order <= 0: parser.error('--order must be positive')
    days = panels(synthetic() if args.synthetic else load_yahoo(args.ticker))
    if len(days) <= 10: raise RuntimeError('Need at least 11 complete trading sessions.')
    metrics, schedule, forecast = backtest(days,args.order)
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    for name, df in [('metrics',metrics),('schedule',schedule),('forecast_errors',forecast)]:
        df.to_csv(out/f'{name}.csv',index=False)
    summary = metrics.groupby('strategy').agg(mean_is_bps=('is_bps','mean'), std_is_bps=('is_bps','std'), mean_cost_bps=('modelled_cost_bps','mean'), mean_vwap_slippage_bps=('vwap_slippage_bps','mean'), max_participation=('max_participation','max'))
    summary.to_csv(out/'summary.csv')
    print('Source:', 'SYNTHETIC' if args.synthetic else args.ticker, '| Test days:',metrics.date.nunique())
    print(summary.round(4).to_string())
    print(forecast.groupby('model')[['mae_shares','share_l1']].mean().round(4))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); ax.plot(SLOTS,np.stack([a[:,2] for _,a in days]).mean(axis=0)); ax.set_xticks(SLOTS[::4]); ax.set(title='Mean intraday volume',xlabel='Time (ET)',ylabel='Shares'); fig.tight_layout(); fig.savefig(out/'volume_profile.png'); plt.close(fig)

if __name__ == '__main__': main()
