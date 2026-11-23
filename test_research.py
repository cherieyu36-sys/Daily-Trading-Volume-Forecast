import numpy as np
from research import panels, synthetic, forecasts, allocate, backtest

def test_no_future_leakage():
    days = panels(synthetic(days=20))
    history = [x for _,x in days[:10]]
    today = days[10][1].copy()
    first = forecasts(history,today)
    today[6:] *= 10
    second = forecasts(history,today)
    for k in first: np.testing.assert_allclose(first[k],second[k])

def test_allocation():
    for risk in [0,.1]:
        q = allocate(np.arange(1,21),10000,risk)
        assert np.isclose(q.sum(),10000) and (q>=0).all()

def test_backtest():
    m,s,f = backtest(panels(synthetic(days=15)))
    assert len(m)==25 and len(f)==15
    np.testing.assert_allclose(s.groupby(['date','strategy']).shares.sum(),10000)
