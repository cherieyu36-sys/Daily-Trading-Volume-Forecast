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
    assert len(m)==40 and len(f)==25
    np.testing.assert_allclose(s.groupby(['date','strategy']).shares.sum(),10000)

def test_profile_forecasts_and_scale_invariance():
    days = panels(synthetic(days=20))
    history = [a for _, a in days[:15]]
    f = forecasts(history, days[15][1])
    for name in ("profile_ridge", "profile_shrinkage"):
        assert np.isfinite(f[name]).all() and (f[name] > 0).all()
        np.testing.assert_allclose(f[name].sum(), f["scaled"].sum())
    np.testing.assert_allclose(allocate(f["scaled"], 10000), allocate(f["historical"], 10000))

def test_shrinkage_validation_is_chronological():
    from profile_models import select_alpha, ALPHAS
    from research import features
    history = [a for _, a in panels(synthetic(days=15))]
    calls = []
    def tracked(a):
        calls.append(a)
        return features(a)
    assert select_alpha(history, tracked) in ALPHAS
    assert all(any(a is h for h in history) or any(np.array_equal(a,h) for h in history) for a in calls)
