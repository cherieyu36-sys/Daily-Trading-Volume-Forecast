"""Execution-aware volume models with chronological shrinkage selection."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CUT = 6
ALPHAS = (0., .25, .5, .75, 1.)

def weights(v):
    v = np.maximum(np.asarray(v, dtype=float), 1e-12)
    return v / v.sum(axis=-1, keepdims=True)

def profile_components(history, today, features):
    h = np.stack(history)
    target = weights(h[:, CUT:, 2])
    baseline = weights(h[:, CUT:, 2].mean(axis=0))
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.))
    model.fit(np.stack([features(a) for a in h]), target)
    predicted = weights(model.predict(features(today)[None, :])[0])
    return baseline, predicted

def select_alpha(history, features):
    # Each validation prediction uses only sessions preceding its own date.
    errors = np.zeros(len(ALPHAS))
    start = max(5, len(history)-10)
    if len(history) <= start:
        return 0.
    for i in range(start, len(history)):
        base, pred = profile_components(history[:i], history[i], features)
        actual = weights(history[i][CUT:, 2])
        for j, alpha in enumerate(ALPHAS):
            errors[j] += np.abs(actual - ((1-alpha)*base + alpha*pred)).sum()
    return ALPHAS[int(np.argmin(errors))]

def two_stage_forecasts(history, today, features):
    h = np.stack(history)
    morning = today[:CUT, 2].sum()
    total = h[:, CUT:, 2].sum(axis=1).mean() * morning / h[:, :CUT, 2].sum(axis=1).mean()
    base, pred = profile_components(history, today, features)
    alpha = select_alpha(history, features)
    return {"profile_ridge": total*pred,
            "profile_shrinkage": total*((1-alpha)*base + alpha*pred)}
