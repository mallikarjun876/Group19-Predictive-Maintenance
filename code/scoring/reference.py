"""
Reference fault signatures: the "ground truth" the IoU score compares against.

IMPORTANT (be ready to defend this on the slides)
-------------------------------------------------
Neither dataset ships a per-sample "these sensors caused the fault" label.
So the reference is built from two independent ingredients:

  1. a domain prior   -> which sensors the fault physically touches
  2. a data check     -> do those sensors actually shift during failures?

A sensor enters the reference only if it passes BOTH. The time dimension can
optionally be restricted to the steps where the sensor really deviates from
its healthy behaviour. Everything is reported, so the reference is auditable.
"""
import numpy as np

# ---------------------------------------------------------------- MetroPT-3
# Failure log from the MetroPT-3 documentation (UCI). VERIFY these against the
# dataset page before presenting; they are entered from memory of that table.
METROPT_FAILURES = [
    ("2020-04-18 00:00", "2020-04-18 23:59", "air leak, reservoirs"),
    ("2020-05-29 23:30", "2020-05-30 06:00", "air leak, reservoirs"),
    ("2020-06-05 10:00", "2020-06-07 14:30", "air leak, reservoirs"),
    ("2020-07-15 14:30", "2020-07-15 19:00", "air leak, reservoirs"),
]

# Domain prior for an air leak in the air-production unit: pressure signals fall
# faster, so the compressor works harder (duty cycle, motor current, oil temp).
# This is a hypothesis from domain reasoning, to be confirmed by effect_sizes().
METROPT_PRIOR = {
    "primary":   ["TP2", "TP3", "Reservoirs"],
    "secondary": ["Motor_current", "Oil_temperature", "COMP", "MPG"],
}


def effect_sizes(X, y, feature_names):
    """
    Standardised mean difference (|Cohen's d|) between fault and normal rows.
    X: (n_samples, n_features) raw or windowed-then-averaged features.
    y: (n_samples,) 0/1.  Returns {feature: d}.
    """
    X = np.asarray(X, float)
    y = np.asarray(y).astype(bool)
    a, b = X[y], X[~y]
    out = {}
    for j, name in enumerate(feature_names):
        pooled = np.sqrt((a[:, j].var(ddof=1) + b[:, j].var(ddof=1)) / 2) + 1e-12
        out[name] = float(abs(a[:, j].mean() - b[:, j].mean()) / pooled)
    return out


def validated_sensors(prior_sensors, effects, min_d=0.5):
    """Keep prior sensors whose fault-vs-normal effect size is at least min_d (medium effect)."""
    return [s for s in prior_sensors if effects.get(s, 0.0) >= min_d]


def sensor_mask(sensor_names, chosen):
    chosen = set(chosen)
    return np.array([s in chosen for s in sensor_names], dtype=bool)


def reference_mask(window, sensor_sel, healthy_mean=None, healthy_std=None, z_thr=None):
    """
    Build the (n_sensors, n_time) reference for ONE window.

    window      (n_sensors, n_time) in the same units as healthy_mean/std
    sensor_sel  boolean (n_sensors,) from sensor_mask()
    z_thr       None  -> every time step of a selected sensor is reference
                float -> only steps where |z-score vs healthy| > z_thr
    """
    S, T = window.shape
    ref = np.zeros((S, T), dtype=bool)
    ref[sensor_sel, :] = True
    if z_thr is not None:
        z = np.abs((window - healthy_mean[:, None]) / (healthy_std[:, None] + 1e-12))
        ref &= z > z_thr
        # a window with no deviating step would give an empty reference; keep the
        # sensor rows whole in that case so the sample is not silently dropped
        if ref.sum() == 0:
            ref[sensor_sel, :] = True
    return ref


# ------------------------------------------------------------------ C-MAPSS
def trend_sensors(sensor_matrix, rul, sensor_names, top_k=6):
    """
    Data-driven degradation sensors for C-MAPSS: rank sensors by |Spearman
    correlation| with RUL (after per-operating-condition normalisation) and keep
    the top_k. Combine with a prior from Saxena et al. (2008) for the final set.
    """
    from scipy.stats import spearmanr
    scores = {}
    for j, name in enumerate(sensor_names):
        col = sensor_matrix[:, j]
        if np.std(col) < 1e-9:
            scores[name] = 0.0
            continue
        rho, _ = spearmanr(col, rul)
        scores[name] = float(abs(rho))
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:top_k], scores
