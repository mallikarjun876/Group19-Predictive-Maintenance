"""Reusable controls and faithfulness utilities for explanation evaluation.

These functions are deliberately model-agnostic. Call them from an evaluation
script with the same (sensor,time) reference mask used by the main scorer.
"""
from __future__ import annotations
import numpy as np
from scoring.iou import score_batch


def explanation_controls(reference_mask, n=200, seed=0):
    """Return random, wrong-sensor and shuffled-reference control IoU summaries."""
    rng = np.random.default_rng(seed)
    ref = np.asarray(reference_mask, dtype=bool)
    if ref.ndim != 2:
        raise ValueError("reference_mask must have shape (sensors, time)")
    S, T = ref.shape
    random_maps = rng.normal(size=(n, S, T))
    wrong = ~ref
    wrong_maps = np.broadcast_to(wrong, (n, S, T)).astype(float)
    shuffled = ref.ravel().copy()
    rng.shuffle(shuffled)
    shuffled = np.broadcast_to(shuffled.reshape(S, T), (n, S, T))
    return {
        "random": score_batch(random_maps, ref, n_null=500, rng=seed),
        "wrong_sensor": score_batch(wrong_maps, ref, n_null=500, rng=seed + 1),
        "shuffled_reference": score_batch(random_maps, shuffled, n_null=500, rng=seed + 2),
    }


def deletion_insertion(predict_fn, window, attribution, baseline, fractions=None):
    """Compute deletion/insertion prediction trajectories for one (T,S) window.

    predict_fn accepts a batch shaped (B,T,S) and returns one scalar per row.
    Deletion replaces highest-ranked cells with baseline; insertion starts at
    baseline and restores cells in descending attribution order.
    """
    x = np.asarray(window, dtype=float)
    a = np.asarray(attribution, dtype=float)
    if a.shape != (x.shape[1], x.shape[0]):
        raise ValueError("attribution must have shape (S,T)")
    base = np.asarray(baseline, dtype=float)
    if base.shape == (x.shape[1],):
        base = np.broadcast_to(base[None, :], x.shape)
    if base.shape != x.shape:
        raise ValueError("baseline must have shape (S,) or (T,S)")
    fractions = np.asarray(fractions if fractions is not None else np.linspace(0, 1, 11))
    order = np.argsort(np.abs(a).ravel())[::-1]
    flat_x, flat_base = x.T.ravel().copy(), base.T.ravel()
    deletion, insertion = [], []
    for frac in fractions:
        k = int(round(float(frac) * len(order)))
        d, ins = flat_x.copy(), flat_base.copy()
        d[order[:k]] = flat_base[order[:k]]
        ins[order[:k]] = flat_x[order[:k]]
        deletion.append(float(np.asarray(predict_fn(d.reshape(x.shape[1], x.shape[0]).T[None])).ravel()[0]))
        insertion.append(float(np.asarray(predict_fn(ins.reshape(x.shape[1], x.shape[0]).T[None])).ravel()[0]))
    return {"fractions": fractions.tolist(), "deletion": deletion, "insertion": insertion}


def trapezoid_auc(values, fractions):
    """Area under a deletion/insertion curve."""
    return float(np.trapz(np.asarray(values, float), np.asarray(fractions, float)))
