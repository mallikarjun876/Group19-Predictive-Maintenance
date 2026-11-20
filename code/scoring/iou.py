"""
Physics-consistency score (Novelty B): IoU between the (sensor, time) cells an
explanation highlights and a reference fault signature.

Conventions used everywhere in this project
-------------------------------------------
* An attribution is an array of shape (n_sensors, n_time) for ONE prediction,
  or (batch, n_sensors, n_time) for many.
* A reference mask is a boolean array with the same shape (True = cell is part
  of the known fault signature).
* By default k (the number of top cells compared) equals the reference size,
  so a perfect explanation can reach IoU = 1 and there is no free parameter
  to tune.
"""
import numpy as np


def topk_mask(attr, k=None, frac=None, use_abs=True):
    """Boolean mask of the k highest-magnitude cells of a single attribution map."""
    a = np.abs(attr) if use_abs else np.asarray(attr)
    flat = a.ravel()
    if k is None:
        if frac is None:
            raise ValueError("give k or frac")
        k = int(round(frac * flat.size))
    k = int(min(max(k, 1), flat.size))
    idx = np.argpartition(flat, -k)[-k:]
    m = np.zeros(flat.size, dtype=bool)
    m[idx] = True
    return m.reshape(a.shape)


def iou(mask_a, mask_b):
    """Intersection over union of two boolean masks (0.0 if both are empty)."""
    a, b = np.asarray(mask_a, bool), np.asarray(mask_b, bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def iou_upper_bound(k, ref_size):
    """Best IoU any explanation can get when comparing k cells to a reference of ref_size."""
    return min(k, ref_size) / max(k, ref_size)


def null_distribution(ref_mask, k, n=2000, rng=None):
    """IoU of n random top-k selections against ref_mask (the 'chance' distribution)."""
    rng = np.random.default_rng(rng)
    ref = np.asarray(ref_mask, bool).ravel()
    size = ref.size
    k = int(min(max(k, 1), size))
    r = rng.random((n, size))
    idx = np.argpartition(r, -k, axis=1)[:, -k:]
    inter = ref[idx].sum(axis=1)
    union = k + ref.sum() - inter
    return inter / union


def physics_iou(attr, ref_mask, k=None, frac=None, n_null=2000, rng=None, use_abs=True):
    """
    Score one explanation against one reference mask.

    Returns a dict with:
      iou        observed IoU of top-k cells vs reference
      k          number of cells compared
      upper      the maximum IoU achievable for this k and reference size
      chance     mean IoU of random selections (what 'no skill' looks like)
      z, p       how far above chance the observed IoU is (permutation test)
    """
    ref_mask = np.asarray(ref_mask, bool)
    if k is None and frac is None:
        k = int(ref_mask.sum())
    pred = topk_mask(attr, k=k, frac=frac, use_abs=use_abs)
    k_used = int(pred.sum())
    obs = iou(pred, ref_mask)
    null = null_distribution(ref_mask, k_used, n=n_null, rng=rng)
    sd = null.std() if null.std() > 0 else 1e-12
    return {
        "iou": obs,
        "k": k_used,
        "upper": iou_upper_bound(k_used, int(ref_mask.sum())),
        "chance": float(null.mean()),
        "z": float((obs - null.mean()) / sd),
        "p": float((1 + (null >= obs).sum()) / (len(null) + 1)),
    }


def sensor_iou(attr, ref_sensor_mask, k=None, use_abs=True):
    """Coarser variant: collapse time, compare the top-k SENSORS to the reference sensors."""
    a = np.abs(attr) if use_abs else np.asarray(attr)
    per_sensor = a.mean(axis=-1)
    ref = np.asarray(ref_sensor_mask, bool)
    if k is None:
        k = int(ref.sum())
    return iou(topk_mask(per_sensor, k=k, use_abs=False), ref)


def score_batch(attrs, ref_masks, k=None, frac=None, n_null=500, rng=0):
    """
    Score many explanations. attrs: (B,S,T). ref_masks: (B,S,T) or a single (S,T) mask.
    Returns a dict of per-sample arrays plus their means.
    """
    attrs = np.asarray(attrs)
    ref_masks = np.asarray(ref_masks, bool)
    if ref_masks.ndim == 2:
        ref_masks = np.broadcast_to(ref_masks, attrs.shape)
    rows = [physics_iou(a, r, k=k, frac=frac, n_null=n_null, rng=rng)
            for a, r in zip(attrs, ref_masks)]
    out = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    out["mean_iou"] = float(out["iou"].mean())
    out["mean_chance"] = float(out["chance"].mean())
    out["mean_z"] = float(out["z"].mean())
    return out
