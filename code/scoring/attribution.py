"""
Attribution methods that produce the (n_sensors, n_time) maps the scorer needs.

occlusion_attribution  model-agnostic, needs only predict_fn(X) -> scores. Works
                       with any model (sklearn, or a torch model wrapped in a function).
grad_x_input           PyTorch gradient x input. NOT tested in this environment
                       (torch was not installed) - run it once on your machine.
random_attribution     chance-level control.
"""
import numpy as np


def occlusion_attribution(predict_fn, window, baseline, block=5):
    """
    window:   (T, S) one window, scaled
    baseline: (S,)   value to occlude with (0 after standardisation = healthy mean)
    block:    time steps occluded together
    Returns (S, T): drop in the fault score when that (sensor, time-block) is
    replaced by the baseline. Large positive value = cell supported the prediction.
    """
    T, S = window.shape
    base = float(predict_fn(window[None])[0])
    attr = np.zeros((S, T))
    for s in range(S):
        for t0 in range(0, T, block):
            w = window.copy()
            w[t0:t0 + block, s] = baseline[s]
            attr[s, t0:t0 + block] = base - float(predict_fn(w[None])[0])
    return attr


def random_attribution(shape, rng=None):
    return np.random.default_rng(rng).random(shape)


def grad_x_input(model, x, target_index=None):
    """
    x: torch tensor (B, T, S). Returns numpy (B, S, T) of |gradient * input|.
    For a single-logit fault classifier leave target_index=None.
    """
    import torch
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    out = out.squeeze(-1) if target_index is None else out[:, target_index]
    out.sum().backward()
    return (x.grad * x).abs().detach().cpu().numpy().transpose(0, 2, 1)
