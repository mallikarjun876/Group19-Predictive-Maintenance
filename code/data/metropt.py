"""
MetroPT-3 preprocessing: load, label failures, window, split by time, EDA plot.

Usage
-----
    from data.metropt import *
    df = load_metropt("MetroPT3(AirCompressor).csv")
    df = label_failures(df)
    print(df.timestamp.diff().median())          # check the real sampling interval
    X, y, ts, cols = make_windows(df, win=60, stride=60)
    splits = chronological_split(ts)
"""
import numpy as np
import pandas as pd
from scoring.reference import METROPT_FAILURES

TIME_COL = "timestamp"


def load_metropt(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df.sort_values(TIME_COL).reset_index(drop=True)


def feature_columns(df):
    return [c for c in df.columns if c not in (TIME_COL, "fault", "event")]


def label_failures(df, failures=METROPT_FAILURES):
    """Adds `fault` (0/1) and `event` (-1 normal, else index of the failure)."""
    df = df.copy()
    df["fault"] = 0
    df["event"] = -1
    for i, (start, end, _) in enumerate(failures):
        m = (df[TIME_COL] >= pd.Timestamp(start)) & (df[TIME_COL] <= pd.Timestamp(end))
        df.loc[m, "fault"] = 1
        df.loc[m, "event"] = i
    return df


def make_windows(df, win=60, stride=60, min_fault_frac=0.5, cols=None):
    """
    Slide a window over the series.
    A window is positive if >= min_fault_frac of its rows are inside a failure.
    Returns X (N, win, n_features), y (N,), t_start (N,), cols.
    Windows are NOT shuffled: keep them in time order for the split.
    """
    cols = cols or feature_columns(df)
    arr = df[cols].to_numpy(dtype=np.float32)
    f = df["fault"].to_numpy()
    starts = np.arange(0, len(df) - win + 1, stride)
    idx = starts[:, None] + np.arange(win)[None, :]
    X = arr[idx]
    y = (f[idx].mean(axis=1) >= min_fault_frac).astype(np.int64)
    t0 = df[TIME_COL].to_numpy()[starts]
    return X, y, t0, cols


def chronological_split(t_start, train_end="2020-05-31", val_end="2020-07-01"):
    """
    Split by TIME, never at random (neighbouring windows overlap in behaviour, so
    a random split leaks). With the 4 logged failures this gives:
      train: failures 1-2   val: failure 3   test: failure 4
    """
    t = pd.to_datetime(t_start)
    tr = np.where(t <= pd.Timestamp(train_end))[0]
    va = np.where((t > pd.Timestamp(train_end)) & (t <= pd.Timestamp(val_end)))[0]
    te = np.where(t > pd.Timestamp(val_end))[0]
    return {"train": tr, "val": va, "test": te}


def fit_scaler(X_train, y_train):
    """Mean/std from NORMAL training windows only (also serves as 'healthy' stats)."""
    normal = X_train[y_train == 0].reshape(-1, X_train.shape[-1])
    mu, sd = normal.mean(axis=0), normal.std(axis=0)
    sd[sd < 1e-8] = 1.0          # constant digital channels
    return mu, sd


def apply_scaler(X, mu, sd):
    return (X - mu) / sd


def class_weights(y_train):
    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    return {"pos_weight": n_neg / max(n_pos, 1), "n_pos": n_pos, "n_neg": n_neg,
            "positive_rate": n_pos / max(n_pos + n_neg, 1)}


def plot_eda(df, cols=None, out_png="metropt_eda.png", failures=METROPT_FAILURES, max_points=20000):
    """Sensor traces with failure intervals shaded (downsampled for speed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = cols or feature_columns(df)
    step = max(1, len(df) // max_points)
    d = df.iloc[::step]
    fig, axes = plt.subplots(len(cols), 1, figsize=(13, 1.6 * len(cols)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), cols):
        ax.plot(d[TIME_COL], d[c], lw=0.5, color="#2b6cb0")
        for s, e, _ in failures:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#e53e3e", alpha=0.35, lw=0)
        ax.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("MetroPT-3 sensors (red = logged failures)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png
