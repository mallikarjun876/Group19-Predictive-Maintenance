"""C-MAPSS loading, per-operating-condition normalisation, RUL labels, windows, metrics."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
USE_SENSORS = [f"s{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]  # standard 14 informative
RUL_CAP = 125


def load_fd(data_dir, fd="FD002"):
    rd = lambda f: pd.read_csv(f"{data_dir}/{f}_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    train, test = rd("train"), rd("test")
    rul_test = pd.read_csv(f"{data_dir}/RUL_{fd}.txt", header=None)[0].to_numpy()
    return train, test, rul_test


def add_rul(df, cap=RUL_CAP):
    df = df.copy()
    df["rul"] = (df.groupby("unit")["cycle"].transform("max") - df["cycle"]).clip(upper=cap)
    return df


def fit_normaliser(train, n_cond):
    """Cluster op settings into n_cond regimes; sensor mean/std PER regime (train only)."""
    ops = ["op1", "op2", "op3"]
    km = KMeans(n_clusters=n_cond, n_init=10, random_state=0).fit(train[ops]) if n_cond > 1 else None
    cond = km.predict(train[ops]) if km else np.zeros(len(train), int)
    stats = {}
    for c in range(n_cond):
        x = train.loc[cond == c, USE_SENSORS]
        sd = x.std().to_numpy().copy(); sd[sd < 1e-8] = 1.0
        stats[c] = (x.mean().to_numpy(), sd)
    return km, stats


def normalise(df, km, stats):
    cond = km.predict(df[["op1", "op2", "op3"]]) if km else np.zeros(len(df), int)
    X = df[USE_SENSORS].to_numpy(float).copy()
    for c, (mu, sd) in stats.items():
        X[cond == c] = (X[cond == c] - mu) / sd
    out = df.copy()
    out[USE_SENSORS] = X
    return out


def train_windows(df, win=30):
    """All sliding windows, label = RUL at the last cycle. Returns X (N,win,S), y (N,)."""
    Xs, ys = [], []
    for _, g in df.groupby("unit"):
        a, r = g[USE_SENSORS].to_numpy(np.float32), g["rul"].to_numpy(np.float32)
        for i in range(len(g) - win + 1):
            Xs.append(a[i:i + win]); ys.append(r[i + win - 1])
    return np.stack(Xs), np.array(ys, np.float32)


def test_windows(df, rul_true, win=30, cap=RUL_CAP):
    """Last window per test engine (front-padded by repeating the first row if short)."""
    Xs = []
    for _, g in df.groupby("unit"):
        a = g[USE_SENSORS].to_numpy(np.float32)
        if len(a) < win:
            a = np.vstack([np.repeat(a[:1], win - len(a), axis=0), a])
        Xs.append(a[-win:])
    return np.stack(Xs), np.minimum(rul_true, cap).astype(np.float32)


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def nasa_score(pred, true):
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
