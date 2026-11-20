"""
Pilot for the physics-consistency score. Runs with NO real data, on synthetic data,
to prove two things before the real runs:

  Part A  the IoU metric behaves correctly: ~chance for random explanations,
          rises with explanation quality, falls BELOW chance for wrong ones.
  Part B  the whole MetroPT pipeline works end to end (load -> label -> window ->
          chronological split -> reference -> attribution -> IoU).

Real run: replace make_synthetic_csv(...) with your MetroPT-3 csv path.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from scoring.iou import physics_iou, score_batch, null_distribution
from scoring.reference import (METROPT_FAILURES, METROPT_PRIOR, effect_sizes,
                               validated_sensors, sensor_mask, reference_mask)
from scoring.attribution import occlusion_attribution, random_attribution
from data.metropt import (load_metropt, label_failures, make_windows, feature_columns,
                          chronological_split, fit_scaler, apply_scaler, class_weights, plot_eda)

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
SENSORS = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current",
           "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]


# ---------------------------------------------------------------- Part A
def part_a(n_windows=200, S=15, T=60, seed=0):
    rng = np.random.default_rng(seed)
    ref = np.zeros((S, T), bool)
    ref[[1, 4, 6], :] = True                          # 3 reference sensors x all time steps
    wrong = ~ref
    settings = [("random", 0.0, ref), ("weak", 0.5, ref), ("medium", 1.0, ref),
                ("strong", 2.0, ref), ("wrong sensors", 2.0, wrong)]
    rows = []
    for name, snr, target in settings:
        attrs = np.abs(snr * target[None] + rng.normal(size=(n_windows, S, T)))
        res = score_batch(attrs, ref, n_null=300, rng=seed)
        rows.append((name, res["mean_iou"], res["mean_chance"], res["mean_z"]))
    print("\nPART A: metric sanity check (synthetic)")
    print(f"{'explanation':<15}{'mean IoU':>10}{'chance':>9}{'mean z':>9}")
    for r in rows:
        print(f"{r[0]:<15}{r[1]:>10.3f}{r[2]:>9.3f}{r[3]:>9.1f}")
    return rows


# ---------------------------------------------------------------- Part B
def make_synthetic_csv(path, step_min=5, seed=0):
    """Synthetic stand-in with the MetroPT-3 schema. Faults shift 4 known sensors."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-02-01", "2020-08-31", freq=f"{step_min}min")
    X = rng.normal(size=(len(ts), len(SENSORS)))
    df = pd.DataFrame(X, columns=SENSORS)
    df.insert(0, "timestamp", ts)
    for s, e, _ in METROPT_FAILURES:
        m = (df.timestamp >= pd.Timestamp(s)) & (df.timestamp <= pd.Timestamp(e))
        for sensor, shift in [("TP3", -1.5), ("Reservoirs", -1.5), ("Motor_current", 1.2), ("Oil_temperature", 1.0)]:
            df.loc[m, sensor] += shift
    df.to_csv(path, index=True)                       # leaves an "Unnamed: 0" column, like the real file
    return path


def part_b(csv_path, win=24, stride=12, n_explain=15, seed=0):
    df = label_failures(load_metropt(csv_path))
    print("\nPART B: pipeline on", os.path.basename(csv_path))
    print("  median sampling interval:", df.timestamp.diff().median())
    plot_eda(df, cols=SENSORS[:8], out_png=os.path.join(OUT, "metropt_eda.png"))

    X, y, t0, cols = make_windows(df, win=win, stride=stride)
    sp = chronological_split(t0)
    mu, sd = fit_scaler(X[sp["train"]], y[sp["train"]])
    Xs = apply_scaler(X, mu, sd)
    cw = class_weights(y[sp["train"]])
    print(f"  windows: {len(X)}  positives(train): {cw['n_pos']}  positive rate: {cw['positive_rate']:.3%}")
    print(f"  split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in sp.items()))

    # simple model: logistic regression on per-sensor window mean + std
    feats = lambda W: np.concatenate([W.mean(axis=1), W.std(axis=1)], axis=1)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(feats(Xs[sp["train"]]), y[sp["train"]])
    pred = clf.predict(feats(Xs[sp["test"]]))
    ytest = y[sp["test"]]
    tp = int(((pred == 1) & (ytest == 1)).sum()); fp = int(((pred == 1) & (ytest == 0)).sum()); fn = int(((pred == 0) & (ytest == 1)).sum())
    print(f"  test: TP={tp} FP={fp} FN={fn}  (recall {tp/max(tp+fn,1):.2f}, precision {tp/max(tp+fp,1):.2f})")
    predict_fn = lambda W: clf.decision_function(feats(W))

    # reference: domain prior x data check (effect size on TRAIN windows only)
    tr = sp["train"]
    d = effect_sizes(feats(Xs[tr])[:, :len(cols)], y[tr], cols)
    prior = METROPT_PRIOR["primary"] + METROPT_PRIOR["secondary"]
    keep = validated_sensors(prior, d, min_d=0.5)
    print("  effect sizes (prior sensors):", {s: round(d[s], 2) for s in prior if s in d})
    print("  reference sensors (prior AND |d|>=0.5):", keep)
    sel = sensor_mask(cols, keep)

    # explain true-positive TEST windows
    rng = np.random.default_rng(seed)
    cand = sp["test"][(y[sp["test"]] == 1) & (clf.predict(feats(Xs[sp["test"]])) == 1)]
    cand = rng.choice(cand, size=min(n_explain, len(cand)), replace=False)
    occ, rnd, refs = [], [], []
    for i in cand:
        w = Xs[i]                                              # (T, S)
        occ.append(occlusion_attribution(predict_fn, w, baseline=np.zeros(w.shape[1]), block=4))
        rnd.append(random_attribution((w.shape[1], w.shape[0]), rng=int(i)))
        refs.append(reference_mask(w.T, sel))
    res_o, res_r = score_batch(np.array(occ), np.array(refs)), score_batch(np.array(rnd), np.array(refs))
    print(f"  IoU  occlusion: {res_o['mean_iou']:.3f}   random: {res_r['mean_iou']:.3f}   chance: {res_o['mean_chance']:.3f}")
    print(f"  z    occlusion: {res_o['mean_z']:.1f}   random: {res_r['mean_z']:.1f}")
    return res_o, res_r


def make_figure(rows, res_o, res_r, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    names = [r[0] for r in rows]
    ax[0].bar(names, [r[1] for r in rows], color="#2b6cb0")
    ax[0].plot(names, [r[2] for r in rows], "k--", marker="o", label="chance")
    ax[0].set_title("A. IoU vs explanation quality (synthetic)"); ax[0].set_ylabel("mean IoU")
    ax[0].tick_params(axis="x", labelrotation=20); ax[0].legend()
    ax[1].bar(["random", "occlusion"], [res_r["mean_iou"], res_o["mean_iou"]], color=["#a0aec0", "#2b6cb0"])
    ax[1].axhline(res_o["mean_chance"], color="k", ls="--", label="chance"); ax[1].legend()
    ax[1].set_title("B. Pilot on MetroPT-like pipeline (synthetic)"); ax[1].set_ylabel("mean IoU")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    rows = part_a()
    csv = sys.argv[1] if len(sys.argv) > 1 else make_synthetic_csv(os.path.join(OUT, "synthetic_metropt.csv"))
    res_o, res_r = part_b(csv)
    make_figure(rows, res_o, res_r, os.path.join(OUT, "pilot_results.png"))
    print("\nSaved figures to", OUT)
