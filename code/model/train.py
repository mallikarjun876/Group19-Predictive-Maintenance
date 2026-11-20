"""Train/evaluate one C-MAPSS configuration and score two explanation maps.

Example:
  python -m model.train --data_dir ../4_dataset/CMAPSS --fd FD002 --model relational --seed 0
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from model.cmapss import (load_fd, add_rul, fit_normaliser, normalise, train_windows,
                          test_windows, USE_SENSORS, RUL_CAP, rmse, nasa_score)
from model.nets import Concat, Relational, Independent, physics_loss
from scoring.reference import trend_sensors, sensor_mask
from scoring.iou import score_batch
from scoring.attribution import grad_x_input


def seed_everything(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--fd", choices=["FD001", "FD002", "FD003", "FD004"], default="FD002")
    ap.add_argument("--model", choices=["concat", "independent", "relational"], default="concat")
    ap.add_argument("--phys_lambda", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--win", type=int, default=None)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_explain", type=int, default=50)
    ap.add_argument("--result_path", default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    if a.phys_lambda < 0:
        ap.error("--phys_lambda must be non-negative")
    if a.phys_lambda and a.model != "relational":
        ap.error("physics loss is only available for --model relational")
    win = a.win or (21 if a.fd in ("FD002", "FD003") else 18)
    seed_everything(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, te, rul_te = load_fd(a.data_dir, a.fd)
    n_cond = 6 if a.fd in ("FD002", "FD004") else 1
    km, stats = fit_normaliser(tr, n_cond)
    tr = add_rul(tr)
    tr = normalise(tr, km, stats)
    te = normalise(te, km, stats)
    Xtr, ytr = train_windows(tr, win)
    Xte, yte = test_windows(te, rul_te, win)
    if len(Xtr) == 0 or len(Xte) == 0:
        raise RuntimeError("No windows were created. Check the dataset files and window length.")

    # The reference is derived from training trajectories only, never test labels.
    ref_names, ref_scores = trend_sensors(
        tr[USE_SENSORS].to_numpy(), tr["rul"].to_numpy(), USE_SENSORS, top_k=6)
    ref_sel = sensor_mask(USE_SENSORS, ref_names)
    ref_mask = np.broadcast_to(ref_sel[:, None], (len(USE_SENSORS), win)).copy()

    models = {"concat": Concat, "independent": Independent, "relational": Relational}
    net = models[a.model](Xtr.shape[2]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=a.lr)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32)
    yt = torch.as_tensor(ytr / RUL_CAP, dtype=torch.float32)
    ref_t = torch.as_tensor(ref_sel, dtype=torch.bool, device=device)

    for epoch in range(a.epochs):
        net.train()
        order = torch.randperm(len(Xt))
        total = 0.0
        for start in range(0, len(Xt), a.bs):
            ids = order[start:start + a.bs]
            xb, yb = Xt[ids].to(device), yt[ids].to(device)
            pred, importance = net(xb)
            loss = nn.functional.mse_loss(pred, yb)
            if a.phys_lambda:
                if importance is None:
                    raise RuntimeError("Selected model does not expose sensor attention.")
                loss = loss + a.phys_lambda * physics_loss(importance, ref_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        print(f"epoch {epoch + 1:03d}/{a.epochs}: loss={total / len(Xt):.6f}")

    net.eval()
    x_test = torch.as_tensor(Xte, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_scaled, importance = net(x_test)
        pred = pred_scaled.detach().cpu().numpy() * RUL_CAP

    n = min(max(a.n_explain, 1), len(Xte))
    x_explain = x_test[:n]
    class PredictionOnly(nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model
        def forward(self, x):
            return self.model(x)[0].unsqueeze(-1)

    # Gradient x input attributions (model-output sensitivity).
    attrs = grad_x_input(PredictionOnly(net), x_explain)
    attr_scores = score_batch(attrs, ref_mask, n_null=500, rng=a.seed)

    # Attention importance is a separate explanation type. It exists only for
    # the relational architecture; do not fabricate it for the other baselines.
    attention_scores = None
    if importance is not None:
        attention = importance[:n].detach().cpu().numpy()
        attention_scores = score_batch(attention, ref_mask, n_null=500, rng=a.seed)

    result = {
        "fd": a.fd, "model": a.model, "phys_lambda": a.phys_lambda,
        "seed": a.seed, "epochs": a.epochs, "window": win,
        "n_train_windows": int(len(Xtr)), "n_test_engines": int(len(Xte)),
        "reference_sensors_train_derived": ref_names,
        "reference_sensor_spearman_abs": ref_scores,
        "rmse": rmse(pred, yte), "nasa_score": nasa_score(pred, yte),
        "attribution_iou": attr_scores["mean_iou"],
        "attribution_chance": attr_scores["mean_chance"],
        "attribution_z": attr_scores["mean_z"],
        "attribution_n": int(n),
    }
    if attention_scores is not None:
        result.update({
            "attention_iou": attention_scores["mean_iou"],
            "attention_chance": attention_scores["mean_chance"],
            "attention_z": attention_scores["mean_z"],
        })
    else:
        result.update({"attention_iou": None, "attention_chance": None, "attention_z": None})
    print(json.dumps(result, indent=2))

    target = Path(a.result_path) if a.result_path else Path("results") / (
        f"{a.fd}_{a.model}_lambda{a.phys_lambda:g}_seed{a.seed}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print("Saved:", target)


if __name__ == "__main__":
    main()
