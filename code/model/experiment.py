"""Run the C-MAPSS experiment matrix described in the project report.

This script launches independent training processes for each seed/configuration.
It does not claim results until the underlying training jobs finish successfully.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path
import numpy as np

MODELS = ("concat", "independent", "relational")
LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--fds", nargs="+", default=["FD002", "FD004"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--explain", type=int, default=50)
    p.add_argument("--output_dir", default="results/experiment_matrix")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for fd in args.fds:
        win = 21 if fd == "FD002" else 18
        for model in MODELS:
            lambdas = LAMBDAS if model == "relational" else (0.0,)
            for lam in lambdas:
                for seed in args.seeds:
                    tag = f"{fd}_{model}_lambda{lam:g}_seed{seed}"
                    out = root / f"{tag}.json"
                    if args.skip_existing and out.exists():
                        print("Skipping", tag)
                        continue
                    cmd = [
                        sys.executable, "-m", "model.train",
                        "--data_dir", args.data_dir, "--fd", fd,
                        "--model", model, "--phys_lambda", str(lam),
                        "--seed", str(seed), "--epochs", str(args.epochs),
                        "--win", str(win), "--bs", str(args.batch_size),
                        "--n_explain", str(args.explain),
                        "--result_path", str(out),
                    ]
                    print("\nRUN:", " ".join(cmd), flush=True)
                    subprocess.run(cmd, check=True)

    rows = []
    for f in sorted(root.glob("*.json")):
        try:
            r = json.loads(f.read_text())
            r["result_file"] = f.name
            rows.append(r)
        except (OSError, json.JSONDecodeError):
            print("Warning: unreadable result", f)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with (root / "all_seed_results.csv").open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        grouped = {}
        for r in rows:
            key = (r.get("fd"), r.get("model"), r.get("phys_lambda"))
            grouped.setdefault(key, []).append(r)
        summary = []
        for key, vals in sorted(grouped.items(), key=lambda x: str(x[0])):
            row = {"fd": key[0], "model": key[1], "phys_lambda": key[2], "n_seeds": len(vals)}
            for metric in ("rmse", "nasa_score", "attention_iou", "attribution_iou",
                           "attention_chance", "attribution_chance"):
                a = np.array([v[metric] for v in vals if metric in v], dtype=float)
                if len(a):
                    row[metric + "_mean"] = float(a.mean())
                    row[metric + "_sd"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
                    if len(a) > 1:
                        row[metric + "_ci95_low"] = float(a.mean() - 2.776 * a.std(ddof=1) / np.sqrt(len(a)))
                        row[metric + "_ci95_high"] = float(a.mean() + 2.776 * a.std(ddof=1) / np.sqrt(len(a)))
            summary.append(row)
        (root / "summary_mean_sd_ci95.json").write_text(json.dumps(summary, indent=2))
        print(f"Saved {len(rows)} run records under {root}")
    else:
        print("No result files found. No experiments were completed.")


if __name__ == "__main__":
    main()
