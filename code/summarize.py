import json, glob
print(f"{'fd':6}{'model':11}{'lambda':>7}{'RMSE':>9}{'NASA':>12}{'IoU':>8}{'chance':>8}{'z':>7}")
for f in sorted(glob.glob("results/*.json")):
    r = json.load(open(f))
    print(f"{r['fd']:6}{r['model']:11}{r['phys_lambda']:>7}{r['rmse']:>9.2f}{r['nasa_score']:>12.0f}{r['iou']:>8.3f}{r['chance_iou']:>8.3f}{r['z']:>7.1f}")
