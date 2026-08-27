# Compare experiments without MLflow, from each run's history.csv + best_metrics.json.
# Three views:
#   python compare_experiments.py --stage seg                 # every run, every group
#   python compare_experiments.py --stage seg --group vit_pup_5   # runs of one group
#   python compare_experiments.py --stage seg --best          # the BEST run of each
#         architecture, compared across architectures (for the thesis report)
#
# It plots all curves (train_loss, val_loss, mIoU, mIoU-no-bg, Dice, precision, recall),
# prints a best-epoch summary, and in --best also writes summary + per-class tables (CSV).

import os
import re
import csv
import glob
import json
import math
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["background", "building-flooded", "building-non-flooded", "road-flooded",
               "road-non-flooded", "water", "tree", "vehicle", "pool", "grass"]

parser = argparse.ArgumentParser()
parser.add_argument("--stage", default="seg", choices=["ssl", "seg"])
parser.add_argument("--group", default=None, help="restrict to one folder (e.g. vit_pup_5)")
parser.add_argument("--best", action="store_true", help="seg only: compare the BEST run of each architecture across architectures")
args = parser.parse_args()

if args.stage == "ssl":
    EXP_ROOT = r".\experiments\ssl_mae"
    CURVES = ["train_loss", "val_loss", "val_psnr", "val_ssim"]
    BEST_KEY, BEST_MAX = "val_loss", False
else:
    EXP_ROOT = r".\experiments\segmentation"
    CURVES = ["train_loss", "val_loss", "val_miou", "val_miou_no_bg", "val_dice", "val_precision", "val_recall"]
    BEST_KEY, BEST_MAX = "val_miou", True


def load_history(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig"), delimiter=";"))
    h = {"epoch": [int(r["epoch"]) for r in rows]}
    for m in CURVES:
        h[m] = [float(r[m]) for r in rows if r.get(m) not in (None, "")]
    return h


def best_idx(h):
    pick = max if BEST_MAX else min
    return pick(range(len(h[BEST_KEY])), key=lambda i: h[BEST_KEY][i])


def plot_curves(runs, out_path, title=None):
    cols = 4
    rows = math.ceil(len(CURVES) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.ravel()
    for ax, m in zip(axes, CURVES):
        for name, h in sorted(runs.items()):
            if h.get(m):
                ax.plot(h["epoch"][:len(h[m])], h[m], label=name)
        ax.set_title(m); ax.set_xlabel("epoch"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    for ax in axes[len(CURVES):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print("saved figure: %s" % out_path)


def print_summary(runs, name_header="experiment"):
    show = [c for c in CURVES if c != "train_loss"]
    print("\n%-26s %5s  %8s  %s" % (name_header, "ep*", "gap", "  ".join("%10s" % m.replace("val_", "") for m in show)))
    table = []
    for name, h in sorted(runs.items()):
        bi = best_idx(h)
        gap = h["val_loss"][bi] - h["train_loss"][bi]
        vals = "  ".join("%10.4f" % h[m][bi] for m in show)
        print("%-26s %5d  %8.3f  %s" % (name, h["epoch"][bi], gap, vals))
        table.append([name, h["epoch"][bi], round(gap, 4)] + [round(h[m][bi], 4) for m in show])
    return ["experiment", "best_epoch", "gap"] + [m.replace("val_", "") for m in show], table


def find_best_per_arch():
    groups = {}
    for d in sorted(glob.glob(os.path.join(EXP_ROOT, "*"))):
        base = os.path.basename(d)
        if not os.path.isdir(d) or "first_try" in base or base.startswith("_"):
            continue
        groups.setdefault(re.sub(r"_\d+$", "", base), []).append(d)

    runs, metrics = {}, {}
    for key, dirs in sorted(groups.items()):
        best = None
        for d in dirs:
            # match best_metrics.json in a run subfolder (vit_pup_5/run/) OR directly in the
            # folder (hybrid_4/), so the hybrid runs are included too
            for bm in glob.glob(os.path.join(d, "**", "best_metrics.json"), recursive=True):
                m = json.load(open(bm))
                if best is None or m["miou"] > best[0]:
                    best = (m["miou"], os.path.dirname(bm), m)
        if best:
            runs[key] = load_history(os.path.join(best[1], "history.csv"))
            metrics[key] = best[2]
            print("  %-22s best run: %s" % (key, os.path.basename(best[1])))
    return runs, metrics


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp, delimiter=";")
        w.writerow(header)
        w.writerows(rows)
    print("saved table: %s" % path)


def per_class_tables(metrics, out_dir):
    archs = list(metrics.keys())
    for key, field in [("iou", "iou_per_class"), ("f1", "f1_per_class"),
                       ("precision", "precision_per_class"), ("recall", "recall_per_class")]:
        rows = [[CLASS_NAMES[i]] + [round(metrics[a][field][i], 4) for a in archs] for i in range(len(CLASS_NAMES))]
        write_csv(os.path.join(out_dir, "per_class_%s.csv" % key), ["class"] + archs, rows)
    print("\nPer-class IoU:")
    print("%-22s %s" % ("class", "  ".join("%10s" % a[:10] for a in archs)))
    for i, cname in enumerate(CLASS_NAMES):
        print("%-22s %s" % (cname, "  ".join("%10.3f" % metrics[a]["iou_per_class"][i] for a in archs)))


if args.best and args.stage == "seg":
    print("Best run of each architecture:")
    runs, metrics = find_best_per_arch()
    if not runs:
        print("No runs found under %s" % EXP_ROOT); raise SystemExit
    out_dir = os.path.join(EXP_ROOT, "_best_comparison")
    os.makedirs(out_dir, exist_ok=True)
    plot_curves(runs, os.path.join(out_dir, "curves.png"), title="best model per architecture")
    header, table = print_summary(runs, name_header="architecture")
    # add the macro metrics from best_metrics.json (F1, pixel_acc) to the summary CSV
    full_header = ["architecture", "best_epoch", "gap", "mIoU", "mIoU_no_bg", "Dice",
                   "macroF1", "weightedF1", "precision", "recall", "pixAcc"]
    full_rows = []
    for name in sorted(runs.keys()):
        h = runs[name]; bi = best_idx(h); m = metrics[name]
        full_rows.append([name, h["epoch"][bi], round(h["val_loss"][bi] - h["train_loss"][bi], 4),
                          round(m["miou"], 4), round(m["miou_no_bg"], 4), round(m["mdice"], 4),
                          round(m["mf1"], 4), round(m.get("wf1", 0), 4), round(m["mprecision"], 4),
                          round(m["mrecall"], 4), round(m["pixel_acc"], 4)])
    write_csv(os.path.join(out_dir, "summary.csv"), full_header, full_rows)
    per_class_tables(metrics, out_dir)
else:
    EXP_DIR = os.path.join(EXP_ROOT, args.group) if args.group else EXP_ROOT
    runs = {}
    for hp in glob.glob(os.path.join(EXP_DIR, "**", "history.csv"), recursive=True):
        runs[os.path.relpath(os.path.dirname(hp), EXP_DIR).replace("\\", "/")] = load_history(hp)
    if not runs:
        print("No history.csv found under %s" % EXP_DIR); raise SystemExit
    plot_curves(runs, os.path.join(EXP_DIR, "comparison.png"))
    print_summary(runs)
