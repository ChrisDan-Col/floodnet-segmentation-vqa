# Quick visual check on a few random masks:
#   python translate_masks.py --masks_csv .\prepared_data\seg_val.csv --sample 8 --show
# Oracle check against the real VQA answers (join all masks, translate only the VQA images):
#   python translate_masks.py --masks_csv .\prepared_data\seg_train.csv,.\prepared_data\seg_val.csv,.\prepared_data\seg_test.csv --vqa_csv .\prepared_data\vqa_val.csv

import os
import csv
import json
import glob
import random
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

parser = argparse.ArgumentParser()
parser.add_argument("--masks_csv", default=None, help="one or more (comma-separated) 'image;mask' CSVs -> ground-truth masks")
parser.add_argument("--masks_dir", default=None, help="folder of predicted mask PNGs named <image_id>.png")
parser.add_argument("--out", default=r".\prepared_data\facts_gt.json")
parser.add_argument("--vqa_csv", default=None, help="translate only the images in this vqa_*.csv and check the counts against the real answers")
parser.add_argument("--sample", type=int, default=0, help="translate only N random masks (0 = all)")
parser.add_argument("--show", action="store_true", help="save a figure with the sampled image + mask + facts")
parser.add_argument("--size", type=int, default=1024, help="working resolution for counting; higher = fewer merged buildings but slower")
parser.add_argument("--split", default="none", choices=["none", "watershed"], help="connected components (none) counts best here; watershed over-segments irregular buildings")
parser.add_argument("--seed", type=int, default=None, help="fix for a reproducible sample; default = different each run")
args = parser.parse_args()

CLASS_NAMES = ["background", "building-flooded", "building-non-flooded", "road-flooded",
               "road-non-flooded", "water", "tree", "vehicle", "pool", "grass"]
PALETTE = [(0, 0, 0), (255, 0, 0), (180, 120, 120), (160, 150, 20), (140, 140, 140),
           (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0), (60, 180, 75)]
BUILDING_FLOODED, BUILDING_NON_FLOODED = 1, 2
ROAD_FLOODED, ROAD_NON_FLOODED = 3, 4
WATER, VEHICLE, POOL = 5, 7, 8
FLOODED_CLASSES = (BUILDING_FLOODED, ROAD_FLOODED, WATER)

FLOOD_THRESHOLD = 0.30
MIN_AREA = max(10, int(3e-5 * args.size * args.size))
MIN_DIST = max(8, args.size // 40)


def count_blobs(mask, class_id):
    binary = mask == class_id
    if not binary.any():
        return 0
    if args.split == "none":
        blobs, n = ndimage.label(binary, structure=np.ones((3, 3)))
        sizes = ndimage.sum(np.ones_like(mask), blobs, index=range(1, n + 1))
        return int((sizes >= MIN_AREA).sum())
    dist = ndimage.distance_transform_edt(binary)
    peaks = peak_local_max(dist, min_distance=MIN_DIST, labels=binary, exclude_border=False)
    if len(peaks) == 0:
        return 0
    markers = np.zeros(binary.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    regions = watershed(-dist, markers, mask=binary)
    sizes = ndimage.sum(np.ones_like(mask), regions, index=range(1, int(regions.max()) + 1))
    return int((sizes >= MIN_AREA).sum())


def road_state(mask):
    flooded = (mask == ROAD_FLOODED).sum() > 0
    dry = (mask == ROAD_NON_FLOODED).sum() > 0
    if flooded and dry:
        return "flooded and non-flooded"
    if flooded:
        return "flooded"
    if dry:
        return "non-flooded"
    return "no road"


def mask_to_facts(mask):
    bf, bnf = count_blobs(mask, BUILDING_FLOODED), count_blobs(mask, BUILDING_NON_FLOODED)
    non_bg = int((mask > 0).sum())
    pct = float(np.isin(mask, FLOODED_CLASSES).sum()) / non_bg if non_bg else 0.0
    facts = {
        "buildings_flooded": bf, "buildings_non_flooded": bnf, "buildings_total": bf + bnf,
        "vehicles": count_blobs(mask, VEHICLE), "pools": count_blobs(mask, POOL),
        "pct_flooded": round(pct, 3), "road_state": road_state(mask),
        "condition": "flooded" if pct >= FLOOD_THRESHOLD else "non-flooded",
    }
    facts["facts_text"] = (
        "flooded buildings: %d; non-flooded buildings: %d; total buildings: %d; "
        "flooded area: %d%% of the scene; road: %s; overall condition: %s" % (
            facts["buildings_flooded"], facts["buildings_non_flooded"], facts["buildings_total"],
            round(facts["pct_flooded"] * 100), facts["road_state"], facts["condition"]))
    return facts


def image_id(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name[:-4] if name.endswith("_lab") else name


def read_mask(path):
    m = np.array(Image.open(path).resize((args.size, args.size), Image.NEAREST))
    return m[:, :, 0] if m.ndim == 3 else m


def colorize(mask):
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        rgb[mask == idx] = color
    return rgb


def build_lookup():
    lookup = {}
    if args.masks_csv:
        for path in args.masks_csv.split(","):
            for r in csv.DictReader(open(path.strip(), encoding="utf-8-sig"), delimiter=";"):
                lookup[image_id(r["image"])] = (r["image"], r["mask"])
    elif args.masks_dir:
        for p in glob.glob(os.path.join(args.masks_dir, "*.png")):
            lookup[image_id(p)] = (None, p)
    else:
        raise SystemExit("pass --masks_csv (ground-truth) or --masks_dir (predicted)")
    return lookup


def interest(f):
    s = f["buildings_flooded"] * 5 + f["pct_flooded"] * 10 + f["buildings_total"]
    if "flooded" in f["road_state"]:
        s += 3
    if f["condition"] == "flooded":
        s += 3
    return s


def sample_examples(lookup):
    ids = list(lookup)
    random.Random(args.seed).shuffle(ids)
    cap = min(len(ids), max(30, args.sample * 5))
    read, facts = [], {}
    for iid in ids[:cap]:
        _, mask_path = lookup[iid]
        if not os.path.exists(mask_path):
            continue
        mask = read_mask(mask_path)
        facts[iid] = mask_to_facts(mask)
        read.append((iid, lookup[iid][0], mask, facts[iid]))
    read.sort(key=lambda e: interest(e[3]), reverse=True)
    return read[:args.sample], facts


def save_figure(examples):
    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(11, 4 * n))
    for row, (iid, img_path, mask, facts) in enumerate(examples):
        ax_img, ax_mask = (axes[row, 0], axes[row, 1]) if n > 1 else (axes[0], axes[1])
        cmask = Image.fromarray(colorize(mask))
        if img_path and os.path.exists(img_path):
            im = Image.open(img_path).convert("RGB"); im.thumbnail((400, 400))
            ax_img.imshow(im)
            cmask = cmask.resize(im.size, Image.NEAREST)
        ax_img.set_title(iid); ax_img.axis("off")
        ax_mask.imshow(cmask)
        ax_mask.set_title(facts["facts_text"], fontsize=7, wrap=True); ax_mask.axis("off")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(args.out), "facts_examples.png")
    plt.savefig(out, dpi=100); plt.close(fig)
    print("saved figure: %s" % out)


lookup = build_lookup()

if args.sample > 0:
    examples, facts = sample_examples(lookup)
    print("sampled %d masks (prefer flooded scenes), size %d, split %s" % (len(examples), args.size, args.split))
    print("\nExamples:")
    for iid, _, _, f in examples:
        print("   %s -> %s" % (iid, f["facts_text"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(facts, open(args.out, "w"), indent=2)
    if args.show:
        save_figure(examples)
    raise SystemExit

if args.vqa_csv:
    want = set()
    for p in args.vqa_csv.split(","):
        want |= {r["image_id"] for r in csv.DictReader(open(p.strip(), encoding="utf-8-sig"), delimiter=";")}
    ids = [i for i in lookup if i in want]
else:
    ids = list(lookup)
print("translating %d masks (size %d, split %s)" % (len(ids), args.size, args.split))
facts = {}
for i, iid in enumerate(ids):
    _, mask_path = lookup[iid]
    if not os.path.exists(mask_path):
        continue
    facts[iid] = mask_to_facts(read_mask(mask_path))
    if (i + 1) % 100 == 0:
        print("  %d/%d" % (i + 1, len(ids)))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(facts, open(args.out, "w"), indent=2)
print("saved %d fact records -> %s" % (len(facts), args.out))

print("\nExamples:")
for iid in list(facts)[:min(8, len(facts))]:
    print("   %s -> %s" % (iid, facts[iid]["facts_text"]))

if args.vqa_csv:
    rows = []
    for p in args.vqa_csv.split(","):
        rows += list(csv.DictReader(open(p.strip(), encoding="utf-8-sig"), delimiter=";"))
    total, correct = 0, 0
    for r in rows:
        if "Counting" not in r["type"] or "building" not in r["question"].lower() or r["image_id"] not in facts:
            continue
        try:
            answer = int(float(r["answer"]))
        except ValueError:
            continue
        q = r["question"].lower()
        key = "buildings_non_flooded" if ("non" in q and "flood" in q) else ("buildings_flooded" if "flood" in q else "buildings_total")
        total += 1
        correct += (facts[r["image_id"]][key] == answer)
    if total:
        print("\nCheck against real answers (%d building-counting questions): %d/%d = %.1f%%" %
              (total, correct, total, 100 * correct / total))
