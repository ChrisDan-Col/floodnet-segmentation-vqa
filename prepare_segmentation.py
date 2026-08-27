import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATA_ROOT = r".\Datasets"
SUP_ROOT = os.path.join(DATA_ROOT, "FloodNet-Supervised_v1.0")
T1_ROOT = os.path.join(DATA_ROOT, "FloodNet Challenge - Track 1")
T2_IMAGES = os.path.join(DATA_ROOT, "FloodNet Challenge - Track 2", "Images")
OUT = r".\prepared_data"

SPLITS = ["train", "val", "test"]
EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

PALETTE = [(0, 0, 0), (255, 0, 0), (180, 120, 120), (160, 150, 20), (140, 140, 140),
           (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0), (60, 180, 75)]


def img_id(name):
    name = os.path.splitext(os.path.basename(name))[0]
    return name[:-4] if name.endswith("_lab") else name


def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


def colorize(mask):
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        rgb[mask == idx] = color
    return rgb


def show_examples(pairs, n=3, size=400):
    n = min(n, len(pairs))
    sample = random.sample(pairs, n)
    fig, axes = plt.subplots(n, 2, figsize=(9, 4 * n))
    for i in range(n):
        image_path, mask_path = sample[i]
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((size, size))
        mask = Image.open(mask_path).resize(img.size, Image.NEAREST)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("input: " + os.path.basename(image_path))
        axes[i, 1].imshow(colorize(np.array(mask)))
        axes[i, 1].set_title("mask: " + os.path.basename(mask_path))
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
    plt.tight_layout()
    fig_path = os.path.join(OUT, "pairing_examples.png")
    plt.savefig(fig_path, dpi=100)
    print("saved figure: %s" % fig_path)
    plt.show()


os.makedirs(OUT, exist_ok=True)

train_pairs = []
test_ids = set()
for split in SPLITS:
    org_dir = os.path.join(SUP_ROOT, split, split + "-org-img")
    lab_dir = os.path.join(SUP_ROOT, split, split + "-label-img")
    masks = {img_id(f): f for f in os.listdir(lab_dir) if f.endswith(EXTS)}

    pairs = []
    for f in sorted(os.listdir(org_dir)):
        if f.endswith(EXTS) and img_id(f) in masks:
            pairs.append((os.path.join(org_dir, f), os.path.join(lab_dir, masks[img_id(f)])))

    save_csv(os.path.join(OUT, "seg_%s.csv" % split), ["image", "mask"], pairs)
    print("%s: %d image-mask pairs" % (split, len(pairs)))
    for image, mask in pairs[:2]:
        print("   %s  <->  %s" % (os.path.basename(image), os.path.basename(mask)))
    if split == "train":
        train_pairs = pairs
    if split == "test":
        test_ids = {img_id(image) for image, _ in pairs}

pool = {}
for root in [SUP_ROOT, T1_ROOT, T2_IMAGES]:
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(EXTS) and not f.endswith("_lab.png") and img_id(f) not in test_ids:
                pool.setdefault(img_id(f), os.path.join(dirpath, f))

save_csv(os.path.join(OUT, "ssl_images.csv"), ["image"], [[p] for p in sorted(pool.values())])
print("ssl pool: %d images (excluded %d segmentation test images)" % (len(pool), len(test_ids)))

show_examples(train_pairs, n=3)
