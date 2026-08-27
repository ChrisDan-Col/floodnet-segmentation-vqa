import os
import csv
import json
import random
import matplotlib.pyplot as plt
from PIL import Image

DATA_ROOT = r".\Datasets"
QUESTIONS = os.path.join(DATA_ROOT, "FloodNet Challenge - Track 2", "Questions", "Training Question.json")
T2_IMAGES = os.path.join(DATA_ROOT, "FloodNet Challenge - Track 2", "Images")
OUT = r".\prepared_data"

SEED = 42
RATIOS = (0.70, 0.15, 0.15)   # train / val / test
QUESTION_TYPES = ["Simple_Counting", "Complex_Counting", "Condition_Recognition", "Yes_No"]
EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def find_images(root):
    lookup = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(EXTS):
                lookup[os.path.splitext(f)[0]] = os.path.join(dirpath, f)
    return lookup


def show_vqa_examples(rows, images, size=400):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, qtype in zip(axes.ravel(), QUESTION_TYPES):
        candidates = [r for r in rows if r["type"] == qtype and r["image_id"] in images]
        r = random.choice(candidates)
        img = Image.open(images[r["image_id"]]).convert("RGB")
        img.thumbnail((size, size))
        ax.imshow(img)
        ax.set_title("%s\nQ: %s\nGT: %s" % (qtype, r["question"], r["answer"]), fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig_path = os.path.join(OUT, "vqa_examples.png")
    plt.savefig(fig_path, dpi=100)
    print("saved figure: %s" % fig_path)
    plt.show()


os.makedirs(OUT, exist_ok=True)

data = json.load(open(QUESTIONS))
rows = []
for v in data.values():
    rows.append({
        "image_id": os.path.splitext(v["Image_ID"])[0],
        "question": v["Question"].strip(),
        "answer": str(v["Ground_Truth"]).strip(),
        "type": v["Question_Type"],
    })

ids = sorted(set(r["image_id"] for r in rows))
random.Random(SEED).shuffle(ids)
n_train = int(RATIOS[0] * len(ids))
n_val = int(RATIOS[1] * len(ids))
groups = {
    "train": set(ids[:n_train]),
    "val": set(ids[n_train:n_train + n_val]),
    "test": set(ids[n_train + n_val:]),
}

for name, id_set in groups.items():
    subset = [r for r in rows if r["image_id"] in id_set]
    path = os.path.join(OUT, "vqa_%s.csv" % name)
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=["image_id", "question", "answer", "type"], delimiter=";")
        w.writeheader()
        w.writerows(subset)
    print("%s: %d images, %d questions" % (name, len(id_set), len(subset)))

images = find_images(T2_IMAGES)
show_vqa_examples(rows, images)
