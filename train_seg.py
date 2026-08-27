import os
import csv
import json
import re
import copy
import math
import time
import random
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageFilter

import utils
from seg_metrics import SegMetrics

try:
    import mlflow
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False


parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", default=r".\prepared_data\seg_train.csv")
parser.add_argument("--val_csv", default=r".\prepared_data\seg_val.csv")
parser.add_argument("--test_csv", default=r".\prepared_data\seg_test.csv")
parser.add_argument("--out_dir", default=r".\experiments\segmentation")
parser.add_argument("--arch", default="vit", choices=["vit", "segformer", "deeplabv3"])
parser.add_argument("--decoder", default="pup", choices=["pup", "naive", "mla"], help="ViT decoder: pup (progressive upsampling), naive (single bilinear), mla (multi-layer aggregation)")
parser.add_argument("--vit_name", default="google/vit-base-patch16-224")
parser.add_argument("--init", default="mae", choices=["mae", "imagenet", "scratch"], help="encoder init (used by --arch vit)")
parser.add_argument("--mae_path", default=r".\experiments\ssl_mae\default\encoder_best")
parser.add_argument("--label_frac", type=float, default=1.0)
parser.add_argument("--img_size", type=int, default=640, help="input size; >224 uses interpolated ViT position embeddings (more detail)")
parser.add_argument("--batch_size", type=int, default=8, help="reduce if the GPU runs out of memory at higher img_size")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=12, help="early-stop patience; higher because the 640px + regularised regime learns slowly")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--dropout", type=float, default=0.1, help="decoder dropout")
parser.add_argument("--label_smoothing", type=float, default=0.0, help="cross-entropy label smoothing; small values (0.05) trim overfitting but too much hurts rare classes")
parser.add_argument("--encoder_dropout", type=float, default=0.0, help="ViT encoder hidden+attention dropout (0.1 = mild); too much underfits small classes")
parser.add_argument("--freeze_layers", type=int, default=0, help="freeze ViT embeddings + first N of 12 blocks (try 3-6): less overfitting AND faster")
parser.add_argument("--class_weights", type=int, default=1, help="1 = frequency-based class weights (capped)")
parser.add_argument("--weight_power", type=float, default=0.5, help="class weight = 1/freq**power; 0.5 = inverse-sqrt (proven best), lower = gentler on rare classes")
parser.add_argument("--weight_cap", type=float, default=5.0, help="max class weight after normalisation")
parser.add_argument("--ignore_bg", type=int, default=1, help="1 = ignore the background class (0) in the loss")
parser.add_argument("--loss", default="combo", choices=["ce", "combo"], help="ce = weighted cross-entropy; combo = CE + Dice")
parser.add_argument("--encoder_lr_scale", type=float, default=0.3, help="encoder lr = lr * this (gentler updates for the pretrained backbone)")
parser.add_argument("--grad_checkpoint", type=int, default=1, help="1 = gradient checkpointing (less memory, allows higher img_size)")
parser.add_argument("--scheduler", default="cosine", choices=["none", "cosine"], help="cosine = warmup + cosine decay (curbs late overfitting)")
parser.add_argument("--warmup_epochs", type=int, default=3)
parser.add_argument("--augment", type=int, default=1, help="1 = data augmentation on the training set")
parser.add_argument("--aug_mode", default="random", choices=["random", "fixed"], help="random = on-the-fly stochastic transforms incl. blur+noise (1x time, best regularisation); fixed = deterministic variants that MULTIPLY the dataset (slower, weaker regularisation)")
parser.add_argument("--aug_mult", type=int, default=4, help="fixed mode only: variants per image from [identity,hflip,vflip,rot90,blur,noise,rot270,bright]. Epoch time scales with this.")
parser.add_argument("--freeze_encoder", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--vis_every", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--test", action="store_true", help="evaluate the run's best model on the test set")
parser.add_argument("--exp_name", default="segmentation")
parser.add_argument("--use_mlflow", type=int, default=0)
parser.add_argument("--agent", action="store_true")
parser.add_argument("--n_trials", type=int, default=4)
parser.add_argument("--llm_name", default="ank028/Llama-3.2-1B-Instruct-medmcqa")
parser.add_argument("--llm_cpu", action="store_true")
args = parser.parse_args()


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 10
CLASS_NAMES = ["background", "building-flooded", "building-non-flooded", "road-flooded",
               "road-non-flooded", "water", "tree", "vehicle", "pool", "grass"]
PALETTE = [(0, 0, 0), (255, 0, 0), (180, 120, 120), (160, 150, 20), (140, 140, 140),
           (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0), (60, 180, 75)]

ABBR = {"batch_size": "bs", "lr": "lr", "weight_decay": "wd", "dropout": "do", "img_size": "img",
        "epochs": "ep", "label_frac": "lf", "init": "init", "arch": "", "decoder": "dec",
        "freeze_encoder": "frozen", "aug_mode": "aug", "aug_mult": "augx", "weight_power": "wp",
        "label_smoothing": "ls", "encoder_dropout": "edo", "freeze_layers": "frz"}
SKIP = {"train_csv", "val_csv", "test_csv", "out_dir", "vit_name", "mae_path", "exp_name",
        "use_mlflow", "resume", "test", "agent", "n_trials", "llm_name", "llm_cpu",
        "num_workers", "vis_every", "seed", "patience", "class_weights", "ignore_bg",
        "scheduler", "warmup_epochs", "augment", "weight_cap",
        "arch", "decoder", "init"}


def run_folder_name(cfg):
    parts = []
    for key, value in sorted(vars(cfg).items()):
        if key in SKIP or value == parser.get_default(key):
            continue
        name = ABBR.get(key, key)
        val = ("%g" % value) if isinstance(value, float) else str(value)
        parts.append("%s%s" % (name, val))
    return "_".join(parts) if parts else "default"


def arch_group(cfg):
    if cfg.arch != "vit":
        return "%s_%s" % (cfg.arch, cfg.init) if cfg.init != "mae" else cfg.arch
    g = "vit_%s" % cfg.decoder
    return g if cfg.init == "mae" else "%s_%s" % (g, cfg.init)


def run_dir_for(cfg):
    return os.path.join(cfg.out_dir, arch_group(cfg), run_folder_name(cfg))


AUG_CATALOG = ["identity", "hflip", "vflip", "rot90", "blur", "noise", "rot270", "bright"]


class SegDataset(data.Dataset):
    def __init__(self, csv_path, img_size, frac=1.0, seed=42, augment=False, aug_mode="fixed", aug_mult=4):
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig"), delimiter=";"))
        if frac < 1.0:
            random.Random(seed).shuffle(rows)
            rows = rows[:max(1, int(frac * len(rows)))]
        self.pairs = [(r["image"], r["mask"]) for r in rows]
        self.img_size = img_size
        self.augment = augment
        self.aug_mode = aug_mode
        self.variants = AUG_CATALOG[:max(1, aug_mult)] if (augment and aug_mode == "fixed") else ["identity"]
        self.norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __len__(self):
        return len(self.pairs) * len(self.variants)

    def _apply_variant(self, img, mask, variant, seed):
        if variant == "hflip":
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        if variant == "vflip":
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        if variant == "rot90":
            return img.transpose(Image.ROTATE_90), mask.transpose(Image.ROTATE_90)
        if variant == "rot270":
            return img.transpose(Image.ROTATE_270), mask.transpose(Image.ROTATE_270)
        if variant == "blur":
            return img.filter(ImageFilter.GaussianBlur(radius=2)), mask
        if variant == "noise":
            arr = np.asarray(img).astype(np.float32) + np.random.RandomState(seed).normal(0, 12, (img.size[1], img.size[0], 3))
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), mask
        if variant == "bright":
            return transforms.functional.adjust_brightness(transforms.functional.adjust_contrast(img, 1.3), 1.2), mask
        return img, mask

    def _random_aug(self, img, mask):
        load = int(self.img_size * 1.15)
        img = img.resize((load, load), Image.BILINEAR)
        mask = mask.resize((load, load), Image.NEAREST)
        x, y = random.randint(0, load - self.img_size), random.randint(0, load - self.img_size)
        box = (x, y, x + self.img_size, y + self.img_size)
        img, mask = img.crop(box), mask.crop(box)
        if random.random() < 0.5:
            img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img, mask = img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        k = random.randint(0, 3)
        if k:
            img, mask = [t.transpose((Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270)[k - 1]) for t in (img, mask)]
        img = self.jitter(img)
        
        if random.random() < 0.15:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.2)))
        if random.random() < 0.15:
            arr = np.asarray(img).astype(np.float32) + np.random.normal(0, random.uniform(4, 10), (self.img_size, self.img_size, 3))
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return img, mask

    def __getitem__(self, index):
        n = len(self.variants)
        base, variant = index // n, self.variants[index % n]
        img = Image.open(self.pairs[base][0]).convert("RGB")
        mask = Image.open(self.pairs[base][1])
        if self.augment and self.aug_mode == "random":
            img, mask = self._random_aug(img, mask)
        else:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            if variant != "identity":
                img, mask = self._apply_variant(img, mask, variant, index)
        img = self.norm(transforms.functional.to_tensor(img))
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


def class_weights(dataset, ignore_bg, power=0.5, cap=5.0, img_size=128):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, mask_path in dataset.pairs:
        m = Image.open(mask_path).resize((img_size, img_size), Image.NEAREST)
        counts += np.bincount(np.array(m).ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    freq = counts / max(counts.sum(), 1)
    w = np.where(freq > 0, 1.0 / np.power(freq + 1e-6, power), 0.0)   # absent classes -> 0
    if ignore_bg:
        w[0] = 0.0
    used = w > 0
    w = w / w[used].mean()
    w = np.clip(w, 0.0, cap)
    return torch.tensor(w, dtype=torch.float32)


def make_vis_batch(dataset, n=4):
    rare = {1, 3, 7, 8}
    scored = []
    for idx, (_, mask_path) in enumerate(dataset.pairs):
        arr = np.array(Image.open(mask_path).resize((128, 128), Image.NEAREST))
        classes = set(np.unique(arr).tolist())
        bg_frac = float((arr == 0).mean())
        scored.append((len(classes & rare), -round(bg_frac, 2), len(classes), idx))
    scored.sort(reverse=True)
    idxs = [t[-1] for t in scored[:n]]
    imgs = torch.stack([dataset[i][0] for i in idxs])
    masks = torch.stack([dataset[i][1] for i in idxs])
    return imgs, masks


def find_mae_encoder(mae_path):
    import glob
    p = os.path.abspath(mae_path)
    if os.path.isdir(p):
        return p
    cwd = os.getcwd()
    patterns = [os.path.join(cwd, "experiments", "ssl_mae", "*", "encoder_best"),
                os.path.join(cwd, "..", "*", "experiments", "ssl_mae", "*", "encoder_best"),
                os.path.join(cwd, "..", "..", "*", "experiments", "ssl_mae", "*", "encoder_best")]
    found = [f for pat in patterns for f in glob.glob(pat) if os.path.isdir(f)]
    if not found:
        return None
    default = [f for f in found if os.path.basename(os.path.dirname(f)) == "default"]
    return os.path.abspath((default or found)[0])


def build_decoder(hidden, dropout, kind):
    if kind == "naive":
        return nn.Sequential(
            nn.Conv2d(hidden, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout), nn.Conv2d(256, NUM_CLASSES, 1))

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
    return nn.Sequential(block(hidden, 256), block(256, 256), block(256, 128), block(128, 64),
                         nn.Dropout2d(dropout), nn.Conv2d(64, NUM_CLASSES, 1))


def tokens_to_grid(tokens):
    tokens = tokens[:, 1:, :]
    b, n, c = tokens.shape
    g = int(n ** 0.5)
    return tokens.transpose(1, 2).reshape(b, c, g, g)


class MLADecoder(nn.Module):
    def __init__(self, hidden, dropout, ch=128):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden, ch, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                          nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)) for _ in range(4)])
        self.aggr = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)) for _ in range(4)])
        self.fuse = nn.Sequential(nn.Conv2d(4 * ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                                  nn.Dropout2d(dropout), nn.Conv2d(ch, NUM_CLASSES, 1))

    def forward(self, feats):
        h = [head(f) for head, f in zip(self.heads, feats)]
        for i in range(len(h) - 2, -1, -1):
            h[i] = h[i] + h[i + 1]
        h = [F.interpolate(a(x), scale_factor=4, mode="bilinear", align_corners=False) for a, x in zip(self.aggr, h)]
        x = self.fuse(torch.cat(h, dim=1)) 
        return F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)  # 56 -> 224


class ViTSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import ViTModel, ViTConfig
        drop = {"hidden_dropout_prob": cfg.encoder_dropout, "attention_probs_dropout_prob": cfg.encoder_dropout}
        if cfg.init == "imagenet":
            self.encoder = ViTModel.from_pretrained(cfg.vit_name, add_pooling_layer=False, **drop)
        else:
            self.encoder = ViTModel(ViTConfig(**drop), add_pooling_layer=False)
            if cfg.init == "mae":
                from transformers import ViTMAEForPreTraining
                mae_path = find_mae_encoder(cfg.mae_path)
                if mae_path is None:
                    raise FileNotFoundError("Could not find a SSL encoder_best folder. Pass --mae_path with its location.")
                mae = ViTMAEForPreTraining.from_pretrained(mae_path)
                self.encoder.load_state_dict(mae.vit.state_dict(), strict=False)
                print("loaded MAE encoder from %s" % mae_path)
        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif cfg.freeze_layers > 0:
            frozen = 0
            for name, p in self.encoder.named_parameters():
                blk = re.search(r"(?:layer|layers|block|blocks)\.(\d+)\.", name.lower())
                if "embed" in name.lower() or (blk and int(blk.group(1)) < cfg.freeze_layers):
                    p.requires_grad = False
                    frozen += 1
            print("froze embeddings + first %d ViT blocks (%d tensors)" % (cfg.freeze_layers, frozen))
        if cfg.grad_checkpoint:
            self.encoder.gradient_checkpointing_enable()
        self.decoder_kind = cfg.decoder
        hidden = self.encoder.config.hidden_size
        self.decoder = MLADecoder(hidden, cfg.dropout) if cfg.decoder == "mla" else build_decoder(hidden, cfg.dropout, cfg.decoder)

    def forward(self, x):
        if self.decoder_kind == "mla":
            hs = self.encoder(pixel_values=x, output_hidden_states=True, interpolate_pos_encoding=True).hidden_states
            feats = [tokens_to_grid(hs[i]) for i in (3, 6, 9, 12)]
            logits = self.decoder(feats)
        else:
            out = self.encoder(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
            logits = self.decoder(tokens_to_grid(out))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class SegFormerSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)

    def forward(self, x):
        logits = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class DeepLabSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50
        from torchvision.models import ResNet50_Weights
        backbone = None if cfg.init == "scratch" else ResNet50_Weights.IMAGENET1K_V1
        self.net = deeplabv3_resnet50(weights=None, weights_backbone=backbone, num_classes=NUM_CLASSES)

    def forward(self, x):
        return self.net(x)["out"]


def build_model(cfg):
    return {"vit": ViTSeg, "segformer": SegFormerSeg, "deeplabv3": DeepLabSeg}[cfg.arch](cfg)


class DiceCELoss(nn.Module):
    def __init__(self, weight, ignore_index, label_smoothing=0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        c = logits.shape[1]
        valid = (target != self.ignore_index).unsqueeze(1).float()
        probs = torch.softmax(logits, dim=1) * valid
        onehot = F.one_hot(target.clamp(min=0), c).permute(0, 3, 1, 2).float() * valid
        dims = (0, 2, 3)
        dice = (2 * (probs * onehot).sum(dims) + 1) / (probs.sum(dims) + onehot.sum(dims) + 1)
        return ce + (1 - dice.mean())


def make_criterion(cfg, weights):
    ignore = 0 if cfg.ignore_bg else -100
    if cfg.loss == "combo":
        return DiceCELoss(weights, ignore, cfg.label_smoothing)
    return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore, label_smoothing=cfg.label_smoothing)


def build_optimizer(model, cfg):
    if hasattr(model, "encoder") and hasattr(model, "decoder") and cfg.encoder_lr_scale != 1.0:
        enc = [p for p in model.encoder.parameters() if p.requires_grad]
        dec = [p for p in model.decoder.parameters() if p.requires_grad]
        groups = [{"params": enc, "lr": cfg.lr * cfg.encoder_lr_scale}, {"params": dec, "lr": cfg.lr}]
        return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    if cfg.scheduler == "none":
        return None
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def plot_seg_curves(history, out_path):
    import matplotlib.pyplot as plt
    epochs = range(1, len(history["train_loss"]) + 1)
    panels = [("loss", [("train_loss", "train"), ("val_loss", "val")]),
              ("mIoU", [("val_miou", "mIoU"), ("val_miou_no_bg", "no-bg")]),
              ("Dice / F1", [("val_dice", "dice")]),
              ("precision", [("val_precision", "precision")]),
              ("recall", [("val_recall", "recall")])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for ax, (title, series) in zip(axes, panels):
        for key, lbl in series:
            ax.plot(epochs, history[key], label=lbl)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(True, alpha=0.3); ax.legend()
    axes[-1].axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=100); plt.close(fig)


def plot_confusion_matrix(cm, class_names, out_path):
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=np.float64)
    disp = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("confusion matrix (row-normalized)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if disp[i, j] >= 0.01:
                ax.text(j, i, "%.2f" % disp[i, j], ha="center", va="center", fontsize=7,
                        color="white" if disp[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(out_path, dpi=100); plt.close(fig)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device):
    model.train()
    meter = utils.AverageMeter()
    for step, (img, mask) in enumerate(loader):
        img, mask = img.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss = criterion(model(img), mask)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        meter.update(loss.item(), img.size(0))
        if step % 20 == 0:
            print("   batch %4d/%d  loss %.4f" % (step, len(loader), loss.item()))
    return meter.avg


def evaluate(model, loader, criterion, device):
    model.eval()
    meter = utils.AverageMeter()
    metrics = SegMetrics(NUM_CLASSES)
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            meter.update(criterion(logits, mask).item(), img.size(0))
            metrics.update(logits.argmax(1).cpu().numpy(), mask.cpu().numpy())
    res = metrics.compute()
    res["val_loss"] = meter.avg
    return res


def colorize(mask):
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        rgb[mask == idx] = color
    return rgb


def save_pred_figure(model, batch, device, out_path):
    try:
        import matplotlib.pyplot as plt
        model.eval()
        img, mask = batch
        with torch.no_grad():
            pred = model(img.to(device)).argmax(1).cpu().numpy()
        mean, std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)
        n = min(4, img.size(0))
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        for i in range(n):
            rgb = (img[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            panels = [(rgb, "input"), (colorize(mask[i].numpy()), "ground truth"), (colorize(pred[i]), "prediction")]
            for j, (im, title) in enumerate(panels):
                ax = axes[i, j] if n > 1 else axes[j]
                ax.imshow(im); ax.set_title(title); ax.axis("off")
        plt.tight_layout(); plt.savefig(out_path, dpi=100); plt.close(fig)
    except Exception as e:
        print("could not save prediction figure: %s" % e)


def safe_write(fn, *fargs, **fkwargs):
    for attempt in range(3):
        try:
            fn(*fargs, **fkwargs)
            return True
        except OSError as e:
            print("file write retry %d/3: %s" % (attempt + 1, e)); time.sleep(1)
    print("skipped a file write after retries (training continues)")
    return False


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2)


def setup_mlflow(cfg, run_name, logger):
    if not (HAS_MLFLOW and bool(cfg.use_mlflow)):
        return False
    try:
        import tempfile
        db = os.path.join(tempfile.gettempdir(), "mlflow_floodnet.db")
        uri = "sqlite:///" + db.replace("\\", "/")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(cfg.exp_name)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(vars(cfg))
        logger.info("to view: mlflow ui --backend-store-uri %s" % uri)
        return True
    except Exception as e:
        logger.info("mlflow disabled (%s)" % e)
        return False


def train_one_experiment(cfg):
    utils.set_seed(cfg.seed)
    run_dir = run_dir_for(cfg)
    os.makedirs(run_dir, exist_ok=True)
    logger = utils.get_logger("seg", os.path.join(run_dir, "train.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s | run dir: %s" % (device, run_dir))
    logger.info("config: %s" % vars(cfg))

    train_set = SegDataset(cfg.train_csv, cfg.img_size, frac=cfg.label_frac, seed=cfg.seed,
                           augment=bool(cfg.augment), aug_mode=cfg.aug_mode, aug_mult=cfg.aug_mult)
    val_set = SegDataset(cfg.val_csv, cfg.img_size)
    pin = torch.cuda.is_available()
    persist = cfg.num_workers > 0
    train_loader = data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                   drop_last=True, pin_memory=pin, persistent_workers=persist)
    val_loader = data.DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                 pin_memory=pin, persistent_workers=persist)
    logger.info("train %d (frac %.2f, aug %s x%d) / val %d | arch %s | init %s" %
                (len(train_set), cfg.label_frac, cfg.aug_mode, len(train_set.variants), len(val_set), cfg.arch, cfg.init))

    weights = class_weights(train_set, cfg.ignore_bg, cfg.weight_power, cfg.weight_cap).to(device) if cfg.class_weights else None
    if weights is not None:
        logger.info("class weights (power %.2f, cap %.1f): %s" % (cfg.weight_power, cfg.weight_cap,
                    ", ".join("%s %.2f" % (n, w) for n, w in zip(CLASS_NAMES, weights.tolist()))))
    model = build_model(cfg).to(device)
    criterion = make_criterion(cfg, weights)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_miou_no_bg": [],
               "val_dice": [], "val_precision": [], "val_recall": []}
    start_epoch, best_miou, no_improve = 0, -1.0, 0
    last_ckpt = os.path.join(run_dir, "last.pt")

    if cfg.resume and os.path.exists(last_ckpt):
        ckpt = utils.load_checkpoint(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt["history"]; start_epoch = ckpt["epoch"]; best_miou = ckpt.get("best_miou", -1.0)
        logger.info("resumed from epoch %d" % start_epoch)

    use_mlflow = setup_mlflow(cfg, run_folder_name(cfg), logger)
    vis_batch = make_vis_batch(val_set, n=4)
    fig_dir = os.path.join(run_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)
    best = {}

    for epoch in range(start_epoch, cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, device)
        res = evaluate(model, val_loader, criterion, device)
        for key, value in [("train_loss", train_loss), ("val_loss", res["val_loss"]),
                           ("val_miou", res["miou"]), ("val_miou_no_bg", res["miou_no_bg"]),
                           ("val_dice", res["mdice"]), ("val_precision", res["mprecision"]), ("val_recall", res["mrecall"])]:
            history[key].append(value)
        logger.info("epoch %3d/%d  train_loss %.4f  val_loss %.4f  mIoU %.4f  mIoU(no-bg) %.4f  F1 %.4f  wF1 %.4f  Prec %.4f  Rec %.4f  pixAcc %.4f" %
                    (epoch + 1, cfg.epochs, train_loss, res["val_loss"], res["miou"], res["miou_no_bg"],
                     res["mf1"], res["wf1"], res["mprecision"], res["mrecall"], res["pixel_acc"]))

        if use_mlflow:
            try:
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": res["val_loss"], "miou": res["miou"],
                                    "dice": res["mdice"], "precision": res["mprecision"], "recall": res["mrecall"]}, step=epoch + 1)
            except Exception:
                use_mlflow = False

        safe_write(plot_seg_curves, history, os.path.join(run_dir, "curves.png"))
        safe_write(utils.save_history_csv, history, os.path.join(run_dir, "history.csv"))
        safe_write(utils.save_checkpoint,
                   {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "history": history, "best_miou": best_miou}, last_ckpt)

        if res["miou"] > best_miou:
            best_miou, no_improve = res["miou"], 0
            best = {"epoch": epoch + 1, **res}
            safe_write(utils.save_checkpoint, {"epoch": epoch + 1, "model": model.state_dict(), "best_miou": best_miou}, os.path.join(run_dir, "best.pt"))
            safe_write(write_json, os.path.join(run_dir, "best_metrics.json"), {"epoch": epoch + 1, "class_names": CLASS_NAMES, **res})
            safe_write(plot_confusion_matrix, res["confusion_matrix"], CLASS_NAMES, os.path.join(run_dir, "confusion_matrix.png"))
        else:
            no_improve += 1

        if (epoch + 1) % cfg.vis_every == 0 or (epoch + 1) == cfg.epochs:
            save_pred_figure(model, vis_batch, device, os.path.join(fig_dir, "pred_epoch_%03d.png" % (epoch + 1)))

        if cfg.patience > 0 and no_improve >= cfg.patience:
            logger.info("early stopping at epoch %d (no mIoU improvement for %d)" % (epoch + 1, cfg.patience))
            break

    logger.info("done. best mIoU %.4f at epoch %d" % (best_miou, best.get("epoch", -1)))
    if use_mlflow:
        try:
            mlflow.end_run()
        except Exception:
            pass
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    bi = int(np.argmax(history["val_miou"]))
    return {"run_dir": run_dir, "best_epoch": bi + 1, "val_miou": history["val_miou"][bi],
            "val_dice": history["val_dice"][bi], "val_miou_no_bg": history["val_miou_no_bg"][bi],
            "train_loss_best": history["train_loss"][bi], "val_loss_best": history["val_loss"][bi]}


def evaluate_on_test(cfg):
    run_dir = run_dir_for(cfg)
    ckpt_path = os.path.join(run_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        print("no best.pt in %s" % run_dir); return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(utils.load_checkpoint(ckpt_path, map_location=device)["model"])
    test_set = SegDataset(cfg.test_csv, cfg.img_size)
    test_loader = data.DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    res = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)

    print("\nTest results (%s, %d images):" % (os.path.basename(run_dir), len(test_set)))
    print("  mIoU %.4f | mIoU(no-bg) %.4f | macroF1 %.4f | weightedF1 %.4f | Prec %.4f | Rec %.4f | pixAcc %.4f" %
          (res["miou"], res["miou_no_bg"], res["mf1"], res["wf1"], res["mprecision"], res["mrecall"], res["pixel_acc"]))
    print("  %-22s %8s %8s %8s %8s" % ("class", "IoU", "F1", "Prec", "Rec"))
    for i, name in enumerate(CLASS_NAMES):
        print("    %-20s %8.4f %8.4f %8.4f %8.4f" %
              (name, res["iou_per_class"][i], res["f1_per_class"][i], res["precision_per_class"][i], res["recall_per_class"][i]))
    write_json(os.path.join(run_dir, "test_metrics.json"), {"class_names": CLASS_NAMES, **res})
    plot_confusion_matrix(res["confusion_matrix"], CLASS_NAMES, os.path.join(run_dir, "confusion_matrix_test.png"))
    print("  saved test_metrics.json and confusion_matrix_test.png")


TUNABLE = ["lr", "weight_decay", "dropout", "encoder_dropout", "label_smoothing", "weight_power"]
GRID = {
    "lr": [5e-5, 1e-4, 1.5e-4, 2e-4, 3e-4],
    "weight_decay": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "encoder_dropout": [0.0, 0.1, 0.2],
    "label_smoothing": [0.0, 0.05, 0.1],
    "weight_power": [0.25, 0.35, 0.5],
}
INT_KEYS = set()
FALLBACK = [{"lr": 2e-4, "weight_decay": 1e-3},
            {"lr": 5e-5, "dropout": 0.2, "encoder_dropout": 0.1},
            {"lr": 3e-4, "weight_decay": 3e-4},
            {"lr": 1.5e-4, "label_smoothing": 0.05, "weight_power": 0.35},
            {"lr": 5e-5, "encoder_dropout": 0.1, "dropout": 0.2},
            {"lr": 2e-4, "weight_decay": 3e-3}]


def snap(key, val):
    g = min(GRID[key], key=lambda x: abs(x - float(val)))
    return int(g) if key in INT_KEYS else float(g)


def clean_proposal(raw):
    out = {}
    for k, v in raw.items():
        if k in TUNABLE:
            try:
                out[k] = snap(k, v)
            except (TypeError, ValueError):
                pass
    return out


def min_diff_to_past(cfg, past):
    return min((sum(cfg[k] != p[k] for k in TUNABLE) for p in past), default=len(TUNABLE))


def diversify(cfg, past):
    cfg = dict(cfg)
    priority = ["lr", "weight_power", "label_smoothing", "encoder_dropout", "weight_decay", "dropout"]
    for _ in range(30):
        if cfg not in past and min_diff_to_past(cfg, past) >= 2:
            return cfg
        moved = False
        for k in priority:
            blocked = {p[k] for p in past} | {cfg[k]}
            opts = [v for v in GRID[k] if v not in blocked]
            if opts:
                cfg[k] = min(opts, key=lambda v: abs(v - cfg[k]))
                moved = True
                break
        if not moved:
            for k in priority:
                alt = [v for v in GRID[k] if v != cfg[k]]
                if alt:
                    cfg[k] = alt[0]
                    break
    return cfg


class SegAgent:
    def __init__(self, llm_name, force_cpu=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("loading reasoning LLM: %s" % llm_name)
        self.device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, dtype=dtype).to(self.device)

    def decide_next_config(self, history, step):
        if step == 1:
            return {}

        best = max(history, key=lambda h: h["val_miou"])
        best_cfg = {k: best[k] for k in TUNABLE}
        past = [{k: h[k] for k in TUNABLE} for h in history]
        tried = {k: {h[k] for h in history} for k in TUNABLE}
        summary = [{"exp": h["exp_id"], "lr": h["lr"], "wd": h["weight_decay"], "do": h["dropout"],
                    "enc_do": h["encoder_dropout"], "ls": h["label_smoothing"], "wpow": h["weight_power"],
                    "val_miou": round(h["val_miou"], 4), "train_loss": round(h["train_loss_best"], 4),
                    "val_loss": round(h["val_loss_best"], 4),
                    "gap": round(h["val_loss_best"] - h["train_loss_best"], 4)} for h in history]
        best_str = ", ".join("%s=%g" % (k, best_cfg[k]) for k in TUNABLE)
        grid_str = "\n".join("  %-16s %s" % (k + ":", ", ".join(("%g*" % v if v not in tried[k] else "%g" % v) for v in GRID[k]))
                             for k in ["lr", "weight_decay", "dropout", "encoder_dropout", "label_smoothing", "weight_power"])

        base_prompt = """You run a hyperparameter SEARCH for a ViT segmentation model on FloodNet (imbalanced aerial, 640px). GOAL: maximise val_miou. Only a few experiments are possible, so each one must EXPLORE a different region - timid tweaks teach us nothing.

Experiments so far (best epoch). "gap" = val_loss - train_loss.
%s

BEST so far (val_miou=%.4f):
  %s

Rules for the next experiment:
1. Make it MEANINGFULLY different from every past run: change 2 to 4 knobs.
2. Learning rate has barely moved - you MUST try an lr value not used yet.
3. Do NOT just nudge weight_decay/dropout by one step.
4. Still reason from the history: big gap + low train_loss -> add regularisation; high train_loss or stalled val_miou -> reduce regularisation or change lr. Too much regularisation collapses the rare classes (vehicle, pool).

Knobs and ALLOWED values (values NOT yet tried are marked *, prefer those):
%s

Use only allowed values, never repeat a past config, and make it clearly different.
Reply with ONLY a JSON dict of the knobs you change, e.g. {"lr": 0.0002, "weight_decay": 0.001, "dropout": 0.2}""" % (
            json.dumps(summary, indent=2), best["val_miou"], best_str, grid_str)

        prompt = base_prompt
        for attempt in range(3):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.llm.generate(**inputs, max_new_tokens=180, do_sample=True,
                                        temperature=0.9 + 0.15 * attempt, top_p=0.95, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print("LLM proposal (attempt %d): %s" % (attempt + 1, response.strip()[:200]))
            try:
                proposed = clean_proposal(json.loads(re.search(r"\{[^{}]*\}", response).group(0)))
                merged = diversify({**best_cfg, **proposed}, past)
                print("  -> config: %s" % {k: merged[k] for k in TUNABLE if merged[k] != best_cfg[k]})
                return merged
            except Exception as e:
                print("could not parse (%s), retrying" % e)
                prompt = base_prompt + "\n\nReply with ONLY a JSON dict, e.g. {\"lr\": 0.0002, \"weight_decay\": 0.001}"

        print("using diverse deterministic fallback")
        off = (step - 2) % len(FALLBACK)
        for pert in FALLBACK[off:] + FALLBACK[:off]:
            cand = diversify({**best_cfg, **pert}, past)
            if cand not in past:
                return cand
        return diversify(dict(best_cfg), past)


def append_experiment_row(path, row):
    for attempt in range(3):
        try:
            exists = os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8-sig") as fp:
                w = csv.DictWriter(fp, fieldnames=list(row.keys()), delimiter=";")
                if not exists:
                    w.writeheader()
                w.writerow(row)
            return True
        except OSError as e:
            print("write failed (%d/3): %s" % (attempt + 1, e)); time.sleep(1)
    return False


def select_best(records):
    if len(records) == 1:
        return records[0]
    mious = [r["val_miou"] for r in records]
    dices = [r["val_dice"] for r in records]

    def norm(vals, v):
        lo, hi = min(vals), max(vals)
        return 1.0 if hi == lo else (v - lo) / (hi - lo)

    for r in records:
        r["score"] = 0.6 * norm(mious, r["val_miou"]) + 0.4 * norm(dices, r["val_dice"])
    return max(records, key=lambda r: r["score"])


def run_agent():
    agent = SegAgent(args.llm_name, force_cpu=args.llm_cpu)
    history = []
    group_dir = os.path.join(args.out_dir, arch_group(args))
    os.makedirs(group_dir, exist_ok=True)
    summary_csv = os.path.join(group_dir, "experiments_history.csv")

    for step in range(1, args.n_trials + 1):
        print("\nTrial %d/%d" % (step, args.n_trials))
        config = agent.decide_next_config(history, step)
        print("next config: %s" % ("defaults (baseline)" if not config else config))
        cfg = copy.deepcopy(args); cfg.agent = False
        for k, v in config.items():
            setattr(cfg, k, type(getattr(args, k))(v))
        try:
            m = train_one_experiment(cfg)
        except Exception as e:
            print("trial %d failed (%s), continuing" % (step, e)); continue
        record = {"exp_id": "trial_%d" % step, "folder": os.path.basename(m["run_dir"]),
                  "lr": cfg.lr, "batch_size": cfg.batch_size, "weight_decay": cfg.weight_decay,
                  "dropout": cfg.dropout, "weight_power": cfg.weight_power, "encoder_dropout": cfg.encoder_dropout,
                  "freeze_layers": cfg.freeze_layers, "label_smoothing": cfg.label_smoothing,
                  "best_epoch": m["best_epoch"], "val_miou": m["val_miou"],
                  "val_miou_no_bg": m["val_miou_no_bg"], "val_dice": m["val_dice"],
                  "train_loss_best": m["train_loss_best"], "val_loss_best": m["val_loss_best"],
                  "run_dir": m["run_dir"]}
        history.append(record)
        append_experiment_row(summary_csv, record)
        print("result: %s | mIoU %.4f | mIoU(no-bg) %.4f | Dice %.4f" %
              (record["folder"], record["val_miou"], record["val_miou_no_bg"], record["val_dice"]))

    if not history:
        print("no trials completed"); return
    best = select_best(history)
    print("\nRanking (best first):")
    for r in sorted(history, key=lambda x: x["score"], reverse=True):
        print("  %-30s score %.3f | mIoU %.4f | Dice %.4f" % (r["folder"], r["score"], r["val_miou"], r["val_dice"]))
    with open(os.path.join(group_dir, "best_experiment.txt"), "w", encoding="utf-8") as fp:
        fp.write("Best: %s\nFolder: %s\nmIoU=%.4f mIoU(no-bg)=%.4f Dice=%.4f\n" %
                 (best["folder"], best["run_dir"], best["val_miou"], best["val_miou_no_bg"], best["val_dice"]))
    print("\nbest model: %s (mIoU %.4f)" % (best["folder"], best["val_miou"]))


def main():
    if args.test:
        evaluate_on_test(args)
    elif args.agent:
        run_agent()
    else:
        train_one_experiment(args)


if __name__ == "__main__":
    main()
