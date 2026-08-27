# HybridSeg: the custom segmentation model. Fixed ResNet-50 encoder (dilated, OS16) plus one optional
# Transformer block for global context, with a swappable decoder head (the "classifier").
# The split is the official FloodNet train/val/test, so test is the paper's test; run --test for the comparison to PSPNet.
#
#   python train_hybrid.py                       # the recommended model (writes to hybrid_deeplabv3plus)
#   python train_hybrid.py --test                # evaluate the best checkpoint on TEST (the paper's split, TTA on)

import os
import csv
import json
import time
import math
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

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", default=r".\prepared_data\seg_train.csv")
parser.add_argument("--val_csv", default=r".\prepared_data\seg_val.csv")
parser.add_argument("--test_csv", default=r".\prepared_data\seg_test.csv")
parser.add_argument("--out_dir", default=None, help="default = experiments/segmentation/hybrid_<decoder>")
parser.add_argument("--decoder", default="deeplabv3plus", choices=["deeplabv3plus", "ppm", "upernet", "segformer"],
                    help="segmentation head (the 'classifier'); deeplabv3plus won the ResNet-50 comparison so it is the default")
parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101"],
                    help="encoder depth (kept resnet50 for the decoder comparison; resnet101 = paper's PSPNet capacity, optional extra lever)")
parser.add_argument("--pretrained", type=int, default=1, help="1 = ImageNet-pretrained backbone")
parser.add_argument("--attn", type=int, default=1, help="1 = add the Transformer block for global context (on each head's OS16 context)")
parser.add_argument("--attn_heads", type=int, default=4)
parser.add_argument("--aux", type=int, default=1, help="1 = auxiliary deep-supervision head on the context features (helps convergence + regularises)")
parser.add_argument("--aux_weight", type=float, default=0.4, help="weight of the auxiliary loss")
parser.add_argument("--img_size", type=int, default=640)
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16], help="16 = default (OS8 was tried and gave NO gain, just slower/OOM-prone). 8 = denser features, needs --grad_checkpoint 1 --batch_size 4")
parser.add_argument("--grad_checkpoint", type=int, default=0, help="1 = gradient checkpointing on the backbone (saves memory; needed only for OS8 / 713px)")
parser.add_argument("--batch_size", type=int, default=8, help="8 fits OS16 on 8-12 GB; drop to 4 for --output_stride 8 or --img_size 713")
parser.add_argument("--epochs", type=int, default=80, help="more epochs: the stronger regularisation below slows fitting, so it needs longer to converge (early stopping guards)")
parser.add_argument("--patience", type=int, default=12)
parser.add_argument("--lr", type=float, default=2e-4, help="lr for the new modules (ASPP/attn/decoder)")
parser.add_argument("--backbone_lr_scale", type=float, default=0.3, help="backbone lr = lr * this; 0.1 was too low (backbone barely adapted to aerial images)")
parser.add_argument("--weight_decay", type=float, default=2e-3, help="raised from 1e-3 to curb the train<<val gap without adding capacity")
parser.add_argument("--dropout", type=float, default=0.2, help="decoder dropout (raised to curb the overfitting seen in the first run)")
parser.add_argument("--erase_p", type=float, default=0.5, help="random-erasing (cutout) probability on the training image: occludes a patch so the model must use context (reduces overfitting, helps flooded/non-flooded)")
parser.add_argument("--weight_power", type=float, default=0.5, help="class weight = 1/freq**power (0.5 = inverse-sqrt; lowering to 0.25 was tried and did NOT change the large-class IoU)")
parser.add_argument("--weight_cap", type=float, default=5.0)
parser.add_argument("--label_smoothing", type=float, default=0.0)
parser.add_argument("--loss", default="combo", choices=["ce", "combo"])
parser.add_argument("--ignore_bg", type=int, default=1)
parser.add_argument("--class_weights", type=int, default=1)
parser.add_argument("--augment", type=int, default=1)
parser.add_argument("--scheduler", default="cosine", choices=["none", "cosine", "poly"], help="cosine (default); poly (paper-faithful) was tried with no measurable gain")
parser.add_argument("--warmup_epochs", type=int, default=3)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--vis_every", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--tta", type=int, default=1, help="test-time augmentation (average over flips) on --test only; free ~1pt, no training change")
parser.add_argument("--occlusion_eval", action="store_true", help="robustness analysis: evaluate the best.pt at increasing input occlusion (a gray square covering 0..50%% of the area) and save the degradation curve")
parser.add_argument("--occlusion_fracs", default="0,0.1,0.2,0.3,0.4,0.5", help="occlusion area fractions to test")
parser.add_argument("--occlusion_vis", type=int, default=6, help="occlusion_eval: save this many sample figures (original + occluded input + prediction + ground truth across occlusion levels)")
args = parser.parse_args()
if args.out_dir is None:                                  # one folder per config, so runs don't collide (keeps the OS16 baselines)
    suffix = ("_os8" if args.output_stride == 8 else "")
    suffix += ("" if args.backbone == "resnet50" else "_" + args.backbone.replace("resnet", "r"))
    args.out_dir = os.path.join(r".\experiments\segmentation", "hybrid_" + args.decoder + suffix)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 10
CLASS_NAMES = ["background", "building-flooded", "building-non-flooded", "road-flooded",
               "road-non-flooded", "water", "tree", "vehicle", "pool", "grass"]
PALETTE = [(0, 0, 0), (255, 0, 0), (180, 120, 120), (160, 150, 20), (140, 140, 140),
           (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0), (60, 180, 75)]


def random_erase(img, size):
    arr = np.asarray(img).copy()
    area = size * size
    for _ in range(10):
        e = random.uniform(0.02, 0.15) * area
        ar = random.uniform(0.3, 3.3)
        eh, ew = int(round((e * ar) ** 0.5)), int(round((e / ar) ** 0.5))
        if 0 < eh < size and 0 < ew < size:
            y, x = random.randint(0, size - eh), random.randint(0, size - ew)
            arr[y:y + eh, x:x + ew] = np.random.randint(0, 256, (eh, ew, 3), dtype=np.uint8)
            return Image.fromarray(arr)
    return img


class SegDataset(data.Dataset):
    def __init__(self, csv_path, img_size, augment=False):
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig"), delimiter=";"))
        self.pairs = [(r["image"], r["mask"]) for r in rows]
        self.img_size = img_size
        self.augment = augment
        self.norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __len__(self):
        return len(self.pairs)

    def _aug(self, img, mask):
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
        if random.random() < args.erase_p:
            img = random_erase(img, self.img_size)
        return img, mask

    def __getitem__(self, index):
        img = Image.open(self.pairs[index][0]).convert("RGB")
        mask = Image.open(self.pairs[index][1])
        if self.augment:
            img, mask = self._aug(img, mask)
        else:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        img = self.norm(transforms.functional.to_tensor(img))
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


def class_weights(dataset, ignore_bg, power, cap, img_size=128):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, mask_path in dataset.pairs:
        m = Image.open(mask_path).resize((img_size, img_size), Image.NEAREST)
        counts += np.bincount(np.array(m).ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    freq = counts / max(counts.sum(), 1)
    w = np.where(freq > 0, 1.0 / np.power(freq + 1e-6, power), 0.0)
    if ignore_bg:
        w[0] = 0.0
    w = w / w[w > 0].mean()
    return torch.tensor(np.clip(w, 0.0, cap), dtype=torch.float32)


def make_vis_batch(dataset, n=4):
    rare = {1, 3, 7, 8}
    scored = []
    for idx, (_, mask_path) in enumerate(dataset.pairs):
        arr = np.array(Image.open(mask_path).resize((128, 128), Image.NEAREST))
        classes = set(np.unique(arr).tolist())
        scored.append((len(classes & rare), -round(float((arr == 0).mean()), 2), len(classes), idx))
    scored.sort(reverse=True)
    idxs = [t[-1] for t in scored[:n]]
    return torch.stack([dataset[i][0] for i in idxs]), torch.stack([dataset[i][1] for i in idxs])


class ASPP(nn.Module):
    def __init__(self, cin, cout=256, rates=(6, 12, 18)):
        super().__init__()
        def branch(k, r):
            return nn.Sequential(nn.Conv2d(cin, cout, k, padding=r if k > 1 else 0, dilation=r, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.branches = nn.ModuleList([branch(1, 1)] + [branch(3, r) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1, bias=False),
                                  nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(cout * (len(rates) + 2), cout, 1, bias=False),
                                     nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Dropout(0.5))

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        feats.append(F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.project(torch.cat(feats, 1))


class PPM(nn.Module):
    def __init__(self, cin, cout=256, bins=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([nn.Sequential(
            nn.AdaptiveAvgPool2d(b), nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True)) for b in bins])
        self.project = nn.Sequential(
            nn.Conv2d(cin + cout * len(bins), cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x):
        size = x.shape[-2:]
        feats = [x] + [F.interpolate(s(x), size=size, mode="bilinear", align_corners=False) for s in self.stages]
        return self.project(torch.cat(feats, 1))


class GlobalAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 2, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        t = self.layer(x.flatten(2).transpose(1, 2))
        return t.transpose(1, 2).reshape(b, c, h, w)


def conv_bn_relu(cin, cout, k=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DeepLabV3PlusHead(nn.Module):
    # ASPP context + low-level (1/4) skip decoder (DeepLabV3+): the current/reference head
    def __init__(self, cfg, attn):
        super().__init__()
        self.aspp = ASPP(2048, 256)
        self.attn = attn
        self.low = conv_bn_relu(256, 48)
        self.decoder = nn.Sequential(
            conv_bn_relu(256 + 48, 256, 3), nn.Dropout(cfg.dropout),
            conv_bn_relu(256, 256, 3), nn.Dropout(cfg.dropout),
            nn.Conv2d(256, NUM_CLASSES, 1))

    def forward(self, feats):
        c2, _, _, c5 = feats
        ctx = self.attn(self.aspp(c5))
        h = F.interpolate(ctx, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        return self.decoder(torch.cat([h, self.low(c2)], 1))


class PPMHead(nn.Module):
    def __init__(self, cfg, attn):
        super().__init__()
        self.ppm = PPM(2048, 256)
        self.attn = attn
        self.cls = nn.Sequential(nn.Dropout(cfg.dropout), nn.Conv2d(256, NUM_CLASSES, 1))

    def forward(self, feats):
        return self.cls(self.attn(self.ppm(feats[-1])))


class UPerNetHead(nn.Module):
    def __init__(self, cfg, attn, chans=(256, 512, 1024, 2048), cout=256):
        super().__init__()
        self.ppm = PPM(chans[-1], cout)
        self.attn = attn
        self.laterals = nn.ModuleList([conv_bn_relu(c, cout, 1) for c in chans[:-1]])   # c2,c3,c4
        self.fpn_convs = nn.ModuleList([conv_bn_relu(cout, cout, 3) for _ in chans[:-1]])
        self.fuse = nn.Sequential(conv_bn_relu(cout * len(chans), cout, 3), nn.Dropout(cfg.dropout),
                                  nn.Conv2d(cout, NUM_CLASSES, 1))

    def forward(self, feats):
        c2, c3, c4, c5 = feats
        p5 = self.attn(self.ppm(c5))
        outs = [self.laterals[i](f) for i, f in enumerate([c2, c3, c4])] + [p5]
        for i in range(len(outs) - 1, 0, -1):                # top-down path
            outs[i - 1] = outs[i - 1] + F.interpolate(outs[i], size=outs[i - 1].shape[-2:], mode="bilinear", align_corners=False)
        fpn = [self.fpn_convs[i](outs[i]) for i in range(3)] + [p5]
        tgt = fpn[0].shape[-2:]
        fpn = [F.interpolate(f, size=tgt, mode="bilinear", align_corners=False) for f in fpn]
        return self.fuse(torch.cat(fpn, 1))


class SegformerHead(nn.Module):
    def __init__(self, cfg, attn, chans=(256, 512, 1024, 2048), cout=256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, cout, 1) for c in chans])
        self.attn = attn
        self.fuse = nn.Sequential(conv_bn_relu(cout * len(chans), cout, 1), nn.Dropout(cfg.dropout),
                                  nn.Conv2d(cout, NUM_CLASSES, 1))

    def forward(self, feats):
        tgt = feats[0].shape[-2:]
        outs = []
        for i, f in enumerate(feats):
            p = self.proj[i](f)
            if i == len(feats) - 1:
                p = self.attn(p)                             # global attention on the deepest stage
            outs.append(F.interpolate(p, size=tgt, mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(outs, 1))


DECODERS = {"deeplabv3plus": DeepLabV3PlusHead, "ppm": PPMHead, "upernet": UPerNetHead, "segformer": SegformerHead}


class HybridSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from torchvision import models as tvm
        
        dil = [False, True, True] if cfg.output_stride == 8 else [False, False, True]
        if cfg.backbone == "resnet101":
            weights = tvm.ResNet101_Weights.IMAGENET1K_V2 if cfg.pretrained else None
            m = tvm.resnet101(weights=weights, replace_stride_with_dilation=dil)
        else:
            weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if cfg.pretrained else None
            m = tvm.resnet50(weights=weights, replace_stride_with_dilation=dil)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.grad_checkpoint = bool(cfg.grad_checkpoint)
        attn = GlobalAttn(256, cfg.attn_heads) if cfg.attn else nn.Identity()
        self.head = DECODERS[cfg.decoder](cfg, attn)
        self.aux_head = nn.Sequential(
            conv_bn_relu(1024, 256, 3), nn.Dropout(cfg.dropout), nn.Conv2d(256, NUM_CLASSES, 1)) if cfg.aux else None

    def backbone_parameters(self):
        for mod in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in mod.parameters():
                yield p

    def forward(self, x):
        ck = self.grad_checkpoint and self.training
        s = self.stem(x)
        if ck:
            from torch.utils.checkpoint import checkpoint
            c2 = checkpoint(self.layer1, s, use_reentrant=False)
            c3 = checkpoint(self.layer2, c2, use_reentrant=False)
            c4 = checkpoint(self.layer3, c3, use_reentrant=False)
            c5 = checkpoint(self.layer4, c4, use_reentrant=False)
        else:
            c2 = self.layer1(s); c3 = self.layer2(c2); c4 = self.layer3(c3); c5 = self.layer4(c4)
        main = F.interpolate(self.head([c2, c3, c4, c5]), size=x.shape[-2:], mode="bilinear", align_corners=False)
        if self.training and self.aux_head is not None:
            aux = F.interpolate(self.aux_head(c4), size=x.shape[-2:], mode="bilinear", align_corners=False)
            return main, aux
        return main


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
    bb_ids = {id(p) for p in model.backbone_parameters()}
    bb = [p for p in model.parameters() if id(p) in bb_ids and p.requires_grad]
    new = [p for p in model.parameters() if id(p) not in bb_ids and p.requires_grad]
    groups = [{"params": bb, "lr": cfg.lr * cfg.backbone_lr_scale}, {"params": new, "lr": cfg.lr}]
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    if cfg.scheduler == "none":
        return None
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        if cfg.scheduler == "poly":
            return (1 - prog) ** 0.9
        return 0.5 * (1 + math.cos(math.pi * prog))
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
            return fn(*fargs, **fkwargs)
        except OSError as e:
            print("write failed (%d/3): %s" % (attempt + 1, e)); time.sleep(1)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device):
    model.train()
    meter = utils.AverageMeter()
    for step, (img, mask) in enumerate(loader):
        img, mask = img.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model(img)
            if isinstance(out, tuple):
                loss = criterion(out[0], mask) + args.aux_weight * criterion(out[1], mask)
            else:
                loss = criterion(out, mask)
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


def evaluate(model, loader, criterion, device, tta=False):
    model.eval()
    meter = utils.AverageMeter()
    metrics = SegMetrics(NUM_CLASSES)
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            meter.update(criterion(logits, mask).item(), img.size(0))
            if tta:
                probs = torch.softmax(logits, 1)
                for dims in [(3,), (2,), (2, 3)]:
                    probs = probs + torch.flip(torch.softmax(model(torch.flip(img, dims)), 1), dims)
                pred = probs.argmax(1)
            else:
                pred = logits.argmax(1)
            metrics.update(pred.cpu().numpy(), mask.cpu().numpy())
    res = metrics.compute()
    res["val_loss"] = meter.avg
    return res


def run():
    utils.set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    logger = utils.get_logger("hybrid", os.path.join(args.out_dir, "train.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s | out dir: %s" % (device, args.out_dir))
    logger.info("config: %s" % vars(args))

    train_set = SegDataset(args.train_csv, args.img_size, augment=bool(args.augment))
    val_set = SegDataset(args.val_csv, args.img_size)
    pin, persist = torch.cuda.is_available(), args.num_workers > 0
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   drop_last=True, pin_memory=pin, persistent_workers=persist)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=pin, persistent_workers=persist)
    logger.info("train %d / val %d" % (len(train_set), len(val_set)))

    weights = class_weights(train_set, args.ignore_bg, args.weight_power, args.weight_cap).to(device) if args.class_weights else None
    if weights is not None:
        logger.info("class weights: %s" % ", ".join("%s %.2f" % (n, w) for n, w in zip(CLASS_NAMES, weights.tolist())))
    model = HybridSeg(args).to(device)
    logger.info("params: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    criterion = make_criterion(args, weights)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {k: [] for k in ["train_loss", "val_loss", "val_miou", "val_miou_no_bg", "val_dice", "val_precision", "val_recall"]}
    start_epoch, best_miou, no_improve, best = 0, -1.0, 0, {}
    last_ckpt = os.path.join(args.out_dir, "last.pt")
    if args.resume and os.path.exists(last_ckpt):
        ckpt = utils.load_checkpoint(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt["history"]; start_epoch = ckpt["epoch"]; best_miou = ckpt.get("best_miou", -1.0)
        logger.info("resumed from epoch %d" % start_epoch)

    vis_batch = make_vis_batch(val_set, n=4)
    fig_dir = os.path.join(args.out_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, device)
        res = evaluate(model, val_loader, criterion, device)
        for key, value in [("train_loss", train_loss), ("val_loss", res["val_loss"]),
                           ("val_miou", res["miou"]), ("val_miou_no_bg", res["miou_no_bg"]),
                           ("val_dice", res["mdice"]), ("val_precision", res["mprecision"]), ("val_recall", res["mrecall"])]:
            history[key].append(value)
        logger.info("epoch %3d/%d  train_loss %.4f  val_loss %.4f  mIoU %.4f  mIoU(no-bg) %.4f  F1 %.4f  wF1 %.4f  Prec %.4f  Rec %.4f  pixAcc %.4f" %
                    (epoch + 1, args.epochs, train_loss, res["val_loss"], res["miou"], res["miou_no_bg"],
                     res["mf1"], res["wf1"], res["mprecision"], res["mrecall"], res["pixel_acc"]))

        safe_write(plot_seg_curves, history, os.path.join(args.out_dir, "curves.png"))
        safe_write(utils.save_history_csv, history, os.path.join(args.out_dir, "history.csv"))
        safe_write(utils.save_checkpoint,
                   {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "history": history, "best_miou": best_miou}, last_ckpt)

        if res["miou"] > best_miou:
            best_miou, no_improve = res["miou"], 0
            best = {"epoch": epoch + 1, **res}
            safe_write(utils.save_checkpoint, {"epoch": epoch + 1, "model": model.state_dict(), "best_miou": best_miou}, os.path.join(args.out_dir, "best.pt"))
            safe_write(write_json, os.path.join(args.out_dir, "best_metrics.json"), {"epoch": epoch + 1, "class_names": CLASS_NAMES, **res})
            safe_write(plot_confusion_matrix, res["confusion_matrix"], CLASS_NAMES, os.path.join(args.out_dir, "confusion_matrix.png"))
        else:
            no_improve += 1

        if (epoch + 1) % args.vis_every == 0 or (epoch + 1) == args.epochs:
            save_pred_figure(model, vis_batch, device, os.path.join(fig_dir, "pred_epoch_%03d.png" % (epoch + 1)))

        if args.patience > 0 and no_improve >= args.patience:
            logger.info("early stopping at epoch %d" % (epoch + 1))
            break

    logger.info("done. best mIoU %.4f at epoch %d" % (best_miou, best.get("epoch", -1)))


def evaluate_on_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(args.out_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        print("no best.pt in %s" % args.out_dir); return
    model = HybridSeg(args).to(device)
    model.load_state_dict(utils.load_checkpoint(ckpt_path, map_location=device)["model"])
    test_set = SegDataset(args.test_csv, args.img_size)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    res = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, tta=bool(args.tta))
    print("\nTest results (%d images%s):" % (len(test_set), ", TTA" if args.tta else ""))
    print("  mIoU %.4f | mIoU(no-bg) %.4f | macroF1 %.4f | weightedF1 %.4f | Prec %.4f | Rec %.4f | pixAcc %.4f" %
          (res["miou"], res["miou_no_bg"], res["mf1"], res["wf1"], res["mprecision"], res["mrecall"], res["pixel_acc"]))
    print("  %-22s %8s %8s %8s %8s" % ("class", "IoU", "F1", "Prec", "Rec"))
    for i, name in enumerate(CLASS_NAMES):
        print("    %-20s %8.4f %8.4f %8.4f %8.4f" %
              (name, res["iou_per_class"][i], res["f1_per_class"][i], res["precision_per_class"][i], res["recall_per_class"][i]))
    write_json(os.path.join(args.out_dir, "test_metrics.json"), {"class_names": CLASS_NAMES, **res})
    plot_confusion_matrix(res["confusion_matrix"], CLASS_NAMES, os.path.join(args.out_dir, "confusion_matrix_test.png"))
    print("  saved test_metrics.json and confusion_matrix_test.png")


def occlude(img, frac, center=False):
    if frac <= 0:
        return img
    b, c, h, w = img.shape
    side = min(int(round((frac * h * w) ** 0.5)), h, w)
    out = img.clone()
    for i in range(b):
        if center:
            y, x = (h - side) // 2, (w - side) // 2
        else:
            y, x = random.randint(0, h - side), random.randint(0, w - side)
        out[i, :, y:y + side, x:x + side] = 0.0
    return out


def occlusion_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(args.out_dir, "best.pt")
    if not os.path.exists(ckpt):
        print("no best.pt in %s" % args.out_dir); return
    model = HybridSeg(args).to(device)
    model.load_state_dict(utils.load_checkpoint(ckpt, map_location=device)["model"]); model.eval()
    split_csv = args.test_csv if args.test else args.val_csv
    dset = SegDataset(split_csv, args.img_size)
    loader = data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    fracs = [float(x) for x in args.occlusion_fracs.split(",")]
    print("occlusion robustness on %s (%d images):" % (os.path.basename(split_csv), len(dset)))
    rows = []
    for frac in fracs:
        utils.set_seed(args.seed)
        metrics = SegMetrics(NUM_CLASSES)
        with torch.no_grad():
            for img, mask in loader:
                pred = model(occlude(img.to(device), frac)).argmax(1)
                metrics.update(pred.cpu().numpy(), mask.cpu().numpy())
        res = metrics.compute()
        rows.append((frac, res["miou_no_bg"], res["miou"], res["pixel_acc"]))
        print("  occ %3d%%  miou_no_bg %.4f  miou %.4f  pixAcc %.4f" % (round(frac * 100), res["miou_no_bg"], res["miou"], res["pixel_acc"]))
    base = rows[0][1]
    print("  degradation at 50%% occlusion: %.4f -> %.4f (%.1f%% relative drop)" % (base, rows[-1][1], 100 * (base - rows[-1][1]) / max(base, 1e-6)))
    with open(os.path.join(args.out_dir, "occlusion_robustness.csv"), "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp, delimiter=";"); w.writerow(["occ_frac", "miou_no_bg", "miou", "pixel_acc"])
        for r in rows:
            w.writerow(["%.2f" % r[0]] + ["%.4f" % v for v in r[1:]])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r[0] * 100 for r in rows], [r[1] for r in rows], "o-", label="mIoU (no bg)")
    ax.plot([r[0] * 100 for r in rows], [r[3] for r in rows], "s--", label="pixel acc")
    ax.set_xlabel("input occluded (%)"); ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title("Occlusion robustness - %s" % os.path.basename(args.out_dir)); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "occlusion_robustness.png"), dpi=110); plt.close(fig)
    print("  saved occlusion_robustness.csv and .png -> %s" % args.out_dir)
    if args.occlusion_vis > 0:
        save_occlusion_figures(model, dset, device, fracs)


def save_occlusion_figures(model, dset, device, fracs):
    import matplotlib.pyplot as plt
    occ = [f for f in fracs if f > 0]
    mean, std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)
    vdir = os.path.join(args.out_dir, "occlusion_samples"); os.makedirs(vdir, exist_ok=True)
    if not occ:
        return
    n = min(args.occlusion_vis, len(dset))
    idxs = random.Random().sample(range(len(dset)), n)
    for k, idx in enumerate(idxs):
        img, mask = dset[idx]
        cols = 1 + len(occ)
        fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6.2), squeeze=False)
        rgb = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0, 0].imshow(rgb); axes[0, 0].set_title("input (original)", fontsize=9); axes[0, 0].axis("off")
        axes[1, 0].imshow(colorize(mask.numpy())); axes[1, 0].set_title("ground truth", fontsize=9); axes[1, 0].axis("off")
        for j, f in enumerate(occ, start=1):
            occ_img = occlude(img.unsqueeze(0).to(device), f, center=True)
            with torch.no_grad():
                pred = model(occ_img).argmax(1)[0].cpu().numpy()
            occ_rgb = (occ_img[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            axes[0, j].imshow(occ_rgb); axes[0, j].set_title("occluded %d%%" % round(f * 100), fontsize=9); axes[0, j].axis("off")
            axes[1, j].imshow(colorize(pred)); axes[1, j].set_title("prediction %d%%" % round(f * 100), fontsize=9); axes[1, j].axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(vdir, "occ_sample_%02d.png" % k), dpi=100); plt.close(fig)
    print("  saved %d occlusion sample figures -> %s" % (n, vdir))


if __name__ == "__main__":
    if args.occlusion_eval:
        occlusion_eval()
    elif args.test:
        evaluate_on_test()
    else:
        run()
