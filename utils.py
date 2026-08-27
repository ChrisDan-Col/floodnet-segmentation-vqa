import os
import csv
import math
import random
import logging

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _ResilientFileHandler(logging.FileHandler):
    def emit(self, record):
        try:
            super().emit(record)
        except OSError:
            pass


def get_logger(name, log_file):
    folder = os.path.dirname(log_file)
    if folder:
        os.makedirs(folder, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # avoid duplicated lines if called twice
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    fh = _ResilientFileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0



def psnr(img1, img2, data_range=1.0):
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def ssim(img1, img2, window_size=11, sigma=1.5, data_range=1.0):
    channels = img1.size(1)
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    smap = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return smap.mean().item()


def save_checkpoint(state, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def save_history_csv(history, path):
    keys = list(history.keys())
    n = len(history[keys[0]])
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp, delimiter=";")
        w.writerow(["epoch"] + keys)
        for i in range(n):
            w.writerow([i + 1] + [history[k][i] for k in keys])


def plot_history(history, out_path, title="training"):
    loss_keys = [k for k in history if "loss" in k.lower()]
    metric_keys = [k for k in history if k not in loss_keys]

    panels = []
    if loss_keys:
        panels.append(("loss", loss_keys))
    for k in metric_keys:
        panels.append((k, [k]))

    epochs = range(1, len(history[list(history.keys())[0]]) + 1)
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]

    for ax, (name, keys) in zip(axes, panels):
        for k in keys:
            ax.plot(epochs, history[k], label=k)
        ax.set_xlabel("epoch"); ax.set_ylabel(name)
        ax.set_title("%s - %s" % (title, name)); ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
