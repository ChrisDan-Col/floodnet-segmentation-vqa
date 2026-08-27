# Segmentation metrics from a confusion matrix.
# For semantic segmentation the right metrics are per-class IoU, mean IoU, Dice (F1), 
# precision, recall and pixel accuracy.

import numpy as np


class SegMetrics:
    def __init__(self, num_classes):
        self.n = num_classes
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.n, self.n), dtype=np.int64)

    def update(self, pred, target):
        pred = np.asarray(pred).reshape(-1)
        target = np.asarray(target).reshape(-1)
        k = target * self.n + pred
        self.cm += np.bincount(k, minlength=self.n * self.n).reshape(self.n, self.n)

    def compute(self):
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        eps = 1e-9

        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        pixel_acc = tp.sum() / (cm.sum() + eps)

        f1 = 2 * precision * recall / (precision + recall + eps)
        present = cm.sum(1) > 0
        no_bg = present.copy(); no_bg[0] = False
        cls_freq = cm.sum(1) / (cm.sum() + eps)
        fwiou = float((cls_freq * iou).sum())
        wf1 = float((cls_freq * f1).sum())

        def mean(x, m):
            return float(x[m].mean()) if m.any() else 0.0

        return {
            "miou": mean(iou, present),
            "miou_no_bg": mean(iou, no_bg),
            "fwiou": fwiou,
            "mdice": mean(dice, present),
            "mf1": mean(f1, present),
            "mf1_no_bg": mean(f1, no_bg),
            "wf1": wf1,
            "mprecision": mean(precision, present),
            "mrecall": mean(recall, present),
            "pixel_acc": float(pixel_acc),
            "iou_per_class": iou.tolist(),
            "dice_per_class": dice.tolist(),
            "f1_per_class": f1.tolist(),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "confusion_matrix": self.cm.astype(int).tolist(),
        }
