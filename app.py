# FloodNet demo app, single screen (PySide6). Load an aerial image, segment it with HybridSeg,
# overlay the mask with a blend slider (image <-> mask), and answer a question about it
# (mask -> facts -> rule VQA). Runs on a plain venv (no Anaconda). Start: python app.py

import os
import re
import sys
import types

import numpy as np
from PIL import Image
import torch
from scipy import ndimage

sys.argv = [sys.argv[0]]
import train_hybrid as seg
import vqa

from PySide6 import QtCore, QtGui, QtWidgets

CKPT = os.path.join("experiments", "segmentation", "hybrid_deeplabv3plus", "best.pt")
ADAPTER = os.path.join("experiments", "vqa", "lora", "adapter")
IMG_SIZE = 640
WORK = 1024
MIN_AREA = max(10, int(3e-5 * WORK * WORK))
QTYPES = ["Simple_Counting", "Complex_Counting", "Yes_No", "Condition_Recognition"]
BUILDING_FLOODED, BUILDING_NON_FLOODED, ROAD_FLOODED, ROAD_NON_FLOODED, WATER = 1, 2, 3, 4, 5
FLOODED_CLASSES = (BUILDING_FLOODED, ROAD_FLOODED, WATER)
FLOOD_THRESHOLD = 0.30
UNSUPPORTED = ["tree", "grass", "water", "vehicle", "car", "pool", "people", "person"]


def count_blobs(mask, cid):
    binary = mask == cid
    if not binary.any():
        return 0
    blobs, n = ndimage.label(binary, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones_like(mask), blobs, index=range(1, n + 1))
    return int((sizes >= MIN_AREA).sum())


def road_state(mask):
    fl, dry = (mask == ROAD_FLOODED).sum() > 0, (mask == ROAD_NON_FLOODED).sum() > 0
    if fl and dry:
        return "flooded and non-flooded"
    return "flooded" if fl else ("non-flooded" if dry else "no road")


def mask_to_facts(mask):
    bf, bnf = count_blobs(mask, BUILDING_FLOODED), count_blobs(mask, BUILDING_NON_FLOODED)
    non_bg = int((mask > 0).sum())
    pct = float(np.isin(mask, FLOODED_CLASSES).sum()) / non_bg if non_bg else 0.0
    f = {"buildings_flooded": bf, "buildings_non_flooded": bnf, "buildings_total": bf + bnf,
         "pct_flooded": round(pct, 3), "road_state": road_state(mask),
         "condition": "flooded" if pct >= FLOOD_THRESHOLD else "non-flooded"}
    f["facts_text"] = ("flooded buildings: %d; non-flooded buildings: %d; total buildings: %d; "
                       "flooded area: %d%% of the scene; road: %s; overall condition: %s" %
                       (bf, bnf, bf + bnf, round(pct * 100), f["road_state"], f["condition"]))
    return f


def rule_answer(question, qtype, f):
    q = question.lower()
    if "Counting" in qtype:
        if "non" in q and "flood" in q:
            return str(f["buildings_non_flooded"])
        if "flood" in q:
            return str(f["buildings_flooded"])
        return str(f["buildings_total"])
    if qtype == "Yes_No":
        entity = f["road_state"] if "road" in q else f["condition"]
        dry = entity in ("non-flooded", "non flooded")
        asks_non = ("non" in q or "not" in q) and "flood" in q
        return ("yes" if dry else "no") if asks_non else ("yes" if entity == "flooded" else "no")
    return f["road_state"] if "road" in q else f["condition"]


def preprocess_question(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


class Segmenter:
    def __init__(self, ckpt):
        self.ckpt, self.model = ckpt, None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        cfg = types.SimpleNamespace(decoder="deeplabv3plus", backbone="resnet50", output_stride=16,
                                    attn=1, attn_heads=4, aux=1, dropout=0.2, pretrained=0, grad_checkpoint=0)
        self.model = seg.HybridSeg(cfg).to(self.device).eval()
        state = torch.load(self.ckpt, map_location=self.device)
        self.model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)

    def predict(self, pil_img):
        if self.model is None:
            self.load()
        w, h = pil_img.size
        x = np.asarray(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
        x = (x - np.array(seg.IMAGENET_MEAN)) / np.array(seg.IMAGENET_STD)
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            pred = self.model(t).argmax(1)[0].cpu().numpy().astype(np.uint8)
        return np.asarray(Image.fromarray(pred).resize((w, h), Image.NEAREST))     # mask at original size


class VQABackend:
    def __init__(self, adapter):
        self.adapter, self.llm, self.failed, self.error = adapter, None, False, ""

    def ready(self):
        return self.llm is not None or self.failed        # True once loaded or once we gave up

    def using_lora(self):
        return self.llm is not None

    def load(self):
        from peft import PeftModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base, tok = vqa.load_llm_base(device)
        model = PeftModel.from_pretrained(base, self.adapter).to(device).eval()
        self.llm = vqa.LLM(model=model, tok=tok)

    def answer(self, question, facts, qtype):
        # returns (answer, raw_llm_reply_or_None)
        if self.llm is None and not self.failed:
            try:
                self.load()
            except Exception as e:
                self.failed, self.error = True, str(e)
        if self.llm is not None:
            raw, pred = self.llm.answer(question, facts["facts_text"], qtype)
            return pred, raw
        return rule_answer(question, qtype, facts), None


DARK = """
QWidget { background:#1e1f22; color:#e6e6e6; font-size:13px; }
QPushButton { background:#2d6cdf; color:white; border:none; padding:8px; border-radius:5px; }
QPushButton:hover { background:#3a7bef; } QPushButton:disabled { background:#3a3d42; color:#888; }
QLineEdit, QComboBox { background:#2b2d31; border:1px solid #3a3d42; padding:6px; border-radius:4px; }
QGroupBox { border:1px solid #3a3d42; border-radius:6px; margin-top:10px; padding-top:8px; }
QGroupBox::title { subcontrol-origin:margin; left:10px; color:#9aa0a6; }
QLabel#answer { font-size:22px; font-weight:bold; color:#4ade80; }
QLabel#title { font-size:16px; font-weight:bold; }
"""


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FloodNet - Segmentation + VQA")
        self.resize(1120, 720)
        self.setStyleSheet(DARK)
        self.segmenter = Segmenter(CKPT)
        self.vqa = VQABackend(ADAPTER)
        self.image = None
        self.rgb = None
        self.mask_rgb = None
        self.mask = None
        self._build()

    def _build(self):
        # left: image + blend slider
        self.view = QtWidgets.QLabel("Load an image to start")
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setMinimumSize(560, 560)
        self.view.setStyleSheet("background:#111214; color:#777; border:1px solid #3a3d42;")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 100); self.slider.setValue(50); self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.update_view)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.view, 1)
        left.addWidget(QtWidgets.QLabel("Image  ◄──►  Mask"))
        left.addWidget(self.slider)

        # right: controls (single column)
        title = QtWidgets.QLabel("FloodNet · Segmentation + VQA"); title.setObjectName("title")

        g1 = QtWidgets.QGroupBox("1. Image")
        b_load = QtWidgets.QPushButton("Load image"); b_load.clicked.connect(self.load_image)
        self.b_proc = QtWidgets.QPushButton("Process (segment)"); self.b_proc.clicked.connect(self.process); self.b_proc.setEnabled(False)
        l1 = QtWidgets.QVBoxLayout(g1); l1.addWidget(b_load); l1.addWidget(self.b_proc)

        g2 = QtWidgets.QGroupBox("2. Question (VQA)")
        self.qtype = QtWidgets.QComboBox(); self.qtype.addItems(QTYPES)
        self.qedit = QtWidgets.QLineEdit(); self.qedit.setPlaceholderText("Type the question...")
        self.qedit.returnPressed.connect(self.ask)
        self.b_ask = QtWidgets.QPushButton("Answer"); self.b_ask.clicked.connect(self.ask); self.b_ask.setEnabled(False)
        self.answer = QtWidgets.QLabel("-"); self.answer.setObjectName("answer"); self.answer.setAlignment(QtCore.Qt.AlignCenter)
        self.facts = QtWidgets.QLabel(""); self.facts.setWordWrap(True); self.facts.setStyleSheet("color:#9aa0a6;")
        l2 = QtWidgets.QVBoxLayout(g2)
        l2.addWidget(QtWidgets.QLabel("Question type:")); l2.addWidget(self.qtype)
        l2.addWidget(self.qedit); l2.addWidget(self.b_ask)
        l2.addWidget(QtWidgets.QLabel("Answer:")); l2.addWidget(self.answer); l2.addWidget(self.facts)

        self.status = QtWidgets.QLabel("Ready. VQA = LoRA (loads on the first question)."); self.status.setStyleSheet("color:#9aa0a6;")
        right = QtWidgets.QVBoxLayout()
        right.addWidget(title); right.addWidget(g1); right.addWidget(g2); right.addStretch(1); right.addWidget(self.status)
        rw = QtWidgets.QWidget(); rw.setLayout(right); rw.setFixedWidth(360)

        root = QtWidgets.QHBoxLayout(self); root.addLayout(left, 1); root.addWidget(rw)

    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load image", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if not path:
            return
        self.image = Image.open(path).convert("RGB")
        self.rgb = np.ascontiguousarray(np.asarray(self.image))
        self.mask = self.mask_rgb = None
        self.slider.setEnabled(False); self.b_ask.setEnabled(False)
        self.answer.setText("-"); self.facts.setText("")
        self.b_proc.setEnabled(True)
        self._show(self.rgb)
        self.status.setText("Image loaded: %s (%dx%d)" % (os.path.basename(path), self.image.width, self.image.height))

    def process(self):
        if self.image is None:
            return
        if not os.path.exists(self.segmenter.ckpt):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select the checkpoint best.pt", "", "Checkpoint (*.pt)")
            if not path:
                self.status.setText("Model (best.pt) not found."); return
            self.segmenter.ckpt = path
        self.status.setText("Segmenting... (the first run loads the model, may take a while)")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            self.mask = self.segmenter.predict(self.image)
            self.mask_rgb = np.ascontiguousarray(seg.colorize(self.mask))
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Error", "Segmentation failed:\n%s" % e); self.status.setText("Segmentation error."); return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.slider.setEnabled(True); self.b_ask.setEnabled(True)
        self.update_view()
        self.status.setText("Segmentation ready. Use the slider or ask a question.")

    def update_view(self):
        if self.mask_rgb is None:
            return
        a = self.slider.value() / 100.0
        blended = ((1 - a) * self.rgb + a * self.mask_rgb).astype(np.uint8)
        self._show(np.ascontiguousarray(blended))

    def _show(self, arr):
        h, w, _ = arr.shape
        qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.view.setPixmap(pix)

    def ask(self):
        if self.mask is None:
            QtWidgets.QMessageBox.information(self, "Not processed", "Load and process (segment) an image first."); return
        raw_q = self.qedit.text().strip()
        if not raw_q:
            QtWidgets.QMessageBox.information(self, "Empty question", "Type a question."); return
        q = preprocess_question(raw_q)                 # normalised text, used only for the validators
        bad = [w for w in UNSUPPORTED if w in q]
        if bad:
            QtWidgets.QMessageBox.warning(self, "Not supported",
                "This VQA only answers about BUILDINGS (counting), the ROAD, or the overall CONDITION "
                "of the scene. The question mentions: %s." % ", ".join(bad)); return
        qtype = self.qtype.currentText()
        if "Counting" in qtype and "building" not in q:
            QtWidgets.QMessageBox.information(self, "Hint", "Counting only applies to buildings: include 'buildings' in the question."); return
        facts = mask_to_facts(np.asarray(Image.fromarray(self.mask.astype(np.uint8)).resize((WORK, WORK), Image.NEAREST)))
        if not self.vqa.ready():
            self.status.setText("Loading the LoRA LLM... (the first run downloads the base model, may take a few minutes)")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor); QtWidgets.QApplication.processEvents()
        ans, raw_llm = self.vqa.answer(raw_q, facts, qtype)   # raw question -> LoRA (matches its training)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.answer.setText(ans)
        ctx = "context (translator):  " + facts["facts_text"]
        if raw_llm is not None:
            ctx += "\nLLM (raw reply):  " + str(raw_llm)
        self.facts.setText(ctx)
        if self.vqa.using_lora():
            self.status.setText("Answered with LoRA (%s)." % qtype)
        else:
            self.status.setText("LLM unavailable (%s) -> answered with rule (%s)." % (self.vqa.error[:50], qtype))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
