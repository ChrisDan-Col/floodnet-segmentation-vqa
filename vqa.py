import os
import re
import csv
import json
import time
import shutil
import tempfile
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--approach", default="rule", choices=["rule", "prompt", "lora", "vilt", "vitbert", "clip", "all"], help="'all' = compare every saved approach on the test set (the final table)")
parser.add_argument("--train_csv", default=r".\prepared_data\vqa_train.csv")
parser.add_argument("--val_csv", default=r".\prepared_data\vqa_val.csv")
parser.add_argument("--test_csv", default=r".\prepared_data\vqa_test.csv")
parser.add_argument("--facts", default=r".\prepared_data\facts_gt.json", help="image_id -> facts (from translate_masks.py)")
parser.add_argument("--images_dir", default=r".\Datasets\FloodNet Challenge - Track 2\Images", help="needed by the vilt/vitbert approaches")
parser.add_argument("--use_facts", type=int, default=0, help="classifier approaches: 1 = give the mask facts to the classifier")
parser.add_argument("--facts_mode", default="feat", choices=["feat", "text"], help="how _facts feeds the facts: feat = numeric feature fusion (default, recommended); text = append facts_text to the question (old ablation, hurts)")
parser.add_argument("--label_smoothing", type=float, default=0.05, help="classifier training: CE label smoothing")
parser.add_argument("--out_dir", default=r".\experiments\vqa")
parser.add_argument("--llm_name", default="ank028/Llama-3.2-1B-Instruct-medmcqa")
parser.add_argument("--llm_cpu", action="store_true")
parser.add_argument("--lora_epochs", type=int, default=3, help="lora: fine-tuning epochs (LLM training is slow; 2-4 is plenty)")
parser.add_argument("--lora_r", type=int, default=8, help="lora: adapter rank")
parser.add_argument("--img_size", type=int, default=0, help="0 = model default (ViLT 384, ViT 224)")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--vis", type=int, default=12, help="save N example Q&A visualisations (image + mask + facts + answer) per run")
parser.add_argument("--masks_csv", default=r".\prepared_data\seg_train.csv,.\prepared_data\seg_val.csv,.\prepared_data\seg_test.csv", help="ground-truth mask CSVs, to show the mask behind the facts in --vis")
args = parser.parse_args()

QUESTION_TYPES = ["Simple_Counting", "Complex_Counting", "Yes_No", "Condition_Recognition"]
EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
PALETTE = [(0, 0, 0), (255, 0, 0), (180, 120, 120), (160, 150, 20), (140, 140, 140),
           (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0), (60, 180, 75)]
TRAINED = {"vilt", "vitbert"}
FACT_DIM = 7


def facts_vector(f):
    rs = f.get("road_state", "no road")
    return [f.get("buildings_flooded", 0) / 10.0,
            f.get("buildings_non_flooded", 0) / 10.0,
            f.get("buildings_total", 0) / 10.0,
            float(f.get("pct_flooded", 0.0)),
            1.0 if rs in ("flooded", "flooded and non-flooded") else 0.0,   # road touches flooding
            1.0 if rs in ("non-flooded", "flooded and non-flooded") else 0.0,  # road has a dry part
            1.0 if f.get("condition") == "flooded" else 0.0]


def norm(a):
    return re.sub(r"\s+", " ", str(a).lower().strip().rstrip(".").replace("-", " "))


def first_int(s):
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else None


def is_correct(pred, gt, qtype):
    p, g = norm(pred), norm(gt)
    if "Counting" in qtype:
        return first_int(p) is not None and first_int(p) == first_int(g)
    if p == g:
        return True
    if re.search(r"\b" + re.escape(g) + r"\b", p):
        if g == "flooded" and re.search(r"\bnon flooded\b", p):   # avoid 'flooded' matching 'non flooded'
            return False
        return True
    return False


def read_rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig"), delimiter=";"))


def find_images(root):
    lookup = {}
    for dp, _, files in os.walk(root):
        for f in files:
            if f.endswith(EXTS):
                lookup[os.path.splitext(f)[0]] = os.path.join(dp, f)
    return lookup


def run_folder():
    suffix = ("_facts" if args.facts_mode == "feat" else "_factstext") if args.use_facts else ""
    return os.path.join(args.out_dir, args.approach + suffix)


def save_json(path, obj):
    json.dump(obj, open(path, "w"), indent=2)


def atomic_write(write_fn, path):
    tmp = os.path.join(tempfile.gettempdir(), "vqatmp_%d_%s" % (os.getpid(), os.path.basename(path)))
    try:
        write_fn(tmp)
    except Exception as e:
        print("could not build %s: %s" % (os.path.basename(path), e)); return
    for i in range(3):
        try:
            shutil.copyfile(tmp, path); os.remove(tmp); return
        except Exception as e:
            print("save to %s failed (%d/3): %s" % (path, i + 1, e)); time.sleep(2)
    print("!! could not save %s -- run with --out_dir on a LOCAL disk (not OneDrive/Documents)" % path)


def score(records):
    per_type = {t: [0, 0] for t in QUESTION_TYPES}
    for r in records:
        per_type[r["type"]][0] += int(r["correct"]); per_type[r["type"]][1] += 1
    ok = sum(v[0] for v in per_type.values()); n = sum(v[1] for v in per_type.values())
    res = {"accuracy": ok / max(n, 1), "n": n}
    for t in QUESTION_TYPES:
        c, m = per_type[t]
        res["acc_" + t] = c / m if m else 0.0
        res["n_" + t] = m
    return res


def score_and_save(records, out, split):
    res = score(records)
    print("\n%s (%s%s): overall %.4f  [SC %.3f CC %.3f YN %.3f Cond %.3f]  (n=%d)" %
          (split, args.approach, "+facts" if args.use_facts else "", res["accuracy"],
           res["acc_Simple_Counting"], res["acc_Complex_Counting"], res["acc_Yes_No"], res["acc_Condition_Recognition"], res["n"]))
    os.makedirs(out, exist_ok=True)
    atomic_write(lambda p: save_json(p, res), os.path.join(out, "test_metrics.json" if split == "test" else "best_metrics.json"))
    atomic_write(lambda p: save_json(p, records), os.path.join(out, "test_results.json" if split == "test" else "results.json"))
    return res


def load_facts():
    return json.load(open(args.facts)) if os.path.exists(args.facts) else {}


def aligned_test_rows(rows, facts, images):
    have_f, have_i = bool(facts), bool(images)
    return [r for r in rows if (not have_f or r["image_id"] in facts) and (not have_i or r["image_id"] in images)]


def _iid(path):
    n = os.path.splitext(os.path.basename(path))[0]
    return n[:-4] if n.endswith("_lab") else n


def _colorize(mask):
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for i, c in enumerate(PALETTE):
        rgb[mask == i] = c
    return rgb


def _wrap(s, width=52):
    import textwrap
    return "\n".join(textwrap.fill(ln, width) for ln in str(s).split("\n"))


def visualize(records, out, split, uses_facts):
    if args.vis <= 0 or not records:
        return
    imgs = find_images(args.images_dir)
    masks = {}
    if uses_facts and args.masks_csv:
        for p in args.masks_csv.split(","):
            if os.path.exists(p.strip()):
                for r in csv.DictReader(open(p.strip(), encoding="utf-8-sig"), delimiter=";"):
                    masks[_iid(r["image"])] = r["mask"]
    facts = load_facts()

    def is_flooded(iid):
        f = facts.get(iid)
        return bool(f) and (f.get("condition") == "flooded" or f.get("buildings_flooded", 0) > 0 or "flood" in str(f.get("road_state", "")))

    def by_type(want_correct):
        return {t: sorted([r for r in records if r["type"] == t and bool(r["correct"]) == want_correct],
                          key=lambda r: 0 if is_flooded(r["image_id"]) else 1) for t in QUESTION_TYPES}

    correct, wrong = by_type(True), by_type(False)
    picked, used, seen = [], set(), set()

    def fill(pool, upto, distinct):
        cur = {t: 0 for t in QUESTION_TYPES}
        progress = True
        while len(picked) < upto and progress:
            progress = False
            for t in QUESTION_TYPES:
                if len(picked) >= upto:
                    break
                while cur[t] < len(pool[t]):
                    r = pool[t][cur[t]]; cur[t] += 1
                    if id(r) in seen or (distinct and r["image_id"] in used):
                        continue
                    picked.append(r); seen.add(id(r)); used.add(r["image_id"]); progress = True
                    break

    n_wrong = max(1, args.vis // 4)
    fill(correct, args.vis - n_wrong, True)
    fill(wrong, args.vis, True)
    fill(correct, args.vis, False)
    fill(wrong, args.vis, False)
    picked = picked[:args.vis]
    vdir = os.path.join(out, "vis_" + split); os.makedirs(vdir, exist_ok=True)
    for k, r in enumerate(picked):
        iid = r["image_id"]
        has_mask = iid in masks and os.path.exists(masks[iid])
        cols = 2 + (1 if has_mask else 0)
        fig, axes = plt.subplots(1, cols, figsize=(5.2 * cols, 5))
        if iid in imgs and os.path.exists(imgs[iid]):
            im = Image.open(imgs[iid]).convert("RGB"); im.thumbnail((520, 520)); axes[0].imshow(im)
        axes[0].set_title("image %s" % iid, fontsize=9); axes[0].axis("off")
        ci = 1
        if has_mask:
            m = np.array(Image.open(masks[iid])); m = m[:, :, 0] if m.ndim == 3 else m
            axes[ci].imshow(_colorize(m)); axes[ci].set_title("GT mask", fontsize=9); axes[ci].axis("off"); ci += 1
        lines = ["TYPE: %s" % r["type"], "", "Q: %s" % _wrap(r["question"]), ""]
        if uses_facts and facts.get(iid):
            lines += ["CONTEXT (translator facts):", _wrap(facts[iid]["facts_text"]), ""]
        if "raw" in r:
            lines += ["LLM raw answer:", _wrap(r["raw"]), ""]
        lines += ["Predicted (scored): %s" % r["pred"], "Ground truth:       %s" % r["gt"],
                  "", "=> %s" % ("CORRECT" if r["correct"] else "WRONG")]
        axes[ci].axis("off")
        axes[ci].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
        atomic_write(lambda p: fig.savefig(p, dpi=90, bbox_inches="tight"), os.path.join(vdir, "ex_%02d_%s_%s.png" % (k, r["type"][:4], "ok" if r["correct"] else "x")))
        plt.close(fig)
    print("saved %d visualisations -> %s" % (len(picked), vdir))


def extract_answer(text, qtype):
    t = norm(text)
    if "Counting" in qtype:
        m = re.search(r"\d+", t)
        return m.group() if m else t
    if qtype == "Yes_No":
        m = re.search(r"\b(yes|no)\b", t)
        return m.group() if m else t
    if "flood" in t and "non" in t and ("and" in t or "," in t):
        return "flooded,non flooded"
    if "non flood" in t or "not flood" in t:
        return "non flooded"
    if "flood" in t:
        return "flooded"
    return t


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
        entirely_flooded, entirely_dry = entity == "flooded", entity in ("non-flooded", "non flooded")
        asks_non = ("non" in q or "not" in q) and "flood" in q
        return ("yes" if entirely_dry else "no") if asks_non else ("yes" if entirely_flooded else "no")
    return f["road_state"] if "road" in q else f["condition"]


def llm_prompt(tok, question, facts_text, qtype):
    fmt = ("Reply with only a whole number." if "Counting" in qtype else
           "Reply with only Yes or No." if qtype == "Yes_No" else
           "Reply with only: flooded, non flooded, or flooded and non flooded.")
    msg = ("Answer the question about a flooded aerial scene using ONLY these facts.\n"
           "Facts: %s\nQuestion: %s\n%s" % (facts_text, question, fmt))
    try:
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
    except Exception:
        return msg + "\nAnswer: "


class LLM:
    def __init__(self, model=None, tok=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = torch.device("cpu" if args.llm_cpu or not torch.cuda.is_available() else "cuda")
        if model is not None:
            self.tok, self.model = tok, model
            return
        print("loading LLM: %s" % args.llm_name)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(args.llm_name)
        self.model = AutoModelForCausalLM.from_pretrained(args.llm_name, dtype=dtype).to(self.device)

    def answer(self, question, facts_text, qtype):
        text = llm_prompt(self.tok, question, facts_text, qtype)
        ins = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ins, max_new_tokens=12, do_sample=False, pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(out[0][ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return raw, extract_answer(raw, qtype)


def symbolic_records(rows, facts, llm, logger=None):
    records = []
    for i, r in enumerate(rows):
        f = facts.get(r["image_id"])
        if f is None:
            continue
        raw = None
        if llm is None:
            pred = rule_answer(r["question"], r["type"], f)
        else:
            raw, pred = llm.answer(r["question"], f["facts_text"], r["type"])
        rec = {"image_id": r["image_id"], "type": r["type"], "question": r["question"],
               "gt": r["answer"], "pred": pred, "correct": bool(is_correct(pred, r["answer"], r["type"]))}
        if raw is not None:
            rec["raw"] = raw
        records.append(rec)
        if logger and llm and (i + 1) % 100 == 0:
            logger.info("answered %d/%d" % (i + 1, len(rows)))
    return records


def run_symbolic(rows, facts, split):
    out = run_folder(); os.makedirs(out, exist_ok=True)
    logger = utils.get_logger("vqa", os.path.join(out, "train.log"))
    logger.info("config: %s" % vars(args))
    if split == "test":
        rows = aligned_test_rows(rows, facts, find_images(args.images_dir))
    llm = LLM() if args.approach == "prompt" else None
    records = symbolic_records(rows, facts, llm, logger)
    logger.info("answered %d, skipped %d (no facts)" % (len(records), len(rows) - len(records)))
    res = score_and_save(records, out, split)
    logger.info("%s: overall %.4f | %s" % (split, res["accuracy"], {t: round(res["acc_" + t], 3) for t in QUESTION_TYPES}))
    visualize(records, out, split, uses_facts=True)


def load_llm_base(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32) if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.llm_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.llm_name, dtype=dtype).to(device)
    return model, tok


def lora_batches(rows, facts, tok):
    enc = []
    for r in rows:
        prompt = llm_prompt(tok, r["question"], facts[r["image_id"]]["facts_text"], r["type"])
        full = prompt + " " + r["answer"] + tok.eos_token
        ids = tok(full, add_special_tokens=False, truncation=True, max_length=256)["input_ids"]
        plen = len(tok(prompt, add_special_tokens=False)["input_ids"])
        labels = [-100] * min(plen, len(ids)) + ids[min(plen, len(ids)):]
        enc.append((ids, labels))
    return enc


def run_lora(split):
    out = run_folder(); os.makedirs(out, exist_ok=True)
    logger = utils.get_logger("vqa", os.path.join(out, "train.log"))
    logger.info("config: %s" % vars(args))
    device = torch.device("cpu" if args.llm_cpu or not torch.cuda.is_available() else "cuda")
    facts = load_facts()
    adapter_dir = os.path.join(out, "adapter")

    if split == "test":
        from peft import PeftModel
        base, tok = load_llm_base(device)
        model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()
        rows = aligned_test_rows(read_rows(args.test_csv), facts, find_images(args.images_dir))
        records = symbolic_records(rows, facts, LLM(model=model, tok=tok), logger)
        res = score_and_save(records, out, "test")
        logger.info("test: overall %.4f | %s" % (res["accuracy"], {t: round(res["acc_" + t], 3) for t in QUESTION_TYPES}))
        visualize(records, out, "test", uses_facts=True)
        return

    from peft import LoraConfig, get_peft_model
    base, tok = load_llm_base(device)
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg)
    train_rows = [r for r in read_rows(args.train_csv) if r["image_id"] in facts]
    if not train_rows:
        logger.info("no train rows have facts -- regenerate facts_gt.json for ALL splits first"); return
    logger.info("LoRA on %d train examples with facts (rank %d, %d epochs)" % (len(train_rows), args.lora_r, args.lora_epochs))

    def collate_lora(batch):
        m = max(len(ids) for ids, _ in batch)
        ii = [ids + [tok.pad_token_id] * (m - len(ids)) for ids, _ in batch]
        ll = [lab + [-100] * (m - len(lab)) for _, lab in batch]
        at = [[1] * len(ids) + [0] * (m - len(ids)) for ids, _ in batch]
        return torch.tensor(ii), torch.tensor(at), torch.tensor(ll)

    loader = data.DataLoader(lora_batches(train_rows, facts, tok), batch_size=args.batch_size, shuffle=True, collate_fn=collate_lora)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for epoch in range(args.lora_epochs):
        model.train(); meter = utils.AverageMeter()
        for step, (ids, at, lab) in enumerate(loader):
            ids, at, lab = ids.to(device), at.to(device), lab.to(device)
            loss = model(input_ids=ids, attention_mask=at, labels=lab).loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            meter.update(loss.item(), ids.size(0))
            if step % 20 == 0:
                print("   epoch %d  batch %4d/%d  loss %.4f" % (epoch + 1, step, len(loader), loss.item()))
        logger.info("epoch %d/%d  train_loss %.4f" % (epoch + 1, args.lora_epochs, meter.avg))
    model.save_pretrained(adapter_dir)
    logger.info("saved LoRA adapter -> %s" % adapter_dir)

    model.eval()
    val_rows = [r for r in read_rows(args.val_csv) if r["image_id"] in facts]
    records = symbolic_records(val_rows, facts, LLM(model=model, tok=tok), logger)
    res = score_and_save(records, out, "val")
    logger.info("val: overall %.4f | %s" % (res["accuracy"], {t: round(res["acc_" + t], 3) for t in QUESTION_TYPES}))
    visualize(records, out, "val", uses_facts=True)


class CLIPScorer:
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-base-patch32"
        print("loading CLIP: %s" % name)
        self.device = torch.device("cpu" if args.llm_cpu or not torch.cuda.is_available() else "cuda")
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    def answer(self, image, question, candidates):
        texts = ["%s %s" % (question, c) for c in candidates]
        inp = self.proc(text=texts, images=image, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            j = self.model(**inp).logits_per_image[0].argmax().item()
        return candidates[j]


def clip_candidates(train_rows):
    cand = {t: [] for t in QUESTION_TYPES}
    for r in train_rows:
        a = norm(r["answer"])
        if a not in cand[r["type"]]:
            cand[r["type"]].append(a)
    return cand


def clip_records(rows, images, scorer, cand, logger=None):
    records = []
    for i, r in enumerate(rows):
        if r["image_id"] not in images or not cand[r["type"]]:
            continue
        pred = scorer.answer(Image.open(images[r["image_id"]]).convert("RGB"), r["question"], cand[r["type"]])
        records.append({"image_id": r["image_id"], "type": r["type"], "question": r["question"],
                        "gt": r["answer"], "pred": pred, "correct": bool(is_correct(pred, r["answer"], r["type"]))})
        if logger and (i + 1) % 100 == 0:
            logger.info("scored %d/%d" % (i + 1, len(rows)))
    return records


def run_clip(split):
    out = run_folder(); os.makedirs(out, exist_ok=True)
    logger = utils.get_logger("vqa", os.path.join(out, "train.log"))
    logger.info("config: %s" % vars(args))
    images = find_images(args.images_dir)
    cand = clip_candidates(read_rows(args.train_csv))
    rows = read_rows(args.test_csv if split == "test" else args.val_csv)
    if split == "test":
        rows = aligned_test_rows(rows, load_facts(), images)
    records = clip_records(rows, images, CLIPScorer(), cand, logger)
    logger.info("scored %d, skipped %d (no image)" % (len(records), len(rows) - len(records)))
    res = score_and_save(records, out, split)
    logger.info("%s: overall %.4f | %s" % (split, res["accuracy"], {t: round(res["acc_" + t], 3) for t in QUESTION_TYPES}))
    visualize(records, out, split, uses_facts=False)


class VQADataset(data.Dataset):
    def __init__(self, rows, answer2idx, images, facts, tf, facts_mode="feat"):
        self.rows = [r for r in rows if r["image_id"] in images]
        self.answer2idx = answer2idx; self.images = images; self.facts = facts; self.tf = tf
        self.facts_mode = facts_mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = self.tf(Image.open(self.images[r["image_id"]]).convert("RGB"))
        text = r["question"]
        fvec = [0.0] * FACT_DIM
        f = self.facts.get(r["image_id"]) if self.facts is not None else None
        if f is not None:
            if self.facts_mode == "text":
                text = text + " context: " + f["facts_text"]
            else:
                fvec = facts_vector(f)
        return img, text, torch.tensor(fvec, dtype=torch.float32), self.answer2idx.get(norm(r["answer"]), -1), r["type"]


def collate(batch):
    imgs, texts, fvecs, labels, types = zip(*batch)
    return torch.stack(imgs), list(texts), torch.stack(fvecs), torch.tensor(labels, dtype=torch.long), list(types)


class FactHead(nn.Module):
    def __init__(self, in_dim, n, fuse):
        super().__init__()
        self.fuse = fuse
        self.fmlp = nn.Sequential(nn.Linear(FACT_DIM, 32), nn.ReLU()) if fuse else None
        self.head = nn.Sequential(nn.Linear(in_dim + (32 if fuse else 0), 256), nn.ReLU(),
                                  nn.Dropout(args.dropout), nn.Linear(256, n))

    def forward(self, feat, facts):
        if self.fuse:
            feat = torch.cat([feat, self.fmlp(facts)], dim=1)
        return self.head(feat)


class ViLTClassifier(nn.Module):
    def __init__(self, n, fuse=False):
        super().__init__()
        from transformers import ViltModel, ViltProcessor
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        for p in self.vilt.parameters():
            p.requires_grad = False
        self.head = FactHead(768, n, fuse)

    def forward(self, images, texts, facts, device):
        tok = self.processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=40).to(device)
        feat = self.vilt(input_ids=tok.input_ids, attention_mask=tok.attention_mask, pixel_values=images).pooler_output
        return self.head(feat, facts.to(device))


class ViTBERTClassifier(nn.Module):
    def __init__(self, n, fuse=False):
        super().__init__()
        from transformers import ViTModel, DistilBertModel, DistilBertTokenizer
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for p in list(self.vit.parameters()) + list(self.bert.parameters()):
            p.requires_grad = False
        self.head = FactHead(768 + 768, n, fuse)

    def forward(self, images, texts, facts, device):
        img_f = self.vit(pixel_values=images).pooler_output
        tok = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        txt_f = self.bert(**tok).last_hidden_state[:, 0, :]
        return self.head(torch.cat([img_f, txt_f], dim=1), facts.to(device))


def build_transform(approach=None):
    approach = approach or args.approach
    if approach == "vilt":
        s = args.img_size or 384
        return transforms.Compose([transforms.Resize((s, s)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    s = args.img_size or 224
    return transforms.Compose([transforms.Resize((s, s)), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def build_type2idx(train_rows, answer2idx):
    t2 = {t: set() for t in QUESTION_TYPES}
    for r in train_rows:
        a = answer2idx.get(norm(r["answer"]))
        if a is not None:
            t2[r["type"]].add(a)
    return t2


def predict_classifier(model, loader, idx2answer, type2idx, device):
    model.eval()
    records = []
    with torch.no_grad():
        for imgs, texts, fvecs, labels, types in loader:
            logits = model(imgs.to(device), texts, fvecs, device)
            for i, t in enumerate(types):
                valid = type2idx.get(t)
                if valid:
                    m = torch.full((logits.size(1),), float("-inf"), device=logits.device)
                    m[list(valid)] = 0.0
                    logits[i] = logits[i] + m
            for p, t in zip(logits.argmax(1).cpu().numpy(), types):
                records.append({"type": t, "pred": idx2answer[int(p)]})
    return records


def classifier_records(rows, model, answer2idx, idx2answer, type2idx, images, facts, tf, device, facts_mode="feat"):
    used = [r for r in rows if r["image_id"] in images]
    loader = data.DataLoader(VQADataset(rows, answer2idx, images, facts, tf, facts_mode), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    preds = predict_classifier(model, loader, idx2answer, type2idx, device)
    return [{"image_id": r["image_id"], "type": r["type"], "question": r["question"], "gt": r["answer"],
             "pred": p["pred"], "correct": bool(is_correct(p["pred"], r["answer"], r["type"]))} for r, p in zip(used, preds)]


def run_classifier(split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = run_folder(); os.makedirs(out, exist_ok=True)
    logger = utils.get_logger("vqa", os.path.join(out, "train.log"))
    utils.set_seed(args.seed)
    images = find_images(args.images_dir)
    facts = json.load(open(args.facts)) if args.use_facts and os.path.exists(args.facts) else None
    tf = build_transform()

    train_rows = read_rows(args.train_csv)
    answer2idx = {a: i for i, a in enumerate(sorted({norm(r["answer"]) for r in train_rows}))}
    idx2answer = {i: a for a, i in answer2idx.items()}
    type2idx = build_type2idx(train_rows, answer2idx)
    atomic_write(lambda p: save_json(p, answer2idx), os.path.join(out, "answers.json"))
    fuse = bool(args.use_facts) and args.facts_mode == "feat"
    model = (ViLTClassifier if args.approach == "vilt" else ViTBERTClassifier)(len(answer2idx), fuse).to(device)
    logger.info("config: %s | answers %d | images %d | facts_fusion %s" % (vars(args), len(answer2idx), len(images), fuse))

    if split == "test":
        model.load_state_dict(utils.load_checkpoint(os.path.join(out, "best.pt"), map_location=device)["model"])
        rows = aligned_test_rows(read_rows(args.test_csv), load_facts(), images)
        recs = classifier_records(rows, model, answer2idx, idx2answer, type2idx, images, facts, tf, device, args.facts_mode)
        score_and_save(recs, out, "test")
        visualize(recs, out, "test", uses_facts=bool(args.use_facts))
        return

    train_ds = VQADataset(train_rows, answer2idx, images, facts, tf, args.facts_mode)
    val_rows = read_rows(args.val_csv)
    val_ds = VQADataset(val_rows, answer2idx, images, facts, tf, args.facts_mode)
    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    history = {k: [] for k in ["train_loss", "accuracy"] + ["acc_" + t for t in QUESTION_TYPES]}
    best_acc, no_improve = -1.0, 0
    used_val = [r for r in val_rows if r["image_id"] in images]
    for epoch in range(args.epochs):
        model.train(); meter = utils.AverageMeter()
        for step, (imgs, texts, fvecs, labels, _) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs, texts, fvecs, device), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            meter.update(loss.item(), imgs.size(0))
            if step % 20 == 0:
                print("   batch %4d/%d  loss %.4f" % (step, len(train_loader), loss.item()))
        preds = predict_classifier(model, val_loader, idx2answer, type2idx, device)
        recs = [{"type": r["type"], "correct": bool(is_correct(p["pred"], r["answer"], r["type"]))} for r, p in zip(used_val, preds)]
        pt = {t: [0, 0] for t in QUESTION_TYPES}
        for r in recs:
            pt[r["type"]][0] += int(r["correct"]); pt[r["type"]][1] += 1
        acc = sum(v[0] for v in pt.values()) / max(sum(v[1] for v in pt.values()), 1)
        history["train_loss"].append(meter.avg); history["accuracy"].append(acc)
        for t in QUESTION_TYPES:
            history["acc_" + t].append(pt[t][0] / pt[t][1] if pt[t][1] else 0.0)
        logger.info("epoch %3d/%d  train_loss %.4f  val_acc %.4f" % (epoch + 1, args.epochs, meter.avg, acc))
        atomic_write(lambda p: utils.save_history_csv(history, p), os.path.join(out, "history.csv"))
        if acc > best_acc:
            best_acc, no_improve = acc, 0
            atomic_write(lambda p: utils.save_checkpoint({"epoch": epoch + 1, "model": model.state_dict()}, p), os.path.join(out, "best.pt"))
            full = [{"image_id": r["image_id"], "type": r["type"], "question": r["question"], "gt": r["answer"],
                     "pred": p["pred"], "correct": bool(is_correct(p["pred"], r["answer"], r["type"]))} for r, p in zip(used_val, preds)]
            score_and_save(full, out, "val")
            visualize(full, out, "val", uses_facts=bool(args.use_facts))
        else:
            no_improve += 1
        if args.patience > 0 and no_improve >= args.patience:
            logger.info("early stopping at epoch %d" % (epoch + 1)); break
    logger.info("done. best val accuracy %.4f" % best_acc)


def parse_folder(name):
    if name.endswith("_factstext"):
        return name[:-10], 1, "text"
    if name.endswith("_facts"):
        return name[:-6], 1, "feat"
    return name, 0, "feat"


def eval_folder_on_rows(folder, d, rows, facts, images, device):
    base, use_facts, facts_mode = parse_folder(folder)
    if base == "rule":
        return symbolic_records(rows, facts, None)
    if base == "prompt":
        return symbolic_records(rows, facts, LLM())
    if base == "clip":
        return clip_records(rows, images, CLIPScorer(), clip_candidates(read_rows(args.train_csv)))
    if base == "lora":
        adir = os.path.join(d, "adapter")
        if not os.path.isdir(adir):
            return None
        from peft import PeftModel
        model_base, tok = load_llm_base(device)
        model = PeftModel.from_pretrained(model_base, adir).to(device).eval()
        return symbolic_records(rows, facts, LLM(model=model, tok=tok))
    if base in TRAINED:
        ckpt, ans = os.path.join(d, "best.pt"), os.path.join(d, "answers.json")
        if not (os.path.exists(ckpt) and os.path.exists(ans)):
            return None
        answer2idx = json.load(open(ans))
        idx2answer = {i: a for a, i in answer2idx.items()}
        type2idx = build_type2idx(read_rows(args.train_csv), answer2idx)
        fuse = bool(use_facts) and facts_mode == "feat"
        model = (ViLTClassifier if base == "vilt" else ViTBERTClassifier)(len(answer2idx), fuse).to(device)
        model.load_state_dict(utils.load_checkpoint(ckpt, map_location=device)["model"])
        return classifier_records(rows, model, answer2idx, idx2answer, type2idx, images,
                                  facts if use_facts else None, build_transform(base), device, facts_mode)
    return None


def compare_on_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    facts, images = load_facts(), find_images(args.images_dir)
    test_rows = read_rows(args.test_csv)
    rows = aligned_test_rows(test_rows, facts, images)
    print("comparison on TEST: %d of %d questions have BOTH a fact and an image (the shared set)" % (len(rows), len(test_rows)))
    if not rows:
        have_f = sum(1 for r in test_rows if r["image_id"] in facts)
        have_i = sum(1 for r in test_rows if r["image_id"] in images)
        print("no shared rows. of %d test questions: %d have a fact, %d have an image." % (len(test_rows), have_f, have_i))
        print("  --facts %s has %d entries; --images_dir %s found %d images." % (args.facts, len(facts), args.images_dir, len(images)))
        if have_f == 0:
            print("  the facts file does NOT cover the test split, regenerate it for ALL splits:")
            print("  python translate_masks.py --masks_csv seg_train,seg_val,seg_test --vqa_csv vqa_train,vqa_val,vqa_test --out .\\prepared_data\\facts_gt.json")
        return
    folders = sorted(f for f in os.listdir(args.out_dir)
                     if os.path.isdir(os.path.join(args.out_dir, f)) and not f.startswith("_"))
    table = []
    for folder in folders:
        try:
            recs = eval_folder_on_rows(folder, os.path.join(args.out_dir, folder), rows, facts, images, device)
        except Exception as e:
            print("  skip %s: %s" % (folder, e)); continue
        if not recs:
            print("  skip %s (no saved best model)" % folder); continue
        res = score(recs); res["approach"] = folder; table.append(res)
        print("  %-16s overall %.4f  [SC %.3f CC %.3f YN %.3f Cond %.3f]" %
              (folder, res["accuracy"], res["acc_Simple_Counting"], res["acc_Complex_Counting"],
               res["acc_Yes_No"], res["acc_Condition_Recognition"]))
        _, uf, _ = parse_folder(folder)
        visualize(recs, os.path.join(args.out_dir, folder), "test", uses_facts=bool(uf))   # per-model test examples
    if not table:
        print("nothing to compare -- run some approaches first"); return
    table.sort(key=lambda r: r["accuracy"], reverse=True)
    cmp_dir = os.path.join(args.out_dir, "_test_comparison"); os.makedirs(cmp_dir, exist_ok=True)

    def write_csv(p):
        with open(p, "w", newline="", encoding="utf-8-sig") as fp:
            w = csv.writer(fp, delimiter=";")
            w.writerow(["approach", "n", "overall"] + QUESTION_TYPES)
            for r in table:
                w.writerow([r["approach"], r["n"], "%.4f" % r["accuracy"]] + ["%.4f" % r["acc_" + t] for t in QUESTION_TYPES])

    def write_md(p):
        lines = ["# VQA comparison on the test set (shared rows, n=%d)\n" % len(rows),
                 "| approach | overall | Simple_C | Complex_C | Yes/No | Condition |",
                 "|---|---|---|---|---|---|"]
        for r in table:
            lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
                r["approach"], r["accuracy"], r["acc_Simple_Counting"], r["acc_Complex_Counting"],
                r["acc_Yes_No"], r["acc_Condition_Recognition"]))
        open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    def write_fig(p):
        labels = [r["approach"] for r in table]
        metrics = [("overall", "accuracy"), ("Simple_C", "acc_Simple_Counting"), ("Complex_C", "acc_Complex_Counting"),
                   ("Yes/No", "acc_Yes_No"), ("Condition", "acc_Condition_Recognition")]
        x = np.arange(len(labels)); w = 0.16
        fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(labels)), 5))
        for k, (lab, key) in enumerate(metrics):
            ax.bar(x + (k - 2) * w, [r[key] for r in table], w, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
        ax.set_title("VQA on test - best model of each approach (shared set, n=%d)" % len(rows))
        ax.legend(ncol=5, fontsize=8); ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)

    atomic_write(write_csv, os.path.join(cmp_dir, "summary.csv"))
    atomic_write(write_md, os.path.join(cmp_dir, "summary.md"))
    atomic_write(write_fig, os.path.join(cmp_dir, "comparison.png"))
    print("\nsaved comparison -> %s (summary.csv, summary.md, comparison.png)" % cmp_dir)


def main():
    if args.approach == "all":
        compare_on_test()
    elif args.approach in TRAINED:
        run_classifier("test" if args.test else "train")
    elif args.approach == "clip":
        run_clip("test" if args.test else "val")
    elif args.approach == "lora":
        run_lora("test" if args.test else "val")
    else:
        facts = load_facts()
        rows = read_rows(args.test_csv if args.test else args.val_csv)
        run_symbolic(rows, facts, "test" if args.test else "val")


if __name__ == "__main__":
    main()
