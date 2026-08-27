# Two ways to run it:
# Single experiment:   python train_ssl.py --batch_size 16 --lr 1e-4
# Agentic search:      python train_ssl.py --agent --n_trials 6

import os
import csv
import math
import json
import re
import copy
import argparse

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

import utils

try:
    import mlflow
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False


parser = argparse.ArgumentParser()
parser.add_argument("--images_csv", default=r".\prepared_data\ssl_images.csv")
parser.add_argument("--out_dir", default=r".\experiments\ssl_mae")
parser.add_argument("--model_name", default="facebook/vit-mae-base")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--mask_ratio", type=float, default=0.75, help="kept fixed (not searched) so PSNR/SSIM stay comparable")
parser.add_argument("--norm_pix_loss", type=int, default=0, help="1 keeps per-patch normalised loss; 0 makes PSNR/SSIM and figures interpretable")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=50, help="max epochs (early stopping usually stops earlier)")
parser.add_argument("--patience", type=int, default=8, help="early stopping patience on val_loss (0 disables it)")
parser.add_argument("--lr", type=float, default=1.5e-4)
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--dropout", type=float, default=0.0, help="dropout in the ViT encoder")
parser.add_argument("--val_frac", type=float, default=0.05)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--vis_every", type=int, default=10, help="save reconstruction figures every N epochs")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", action="store_true", help="resume from last checkpoint")
parser.add_argument("--exp_name", default="ssl_mae")
parser.add_argument("--use_mlflow", type=int, default=0, help="0 = off (default; use compare_experiments.py). MLflow fails on this lab machine")
# agentic search
parser.add_argument("--agent", action="store_true", help="let an LLM search hyper-parameters over several experiments")
parser.add_argument("--n_trials", type=int, default=6, help="number of experiments the agent runs")
parser.add_argument("--llm_name", default="ank028/Llama-3.2-1B-Instruct-medmcqa")
parser.add_argument("--llm_cpu", action="store_true", help="keep the reasoning LLM on CPU to save GPU memory")
args = parser.parse_args()


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ABBR = {"batch_size": "bs", "lr": "lr", "weight_decay": "wd", "dropout": "do",
        "mask_ratio": "mask", "img_size": "img", "epochs": "ep", "norm_pix_loss": "npl",
        "val_frac": "val", "num_workers": "nw", "seed": "seed", "vis_every": "vis"}
SKIP = {"images_csv", "out_dir", "exp_name", "use_mlflow", "resume", "model_name",
        "agent", "n_trials", "llm_name", "llm_cpu", "patience"}


def run_folder_name(cfg):
    parts = []
    for key, value in sorted(vars(cfg).items()):
        if key in SKIP or value == parser.get_default(key):
            continue
        name = ABBR.get(key, key)
        val = ("%g" % value) if isinstance(value, float) else str(value)
        parts.append("%s%s" % (name, val))
    return "_".join(parts) if parts else "default"


class SSLImageDataset(data.Dataset):
    def __init__(self, csv_path, img_size):
        with open(csv_path, encoding="utf-8-sig") as fp:
            reader = csv.DictReader(fp, delimiter=";")
            self.paths = [row["image"] for row in reader]
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        return self.transform(img)


def decode_outputs(model, out, images):
    rec = model.unpatchify(out.logits).float()
    pixel_dim = model.config.patch_size ** 2 * 3
    mask = out.mask.unsqueeze(-1).repeat(1, 1, pixel_dim)
    mask = model.unpatchify(mask).float() 

    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    orig = (images * std + mean).clamp(0, 1)
    rec = (rec * std + mean).clamp(0, 1)
    masked = orig * (1 - mask)
    reconstruction = orig * (1 - mask) + rec * mask
    return orig, masked, reconstruction, mask


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    meter = utils.AverageMeter()
    for step, images in enumerate(loader):
        images = images.to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss = model(pixel_values=images).loss
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        meter.update(loss.item(), images.size(0))
        if step % 20 == 0:
            print("   batch %4d/%d  loss %.4f" % (step, len(loader), loss.item()))
    return meter.avg


def evaluate(model, loader, device, seed):
    model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)

    loss_m, psnr_m, ssim_m = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            out = model(pixel_values=images)
            orig, _, reconstruction, mask = decode_outputs(model, out, images)
            loss_m.update(out.loss.item(), images.size(0))
            mse = ((orig - reconstruction) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            psnr_val = 99.0 if mse.item() <= 0 else 10.0 * math.log10(1.0 / mse.item())
            psnr_m.update(psnr_val, images.size(0))
            ssim_m.update(utils.ssim(reconstruction, orig), images.size(0))

    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return loss_m.avg, psnr_m.avg, ssim_m.avg


def save_reconstructions(model, images, device, out_path):
    try:
        import matplotlib.pyplot as plt
        model.eval()
        with torch.no_grad():
            out = model(pixel_values=images.to(device))
        orig, masked, reconstruction, _ = decode_outputs(model, out, images.to(device))
        orig, masked, reconstruction = orig.cpu(), masked.cpu(), reconstruction.cpu()

        n = min(3, images.size(0))
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        panels = [(masked, "masked input"), (reconstruction, "reconstruction"), (orig, "original (GT)")]
        for i in range(n):
            for j, (img, title) in enumerate(panels):
                ax = axes[i, j] if n > 1 else axes[j]
                ax.imshow(img[i].permute(1, 2, 0).numpy())
                ax.set_title(title); ax.axis("off")
        plt.tight_layout(); plt.savefig(out_path, dpi=100); plt.close(fig)
    except Exception as e:
        print("   (could not save reconstructions: %s)" % e)


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
        logger.info("to view ->  mlflow ui --backend-store-uri %s" % uri)
        return True
    except Exception as e:
        logger.info("mlflow disabled (setup failed: %s)" % e)
        return False


def train_one_experiment(cfg):
    from transformers import ViTMAEForPreTraining

    utils.set_seed(cfg.seed)
    run_dir = os.path.join(cfg.out_dir, run_folder_name(cfg))
    os.makedirs(run_dir, exist_ok=True)
    logger = utils.get_logger("ssl", os.path.join(run_dir, "train.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s | run dir: %s" % (device, run_dir))
    logger.info("config: %s" % vars(cfg))

    full = SSLImageDataset(cfg.images_csv, cfg.img_size)
    n_val = max(1, int(cfg.val_frac * len(full)))
    n_train = len(full) - n_val
    train_set, val_set = data.random_split(full, [n_train, n_val],
                                           generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = data.DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    logger.info("images: %d (train %d / val %d)" % (len(full), n_train, n_val))

    model = ViTMAEForPreTraining.from_pretrained(
        cfg.model_name, mask_ratio=cfg.mask_ratio, norm_pix_loss=bool(cfg.norm_pix_loss),
        hidden_dropout_prob=cfg.dropout, attention_probs_dropout_prob=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}
    start_epoch = 0
    best_val = float("inf")
    epochs_no_improve = 0
    last_ckpt = os.path.join(run_dir, "last.pt")

    if cfg.resume and os.path.exists(last_ckpt):
        ckpt = utils.load_checkpoint(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"]
        best_val = ckpt.get("best_val", float("inf"))
        logger.info("resumed from epoch %d" % start_epoch)

    use_mlflow = setup_mlflow(cfg, run_folder_name(cfg), logger)
    vis_batch = next(iter(val_loader))
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    for epoch in range(start_epoch, cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_psnr, val_ssim = evaluate(model, val_loader, device, cfg.seed)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        logger.info("epoch %3d/%d  train_loss %.4f  val_loss %.4f  val_psnr %.2f  val_ssim %.4f" %
                    (epoch + 1, cfg.epochs, train_loss, val_loss, val_psnr, val_ssim))

        if use_mlflow:
            try:
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                    "val_psnr": val_psnr, "val_ssim": val_ssim}, step=epoch + 1)
            except Exception as e:
                logger.info("mlflow logging failed (%s); disabling mlflow, training continues" % e)
                use_mlflow = False

        utils.plot_history(history, os.path.join(run_dir, "curves.png"), title="SSL / MAE")
        utils.save_history_csv(history, os.path.join(run_dir, "history.csv"))
        state = {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "history": history, "best_val": best_val}
        utils.save_checkpoint(state, last_ckpt)

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            state["best_val"] = best_val
            utils.save_checkpoint(state, os.path.join(run_dir, "best.pt"))
            model.save_pretrained(os.path.join(run_dir, "encoder_best"))
        else:
            epochs_no_improve += 1

        if (epoch + 1) % cfg.vis_every == 0 or (epoch + 1) == cfg.epochs:
            save_reconstructions(model, vis_batch, device, os.path.join(fig_dir, "recon_epoch_%03d.png" % (epoch + 1)))

        if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
            logger.info("early stopping at epoch %d (no val improvement for %d epochs)" % (epoch + 1, cfg.patience))
            break

    logger.info("done. best val_loss = %.4f" % best_val)
    if use_mlflow:
        try:
            mlflow.end_run()
        except Exception:
            pass

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    bi = min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])
    return {
        "run_dir": run_dir,
        "history": history,
        "best_epoch": bi + 1,
        "train_loss_best": history["train_loss"][bi],
        "val_loss_best": history["val_loss"][bi],
        "val_psnr_best": history["val_psnr"][bi],
        "val_ssim_best": history["val_ssim"][bi],
        "overfit_gap": history["val_loss"][bi] - history["train_loss"][bi],
    }


TUNABLE = ["lr", "batch_size", "weight_decay", "dropout"]


class MAE_Agent:
    def __init__(self, llm_name, force_cpu=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading LLM for agentic reasoning: %s" % llm_name)
        self.device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, dtype=dtype).to(self.device)

    def decide_next_config(self, history, step):
        if step == 1:
            return {"lr": 1.5e-4, "batch_size": 32, "weight_decay": 0.05, "dropout": 0.0}

        past_configs = [{k: h[k] for k in TUNABLE} for h in history]
        formatted = [{
            "exp_id": h["exp_id"], "lr": h["lr"], "batch_size": h["batch_size"],
            "weight_decay": h["weight_decay"], "dropout": h["dropout"],
            "train_loss_best": round(h["train_loss_best"], 4), "val_loss_best": round(h["val_loss_best"], 4),
            "val_psnr_best": round(h["val_psnr_best"], 2), "val_ssim_best": round(h["val_ssim_best"], 4),
            "overfit_gap": round(h["overfit_gap"], 4),
        } for h in history]

        base_prompt = """You are an expert ML engineer tuning a Masked Autoencoder (MAE). Higher val_psnr and
val_ssim are better; overfit_gap is (val_loss - train_loss) at the best epoch (large = overfitting).
History of experiments:
%s

Design the next configuration. Rules:
1. Overfitting (large overfit_gap): increase 'weight_decay' (e.g. 0.05, 0.1) and/or 'dropout' (0.1, 0.2).
2. Underfitting (train_loss_best still high / low psnr-ssim): reduce regularization (dropout=0, weight_decay=1e-4) or change 'lr' (5e-5, 1e-4, 3e-4).
3. If results plateau, change 'batch_size' (8, 16, 32).
4. Change at least TWO parameters versus the last experiment and do NOT repeat any past configuration.
Provide ONLY a valid JSON dictionary with exactly these keys: 'lr', 'batch_size', 'weight_decay', 'dropout'. No extra text.""" % json.dumps(formatted, indent=2)

        prompt = base_prompt
        for attempt in range(3):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.llm.generate(**inputs, max_new_tokens=200, do_sample=True,
                                            temperature=0.7 + 0.1 * attempt,
                                            pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print("LLM proposal (attempt %d): %s" % (attempt + 1, response.strip()[:200]))
            try:
                match = re.search(r"\{[^{}]*\}", response)
                if not match:
                    raise ValueError("no JSON found")
                config = {k: v for k, v in json.loads(match.group(0)).items() if k in TUNABLE}
                if not all(k in config for k in TUNABLE):
                    raise ValueError("missing keys")
                if {k: config[k] for k in TUNABLE} in past_configs:
                    print("repeated configuration, asking again")
                    prompt = base_prompt + "\n\nWARNING: that configuration was already tried, propose a strictly different one."
                    continue
                return config
            except Exception as e:
                print("could not parse (%s), retrying" % e)
                prompt = base_prompt + "\n\nWARNING: reply with ONLY a valid JSON dictionary."

        print("[*] using deterministic fallback")
        options = [
            {"lr": 1e-4, "batch_size": 32, "weight_decay": 0.05, "dropout": 0.1},
            {"lr": 3e-4, "batch_size": 32, "weight_decay": 1e-4, "dropout": 0.0},
            {"lr": 5e-5, "batch_size": 16, "weight_decay": 0.1, "dropout": 0.2},
            {"lr": 1e-4, "batch_size": 16, "weight_decay": 0.05, "dropout": 0.0},
            {"lr": 2e-4, "batch_size": 8, "weight_decay": 1e-4, "dropout": 0.1},
        ]
        return options[(step - 2) % len(options)]


def append_experiment_row(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(row.keys()), delimiter=";")
        if not exists:
            w.writeheader()
        w.writerow(row)


def select_best(records):
    if len(records) == 1:
        return records[0]
    psnrs = [r["val_psnr_best"] for r in records]
    ssims = [r["val_ssim_best"] for r in records]
    overs = [r["overfit_gap"] for r in records]

    def norm(vals, v, higher_better=True):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return 1.0
        x = (v - lo) / (hi - lo)
        return x if higher_better else 1.0 - x

    for r in records:
        r["score"] = (0.4 * norm(ssims, r["val_ssim_best"]) +
                      0.4 * norm(psnrs, r["val_psnr_best"]) +
                      0.2 * norm(overs, r["overfit_gap"], higher_better=False))
    return max(records, key=lambda r: r["score"])


def run_agent():
    agent = MAE_Agent(args.llm_name, force_cpu=args.llm_cpu)
    history = []
    summary_csv = os.path.join(args.out_dir, "experiments_history.csv")
    os.makedirs(args.out_dir, exist_ok=True)

    for step in range(1, args.n_trials + 1):
        print("\nTrial %d/%d" % (step, args.n_trials))
        config = agent.decide_next_config(history, step)
        print("next config: %s" % config)

        cfg = copy.deepcopy(args)
        cfg.agent = False 
        for k, v in config.items():
            setattr(cfg, k, type(getattr(args, k))(v)) 

        m = train_one_experiment(cfg)
        record = {"exp_id": "trial_%d" % step, "folder": os.path.basename(m["run_dir"]),
                  "lr": cfg.lr, "batch_size": cfg.batch_size, "weight_decay": cfg.weight_decay,
                  "dropout": cfg.dropout, "best_epoch": m["best_epoch"],
                  "train_loss_best": m["train_loss_best"], "val_loss_best": m["val_loss_best"],
                  "val_psnr_best": m["val_psnr_best"], "val_ssim_best": m["val_ssim_best"],
                  "overfit_gap": m["overfit_gap"], "run_dir": m["run_dir"]}
        history.append(record)
        append_experiment_row(summary_csv, record)
        print("result: %s | val_loss %.4f | psnr %.2f | ssim %.4f | overfit %.4f" %
              (record["folder"], record["val_loss_best"], record["val_psnr_best"],
               record["val_ssim_best"], record["overfit_gap"]))

    best = select_best(history)
    ranked = sorted(history, key=lambda r: r["score"], reverse=True)
    print("\nRanking (best first):")
    for r in ranked:
        print("  %-28s score %.3f | psnr %.2f | ssim %.4f | overfit %.4f" %
              (r["folder"], r["score"], r["val_psnr_best"], r["val_ssim_best"], r["overfit_gap"]))

    best_file = os.path.join(args.out_dir, "best_experiment.txt")
    with open(best_file, "w", encoding="utf-8") as fp:
        fp.write("Best experiment: %s\n" % best["folder"])
        fp.write("Folder: %s\n" % best["run_dir"])
        fp.write("Best model checkpoint: %s\n" % os.path.join(best["run_dir"], "encoder_best"))
        fp.write("val_psnr=%.2f  val_ssim=%.4f  val_loss=%.4f  overfit_gap=%.4f  score=%.3f\n" %
                 (best["val_psnr_best"], best["val_ssim_best"], best["val_loss_best"],
                  best["overfit_gap"], best["score"]))
    print("\nBEST MODEL -> folder '%s'  (encoder in %s)" %
          (best["folder"], os.path.join(best["run_dir"], "encoder_best")))
    print("Details written to %s" % best_file)


def main():
    if args.agent:
        run_agent()
    else:
        train_one_experiment(args)


if __name__ == "__main__":
    main()
