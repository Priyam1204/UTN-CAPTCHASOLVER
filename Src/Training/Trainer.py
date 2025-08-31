# Src/Training/Trainer.py
import os
import csv
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Src.Model.Model import CaptchaSolverModel
from Src.Model.Loss import ModelLoss
from Src.Data.DataLoader import CaptchaDataLoader
from Src.Utils.TargetPreparer import TargetPreparer


# -------------------- small utilities --------------------

class AverageMeter:
    def __init__(self, keys):
        self.keys = list(keys)
        self.reset()
    def reset(self):
        self.sums = {k: 0.0 for k in self.keys}
        self.cnts = 0
    def update(self, loss_dict: Dict[str, float], n: int):
        self.cnts += n
        for k in self.keys:
            if k in loss_dict:
                self.sums[k] += float(loss_dict[k]) * n
    def averages(self) -> Dict[str, float]:
        denom = max(1, self.cnts)
        return {k: self.sums.get(k, 0.0) / denom for k in self.keys}

class CSVLogger:
    def __init__(self, path: str, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._new = not os.path.exists(path)
        self.fh = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=fieldnames)
        if self._new:
            self.writer.writeheader()
    def log(self, row: Dict[str, Any]):
        self.writer.writerow(row); self.fh.flush()
    def close(self):
        try: self.fh.close()
        except: pass

class CheckpointManager:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.best_val = math.inf
    def path(self, tag: str):
        return os.path.join(self.out_dir, f"{tag}.pth")
    def save(self, tag: str, state: Dict[str, Any]):
        torch.save(state, self.path(tag))
    def save_last_and_best(self, state: Dict[str, Any], val_loss: float):
        # always save last
        self.save("last", state)
        # update best
        if val_loss <= self.best_val:
            self.best_val = val_loss
            self.save("best", state)
            return True
        return False
    def try_load(self, tag: str) -> Optional[Dict[str, Any]]:
        p = self.path(tag)
        if os.path.exists(p):
            return torch.load(p, map_location="cpu")
        return None


# -------------------- config --------------------

@dataclass
class TrainConfig:
    data_dir: str = ""
    val_data_dir: Optional[str] = None     # if None, uses data_dir with val settings
    num_classes: int = 36
    grid_height: int = 20
    grid_width: int = 80
    img_width: int = 640
    img_height: int = 160

    device: str = "cuda"
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 30
    grad_clip: Optional[float] = 1.0
    use_amp: bool = True

    ckpt_dir: str = "./checkpoints"
    log_dir: Optional[str] = None          # default: same as ckpt_dir
    log_every: int = 0                     # per-iter logging to CSV/TB if > 0
    resume: bool = False                   # resume from last.pth if available


# -------------------- main trainer --------------------

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # model + loss
        self.model = CaptchaSolverModel(
            num_classes=cfg.num_classes,
            grid_height=cfg.grid_height,
            grid_width=cfg.grid_width,
        ).to(self.device)

        self.criterion = ModelLoss(
            num_classes=cfg.num_classes,
            GridHeight=cfg.grid_height,
            GridWidth=cfg.grid_width,
        ).to(self.device)

        # optimizer
        self.optimizer = Adam(self.model.parameters(), lr=cfg.lr)

        # data loaders
        self.train_loader = CaptchaDataLoader(
            cfg.data_dir,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            use_geo_aug=True,       # boxes follow image transforms
        )
        # prefer a separate validation dir if provided
        val_dir = cfg.val_data_dir if cfg.val_data_dir else cfg.data_dir
        self.val_loader = CaptchaDataLoader(
            val_dir,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            use_geo_aug=False,      # determinism for val
        )

        # targets
        self.target_preparer = TargetPreparer(
            GridHeight=cfg.grid_height,
            GridWidth=cfg.grid_width,
            num_classes=cfg.num_classes,
            img_width=cfg.img_width,
            img_height=cfg.img_height,
        )

        # logging + checkpoints
        self.ckpt = CheckpointManager(cfg.ckpt_dir)
        log_root = cfg.log_dir or cfg.ckpt_dir
        os.makedirs(log_root, exist_ok=True)
        self.tb = SummaryWriter(log_root)
        self.csv = CSVLogger(
            os.path.join(log_root, "log.csv"),
            fieldnames=["epoch", "phase", "step", "loss_total", "loss_bbox", "loss_obj", "loss_cls", "lr"]
        )

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

        # bookkeeping
        self.global_step = 0
        self.start_epoch = 1
        self.best_val = math.inf

        # maybe resume
        if cfg.resume:
            last = self.ckpt.try_load("last")
            if last is not None:
                self.model.load_state_dict(last["model"])
                if "optimizer" in last and last["optimizer"] is not None:
                    self.optimizer.load_state_dict(last["optimizer"])
                self.best_val = float(last.get("best_val", math.inf))
                self.start_epoch = int(last.get("epoch", 0)) + 1
                self.ckpt.best_val = self.best_val
                print(f"[resume] from epoch {self.start_epoch}, best_val={self.best_val:.4f}")
            else:
                print("[resume] no last.pth found; starting fresh.")

    # ---------- core loops ----------

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        meter = AverageMeter(["total_loss", "bbox_loss", "obj_loss", "class_loss"])
        pbar = tqdm(self.val_loader, desc=f"Val {epoch:03d}", leave=False)
        for batch in pbar:
            images = batch["Image"].to(self.device, non_blocking=True)
            targets = self.target_preparer(batch).to(self.device)
            preds = self.model(images)
            loss_dict = self.criterion(preds, targets)  # expects keys seen below

            # unify keys in case your criterion uses slightly different names
            normalized = {
                "total_loss": float(loss_dict.get("total_loss", 0.0)),
                "bbox_loss":  float(loss_dict.get("bbox_loss", 0.0)),
                "obj_loss":   float(loss_dict.get("obj_loss", 0.0)),
                "class_loss": float(loss_dict.get("class_loss", 0.0)),
            }
            meter.update(normalized, n=images.size(0))
            pbar.set_postfix({"tot": f"{normalized['total_loss']:.3f}"})

        avg = meter.averages()
        # TB + CSV (epoch)
        self.tb.add_scalar("val/total_loss", avg["total_loss"], epoch)
        self.tb.add_scalar("val/bbox_loss",  avg["bbox_loss"],  epoch)
        self.tb.add_scalar("val/obj_loss",   avg["obj_loss"],   epoch)
        self.tb.add_scalar("val/class_loss", avg["class_loss"], epoch)
        self.csv.log({"epoch": epoch, "phase": "val", "step": self.global_step,
                      "loss_total": avg["total_loss"], "loss_bbox": avg["bbox_loss"],
                      "loss_obj": avg["obj_loss"], "loss_cls": avg["class_loss"],
                      "lr": self._get_lr()})
        return avg

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meter = AverageMeter(["total_loss", "bbox_loss", "obj_loss", "class_loss"])

        pbar = tqdm(self.train_loader, desc=f"Train {epoch:03d}", leave=False)
        for it, batch in enumerate(pbar, start=1):
            images = batch["Image"].to(self.device, non_blocking=True)
            targets = self.target_preparer(batch).to(self.device)

            with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                preds = self.model(images)
                loss_dict = self.criterion(preds, targets)

                total_loss = loss_dict["total_loss"]

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()

            if self.cfg.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # accumulate + progress
            normalized = {
                "total_loss": float(loss_dict.get("total_loss", 0.0)),
                "bbox_loss":  float(loss_dict.get("bbox_loss", 0.0)),
                "obj_loss":   float(loss_dict.get("obj_loss", 0.0)),
                "class_loss": float(loss_dict.get("class_loss", 0.0)),
            }
            meter.update(normalized, n=images.size(0))
            pbar.set_postfix({"tot": f"{normalized['total_loss']:.3f}"})

            self.global_step += 1

            # optional per-iter logging
            if self.cfg.log_every and (it % self.cfg.log_every == 0 or it == 1):
                self.tb.add_scalar("train/total_loss_iter", normalized["total_loss"], self.global_step)
                self.tb.add_scalar("train/lr", self._get_lr(), self.global_step)
                self.csv.log({"epoch": epoch, "phase": "train_iter", "step": self.global_step,
                              "loss_total": normalized["total_loss"], "loss_bbox": normalized["bbox_loss"],
                              "loss_obj": normalized["obj_loss"], "loss_cls": normalized["class_loss"],
                              "lr": self._get_lr()})

        avg = meter.averages()
        # TB + CSV (epoch)
        self.tb.add_scalar("train/total_loss", avg["total_loss"], epoch)
        self.tb.add_scalar("train/bbox_loss",  avg["bbox_loss"],  epoch)
        self.tb.add_scalar("train/obj_loss",   avg["obj_loss"],   epoch)
        self.tb.add_scalar("train/class_loss", avg["class_loss"], epoch)
        self.csv.log({"epoch": epoch, "phase": "train", "step": self.global_step,
                      "loss_total": avg["total_loss"], "loss_bbox": avg["bbox_loss"],
                      "loss_obj": avg["obj_loss"], "loss_cls": avg["class_loss"],
                      "lr": self._get_lr()})
        return avg

    # ---------- helpers ----------

    def _get_lr(self) -> float:
        for g in self.optimizer.param_groups:
            return float(g["lr"])
        return 0.0

    # ---------- public API ----------

    def fit(self):
        print(f"[cfg] {asdict(self.cfg)}")
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            train_avg = self.train_epoch(epoch)
            val_avg   = self.validate_epoch(epoch)

            # checkpointing
            state = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_val": min(self.best_val, val_avg["total_loss"]),
            }
            improved = self.ckpt.save_last_and_best(state, val_avg["total_loss"])
            if improved:
                self.best_val = val_avg["total_loss"]
                print(f"✅ epoch {epoch:03d}: new best val={self.best_val:.4f} -> saved best.pth")
            else:
                print(f"[epoch {epoch:03d}] train={train_avg['total_loss']:.4f}  "
                      f"val={val_avg['total_loss']:.4f}  best={self.best_val:.4f}")

        self.csv.close()
        self.tb.flush(); self.tb.close()
        print("\nTraining finished.")
        print(f"Best val loss: {self.best_val:.4f}")
        print(f"Checkpoints in: {self.ckpt.out_dir}")
