# Src/Training/Trainer.py
import os
import torch
from torch.optim import Adam
from tqdm import tqdm

from Src.Model.Model import CaptchaSolverModel
from Src.Model.Loss import ModelLoss
from Src.Data.DataLoader import CaptchaDataLoader
from Src.Utils.TargetPreparer import TargetPreparer


class Trainer:
    def __init__(
        self,
        data_dir,
        num_classes=36,
        grid_height=20,
        grid_width=80,
        device='cuda',
        batch_size=32,
        lr=1e-3,
        ckpt_dir="./checkpoints",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # model + loss
        self.model = CaptchaSolverModel(
            num_classes=num_classes, grid_height=grid_height, grid_width=grid_width
        ).to(self.device)
        self.criterion = ModelLoss(
            num_classes=num_classes, GridHeight=grid_height, GridWidth=grid_width
        ).to(self.device)

        # optimizer
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        # dataloader
        self.train_loader = CaptchaDataLoader(
            data_dir, batch_size=batch_size, shuffle=True
        )

        # target preparer
        self.target_preparer = TargetPreparer(
            GridHeight=grid_height,
            GridWidth=grid_width,
            num_classes=num_classes,
            img_width=640,
            img_height=160,
        )

        # checkpoints
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_loss = float("inf")
        self.best_path = os.path.join(self.ckpt_dir, "best.pth")

    def train_epoch(self):
        self.model.train()
        running = {
            "total_loss": 0.0,
            "bbox_loss": 0.0,
            "obj_loss": 0.0,
            "class_loss": 0.0,
        }
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for batch in pbar:
            images = batch["Image"].to(self.device)
            targets = self.target_preparer(batch).to(self.device)

            preds = self.model(images)
            loss_dict = self.criterion(preds, targets)

            self.optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            self.optimizer.step()

            num_batches += 1
            for k in running:
                if k in loss_dict:
                    running[k] += float(loss_dict[k])

            pbar.set_postfix(
                {
                    "tot": f"{loss_dict['total_loss']:.2f}",
                    "bbox": f"{loss_dict.get('bbox_loss', 0):.2f}",
                    "obj": f"{loss_dict.get('obj_loss', 0):.2f}",
                    "cls": f"{loss_dict.get('class_loss', 0):.2f}",
                }
            )

        # averages
        for k in running:
            running[k] /= max(1, num_batches)
        return running

    def _maybe_save_best(self, epoch_avg):
        cur = epoch_avg["total_loss"]
        if cur < self.best_loss:
            self.best_loss = cur
            torch.save(
                {
                    "epoch_avg": epoch_avg,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_loss": self.best_loss,
                },
                self.best_path,
            )
            print(f"✅ Saved new best to {self.best_path} (loss={self.best_loss:.4f})")

    def train(self, epochs):
        for ep in range(1, epochs + 1):
            print(f"\nEpoch {ep}/{epochs}")
            avg = self.train_epoch()
            print(
                "avg -> "
                f"total={avg['total_loss']:.4f} | "
                f"bbox={avg['bbox_loss']:.4f} | "
                f"obj={avg['obj_loss']:.4f} | "
                f"cls={avg['class_loss']:.4f}"
            )
            self._maybe_save_best(avg)

        print("\nTraining done.")
        print(f"Best loss: {self.best_loss:.4f}  ->  checkpoint: {self.best_path}")
