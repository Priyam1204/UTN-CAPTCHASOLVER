import torch
import os
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from Src.Model.Model import CaptchaSolverModel
from Src.Model.Loss import ModelLoss
from Src.Data.DataLoader import CaptchaDataLoader
from Src.Utils.TargetPreparer import TargetPreparer
from tqdm.auto import tqdm
import sys
import csv
from datetime import datetime
from collections import defaultdict

class Trainer:
    def __init__(self, data_dir, val_data_dir=None,  # NEW: validation dir
                 num_classes=36, grid_height=20, grid_width=80,
                 device='cuda', optimizer_type='adam', learning_rate=0.001,
                 weight_decay=1e-4, save_dir='./checkpoints'):
        
        # Basic setup
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.grid_height = grid_height
        self.grid_width = grid_width

        # Make log dir + show absolute paths up front
        os.makedirs(self.save_dir, exist_ok=True)

        # CSV paths (headers written lazily once we know the component keys)
        self.csv_path_batches = os.path.join(save_dir, "train_batches.csv")
        self.csv_path_epochs  = os.path.join(save_dir, "train_epochs.csv")
        self.csv_path_val_epochs = os.path.join(save_dir, "val_epochs.csv")  # NEW

        # Make sure files exist so the folder shows up immediately
        for p in (self.csv_path_batches, self.csv_path_epochs, self.csv_path_val_epochs):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if not os.path.exists(p):
                open(p, "a").close()

        print(f"[TRAINER MODULE] {__file__}")
        print(f"[LOG DIR] {os.path.abspath(self.save_dir)}")
        print("[CSV PATHS]")
        print(f"  batches: {os.path.abspath(self.csv_path_batches)}")
        print(f"  epochs : {os.path.abspath(self.csv_path_epochs)}")
        print(f"  val    : {os.path.abspath(self.csv_path_val_epochs)}")

        # Model / loss / optim / sched
        self.model = CaptchaSolverModel(
            num_classes=num_classes,
            grid_height=grid_height,
            grid_width=grid_width
        ).to(self.device)

        self.criterion = ModelLoss(
            num_classes=num_classes,
            GridHeight=grid_height,
            GridWidth=grid_width
        ).to(self.device)

        self.optimizer = self._get_optimizer(optimizer_type, learning_rate, weight_decay)
        self.scheduler = self._get_scheduler()

        # Data
        self.train_loader = CaptchaDataLoader(
            data_dir,
            batch_size=32,
            shuffle=True,
            use_geo_aug=True
        )

        self.has_val = val_data_dir is not None
        if self.has_val:
            self.val_loader = CaptchaDataLoader(
                val_data_dir,
                batch_size=32,
                shuffle=False,
                use_geo_aug=False
            )
        else:
            self.val_loader = None

        # Targets
        self.target_preparer = TargetPreparer(
            GridHeight=grid_height,
            GridWidth=grid_width,
            num_classes=num_classes,
            img_width=640,
            img_height=160
        )

        # Tracking
        self.best_loss = float('inf')  # still based on TRAIN total; can switch to VAL later
        self.train_losses = []

        # Internal flag for one-time prints
        self._printed_loss_keys = False

    # ---------- CSV helpers ----------
    def _csv_append(self, path, row):
        try:
            with open(path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            print(f"[CSV ERROR] Could not write to {os.path.abspath(path)}: {e}")

    def _ensure_csv_with_header(self, path, header):
        need_header = not os.path.exists(path) or os.path.getsize(path) == 0
        if need_header:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)
            print(f"[CSV INIT] Created header: {os.path.abspath(path)}")

    def _ensure_val_header(self):
        self._ensure_csv_with_header(self.csv_path_val_epochs, ["timestamp","epoch","avg_val_loss","lr"])

    # Preserve the incoming order of component_keys (don’t sort)
    def _build_batch_header(self, component_keys):
        others = [k for k in component_keys if k != 'total_loss']
        return ["timestamp","epoch","batch_idx","num_batches","lr","total_loss"] + others + ["avg_total_loss"]

    def _build_epoch_header(self, component_keys):
        others = [k for k in component_keys if k != 'total_loss']
        return ["timestamp","epoch","lr","is_best","avg_total_loss"] + [f"avg_{k}" for k in others]

    # ---------- optim/sched ----------
    def _get_optimizer(self, optimizer_type, learning_rate, weight_decay):
        ot = optimizer_type.lower()
        if ot == 'adam':
            return Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif ot == 'adamw':
            return AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif ot == 'sgd':
            return SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _get_scheduler(self):
        return StepLR(self.optimizer, step_size=10, gamma=0.1)
        # Alternatives:
        # return CosineAnnealingLR(self.optimizer, T_max=50)
        # return ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

    # ---------- checkpoints ----------
    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'model_config': {
                'num_classes': self.num_classes,
                'grid_height': self.grid_height,
                'grid_width': self.grid_width
            }
        }
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        print(f"Saved checkpoint to {os.path.abspath(latest_path)}")

        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {os.path.abspath(best_path)}")

        epoch_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint['train_losses']

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
        return epoch

    # ---------- train/val loops ----------
    def train_epoch(self, epoch_idx: int, total_epochs: int):
        self.model.train()
        total_loss_sum = 0.0
        num_batches = len(self.train_loader)
        disable_bar = not sys.stdout.isatty()
        pbar = tqdm(self.train_loader, total=num_batches,
                    desc=f"Epoch {epoch_idx}/{total_epochs}", leave=False, disable=disable_bar)

        comp_sums = defaultdict(float)  # per-component accumulation for epoch averages
        batch_header = None
        epoch_header = None

        # lock a presentation-friendly key order; only include keys that exist
        preferred = ["total_loss", "bbox_loss", "obj_loss", "class_loss"]
        ordered_keys = None

        for batch_idx, batch in enumerate(pbar):
            images = batch['Image'].to(self.device)
            targets = self.target_preparer(batch).to(self.device)

            predictions = self.model(images)
            loss_dict = self.criterion(predictions, targets)  # dict with total + components

            # show loss keys once for sanity
            if not self._printed_loss_keys:
                print("[LOSS KEYS]", list(loss_dict.keys()))
                self._printed_loss_keys = True

            # Filter to actual losses only (skip num_pos/num_neg)
            loss_keys = [k for k in loss_dict.keys() if k == "total_loss" or k.endswith("_loss")]
            if ordered_keys is None:
                ordered_keys = [k for k in preferred if k in loss_keys] + [k for k in loss_keys if k not in preferred]

            # Backprop on total
            loss = loss_dict['total_loss']
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # scalars
            loss_f = float(loss.detach().item())
            total_loss_sum += loss_f
            avg_total = total_loss_sum / (batch_idx + 1)
            current_lr = self.optimizer.param_groups[0]['lr']

            # accumulate per-component
            for k in ordered_keys:
                v = loss_dict[k]
                v_float = float(v.detach().item()) if torch.is_tensor(v) else float(v)
                comp_sums[k] += v_float

            # init header on first batch
            if batch_header is None:
                batch_header = self._build_batch_header(ordered_keys)
                self._ensure_csv_with_header(self.csv_path_batches, batch_header)

            # progress bar
            pbar.set_postfix({"total": f"{loss_f:.4f}", "lr": f"{current_lr:.6f}"})

            # write batch row
            row = [
                datetime.utcnow().isoformat(timespec="seconds"),
                epoch_idx,
                batch_idx,
                num_batches,
                f"{current_lr:.8f}",
                f"{float(loss_dict['total_loss'].detach().item()):.6f}",
            ]
            # components between total and avg_total_loss, keep header order
            comp_cols = [c for c in batch_header[6:-1]]
            for c in comp_cols:
                val = loss_dict.get(c, 0.0)
                val = float(val.detach().item()) if torch.is_tensor(val) else float(val)
                row.append(f"{val:.6f}")
            row.append(f"{avg_total:.6f}")  # running avg of total
            self._csv_append(self.csv_path_batches, row)

        # epoch header & averages
        if epoch_header is None:
            epoch_header = self._build_epoch_header(ordered_keys)
            self._ensure_csv_with_header(self.csv_path_epochs, epoch_header)

        avg_total_epoch = comp_sums['total_loss'] / num_batches if num_batches else 0.0
        is_best = avg_total_epoch < self.best_loss
        if is_best:
            self.best_loss = avg_total_epoch
            self.save_checkpoint(epoch_idx, avg_total_epoch, is_best=True)

        epoch_row = [
            datetime.utcnow().isoformat(timespec="seconds"),
            epoch_idx,
            f"{self.optimizer.param_groups[0]['lr']:.8f}",
            int(is_best),
            f"{avg_total_epoch:.6f}",
        ]
        for k in [k for k in ordered_keys if k != 'total_loss']:
            epoch_row.append(f"{(comp_sums[k]/num_batches):.6f}")
        self._csv_append(self.csv_path_epochs, epoch_row)

        print(f"  Train Avg Total: {avg_total_epoch:.4f}")
        return avg_total_epoch

    @torch.no_grad()
    def validate_epoch(self, epoch_idx: int):
        """Validation: epoch-level TOTAL loss only (simple & presentation-friendly)."""
        if not self.has_val or self.val_loader is None:
            return None
        self.model.eval()
        total_val = 0.0
        n = 0
        for batch in self.val_loader:
            images = batch['Image'].to(self.device)
            targets = self.target_preparer(batch).to(self.device)
            preds = self.model(images)
            loss_dict = self.criterion(preds, targets)
            total_val += float(loss_dict['total_loss'].detach().item())
            n += 1
        if n == 0:
            return None
        avg_val = total_val / n
        self._ensure_val_header()
        self._csv_append(self.csv_path_val_epochs, [
            datetime.utcnow().isoformat(timespec="seconds"),
            epoch_idx,
            f"{avg_val:.6f}",
            f"{self.optimizer.param_groups[0]['lr']:.8f}",
        ])
        return avg_val

    def train(self, epochs, resume_from=None, save_every=5):
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)

        print(f"Starting training from epoch {start_epoch + 1} to {epochs}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(start_epoch, epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*50}")

            # Train
            epoch_loss = self.train_epoch(epoch_idx=epoch + 1, total_epochs=epochs)

            # Validate (total only)
            val_loss = self.validate_epoch(epoch_idx=epoch + 1)

            # Scheduler
            self.scheduler.step()

            # Track (legacy list)
            self.train_losses.append(epoch_loss)

            # Best check already done in train_epoch (based on train total)
            if val_loss is not None:
                print(f"  Train Avg Loss: {epoch_loss:.4f} | Val Avg Loss: {val_loss:.4f}")
            else:
                print(f"  Train Avg Loss: {epoch_loss:.4f}")
            print(f"  Best Train Total: {self.best_loss:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # periodic checkpoint
            if ((epoch + 1) % save_every == 0) or (epoch == epochs - 1):
                self.save_checkpoint(epoch + 1, epoch_loss, is_best=False)

        print(f"\nTraining completed! Best train total loss: {self.best_loss:.4f}")
        return self.train_losses
