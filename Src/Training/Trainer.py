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
import csv  # NEW
from datetime import datetime  # NEW

class Trainer:
    def __init__(self, data_dir, num_classes=36, grid_height=20, grid_width=80, 
                 device='cuda', optimizer_type='adam', learning_rate=0.001, 
                 weight_decay=1e-4, save_dir='./checkpoints'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # --- CSV logging setup (NEW) ---
        self.csv_path_batches = os.path.join(save_dir, "train_batches.csv")
        self.csv_path_epochs = os.path.join(save_dir, "train_epochs.csv")

        def _ensure_csv(path, header):
            need_header = not os.path.exists(path) or os.path.getsize(path) == 0
            if need_header:
                with open(path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

        _ensure_csv(self.csv_path_batches, [
            "timestamp","epoch","batch_idx","num_batches","batch_loss","avg_loss","lr"
        ])
        _ensure_csv(self.csv_path_epochs, [
            "timestamp","epoch","avg_loss","best_loss","lr","is_best"
        ])
        # --------------------------------

        
        # Initialize model
        self.model = CaptchaSolverModel(
            num_classes=num_classes, 
            grid_height=grid_height, 
            grid_width=grid_width
        ).to(self.device)
        
        # Initialize loss function
        self.criterion = ModelLoss(
            num_classes=num_classes, 
            GridHeight=grid_height, 
            GridWidth=grid_width
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer(optimizer_type, learning_rate, weight_decay)
        
        # Initialize learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Initialize data loader (geo augmentation ON)
        self.train_loader = CaptchaDataLoader(
            data_dir,
            batch_size=32,
            shuffle=True,
            use_geo_aug=True
        )
        
        # Initialize target preparer
        self.target_preparer = TargetPreparer(
            GridHeight=grid_height, 
            GridWidth=grid_width, 
            num_classes=num_classes,
            img_width=640, 
            img_height=160
        )
        
        # Training metrics
        self.best_loss = float('inf')
        self.train_losses = []

    def _csv_append(self, path, row):  # NEW
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _get_optimizer(self, optimizer_type, learning_rate, weight_decay):
        """
        Initialize the optimizer based on the specified type.
        """
        if optimizer_type.lower() == 'adam':
            return Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            return AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            return SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def _get_scheduler(self):
        """
        Initialize the learning rate scheduler.
        """
        # You can choose different schedulers based on your needs
        return StepLR(self.optimizer, step_size=10, gamma=0.1)
        # Alternative schedulers:
        # return CosineAnnealingLR(self.optimizer, T_max=50)
        # return ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """
        Save model checkpoint.
        """
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
        
        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        print(f"Saved checkpoint to {latest_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Save epoch-specific checkpoint
        epoch_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        """
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
        
    def train_epoch(self, epoch_idx: int, total_epochs: int):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        disable_bar = not sys.stdout.isatty()
        pbar = tqdm(self.train_loader, total=num_batches, desc=f"Epoch {epoch_idx}/{total_epochs}",
                    leave=False, disable=disable_bar)

        for batch_idx, batch in enumerate(pbar):
            images = batch['Image'].to(self.device)
            targets = self.target_preparer(batch).to(self.device)

            predictions = self.model(images)
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict['total_loss']

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            loss_val = float(loss.detach().item())
            total_loss += loss_val

            avg_loss = total_loss / (batch_idx + 1)
            current_lr = self.optimizer.param_groups[0]['lr']

            # progress bar
            pbar.set_postfix({
                "batch_loss": f"{loss_val:.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.6f}"
            })

            # --- CSV per-batch log (NEW) ---
            self._csv_append(self.csv_path_batches, [
                datetime.utcnow().isoformat(timespec="seconds"),
                epoch_idx,
                batch_idx,
                num_batches,
                f"{loss_val:.6f}",
                f"{avg_loss:.6f}",
                f"{current_lr:.8f}",
            ])
            # --------------------------------

        return total_loss / num_batches


    
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

            # Train for one epoch with progress bar
            epoch_loss = self.train_epoch(epoch_idx=epoch + 1, total_epochs=epochs)

            # Step scheduler
            self.scheduler.step()

            # Track losses
            self.train_losses.append(epoch_loss)

            # Best check
            is_best = epoch_loss < self.best_loss
            if is_best:
                self.best_loss = epoch_loss
                # Save best immediately so you never lose it on crashes
                self.save_checkpoint(epoch + 1, epoch_loss, is_best=True)

            # --- CSV per-epoch log (NEW) ---
            self._csv_append(self.csv_path_epochs, [
                datetime.utcnow().isoformat(timespec="seconds"),
                epoch + 1,
                f"{epoch_loss:.6f}",
                f"{self.best_loss:.6f}",
                f"{self.optimizer.param_groups[0]['lr']:.8f}",
                int(is_best),
            ])
            # --------------------------------


            print(f"Epoch {epoch + 1}/{epochs} completed:")
            print(f"  Average Loss: {epoch_loss:.4f}")
            print(f"  Best Loss: {self.best_loss:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Periodic/last checkpoint (latest + epoch_{N})
            if ((epoch + 1) % save_every == 0) or (epoch == epochs - 1):
                self.save_checkpoint(epoch + 1, epoch_loss, is_best=False)

        print(f"\nTraining completed! Best loss: {self.best_loss:.4f}")
        return self.train_losses
