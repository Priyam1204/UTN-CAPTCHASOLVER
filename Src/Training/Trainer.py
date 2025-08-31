import torch
import os
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from Src.Model.Model import CaptchaSolverModel
from Src.Model.Loss import ModelLoss
from Src.Data.DataLoader import CaptchaDataLoader
from Src.Utils.TargetPreparer import TargetPreparer

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
        
        # Initialize data loader
        self.train_loader = CaptchaDataLoader(data_dir, batch_size=32, shuffle=True)
        
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
    
    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['Image'].to(self.device)  # Input images
            
            # Convert raw annotations to YOLO target format
            targets = self.target_preparer(batch).to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            loss_dict = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Gradient clipping 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss_dict['total_loss'].item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{num_batches}, Loss: {loss_dict['total_loss'].item():.4f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        return total_loss / num_batches
    
    def train(self, epochs, resume_from=None, save_every=5):
        """
        Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
            save_every: Save checkpoint every N epochs
        """
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        print(f"Starting training from epoch {start_epoch + 1} to {epochs}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(start_epoch, epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*50}")
            
            # Train for one epoch
            epoch_loss = self.train_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Track losses
            self.train_losses.append(epoch_loss)
            
            # Check if this is the best model
            is_best = epoch_loss < self.best_loss
            if is_best:
                self.best_loss = epoch_loss
            
            print(f"Epoch {epoch + 1}/{epochs} completed:")
            print(f"  Average Loss: {epoch_loss:.4f}")
            print(f"  Best Loss: {self.best_loss:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or is_best or epoch == epochs - 1:
                self.save_checkpoint(epoch + 1, epoch_loss, is_best)
        
        print(f"\nTraining completed! Best loss: {self.best_loss:.4f}")
        return self.train_losses