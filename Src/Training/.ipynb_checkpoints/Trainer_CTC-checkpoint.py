import torch
import os
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from Src.Model.CTC_Model import CaptchaSolverModel
from Src.Model.CTC_Loss import ModelLoss
from Src.Data.DataLoader import CaptchaDataLoader
#from Src.Utils.TargetPreparer_CTC import TargetPreparerCTC

class Trainer:
    def __init__(self, data_dir, num_classes=36, grid_height=10, grid_width=40, 
                 device='cuda', optimizer_type='adam', learning_rate=0.00003, 
                 weight_decay=1e-4, save_dir='./checkpoints'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        '''
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
        '''
        self.model = CaptchaSolverModel(num_classes=num_classes).to(self.device)

        # CTC loss wrapper (uses blank=Classes as last index)
        self.criterion = ModelLoss(Classes=num_classes).to(self.device)
        self.optimizer = self._get_optimizer(optimizer_type, learning_rate, weight_decay)
        self.scheduler = self._get_scheduler()

        # No need for YOLO-style target preparer anymore
        self.target_preparer = None

        self.train_loader = CaptchaDataLoader(data_dir, batch_size=8, shuffle=True)


        
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
        return StepLR(self.optimizer, step_size=3, gamma=0.8)
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
    
    def analyze_objectness_predictions(self, predictions):
        """
        Analyze objectness predictions to track fix effectiveness
        """
        # ✅ NO RESHAPING NEEDED: predictions is already [batch, height, width, channels]
        # Extract objectness predictions (4th channel)
        obj_preds = predictions[:, :, :, 4]  # Shape: [batch, height, width]
        
        # Apply sigmoid to get probabilities
        obj_probs = torch.sigmoid(obj_preds)
        
        # Calculate statistics
        stats = {
            'min_prob': obj_probs.min().item(),
            'max_prob': obj_probs.max().item(),
            'mean_prob': obj_probs.mean().item(),
            'std_prob': obj_probs.std().item(),
        }
        
        # Count confident predictions at different thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for thresh in thresholds:
            count = (obj_probs > thresh).sum().item()
            total = obj_probs.numel()
            stats[f'above_{thresh}'] = count
            stats[f'pct_above_{thresh}'] = (count / total) * 100
        
        return stats

    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        total_obj_loss = 0
        total_bbox_loss = 0 
        total_class_loss = 0
        num_positive_samples = 0
        # ✅ ADD: Track sampling statistics
        total_negative_samples = 0
        total_samples_used = 0
        
        # ✅ ADD: Analyze first batch objectness every 10 batches
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['Image'].to(self.device)
            # Convert captcha strings → label indices + lengths
            targets, target_lengths = batch['Targets'].to(self.device), batch['TargetLengths'].to(self.device)

            predictions = self.model(images)  # logits (T, N, C)

            # Input lengths: all same = sequence length T
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=predictions.size(0),
                dtype=torch.long,
                device=self.device
            )

            loss = self.criterion(predictions, targets, input_lengths, target_lengths)
            print(loss)
            
            # Track detailed losses
            
            total_loss += loss.item()
            '''
            total_obj_loss += loss_dict.get('obj_loss', 0)
            total_bbox_loss += loss_dict.get('bbox_loss', 0)
            total_class_loss += loss_dict.get('class_loss', 0)
            num_positive_samples += loss_dict.get('num_pos', 0)
            # ✅ ADD: Track sampling statistics
            total_negative_samples += loss_dict.get('num_neg', 0)
            total_samples_used += loss_dict.get('total_samples_used', 0)
            '''
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            '''
            # ✅ ENHANCED: Log progress with more details every 20 batches
            if batch_idx % 20 == 0:
                pos_samples = loss_dict.get('num_pos', 0)
                neg_samples = loss_dict.get('num_neg', 0)
                total_used = loss_dict.get('total_samples_used', 0)
                
                # Calculate sampling ratio
                if pos_samples > 0:
                    sampling_ratio = neg_samples / pos_samples
                else:
                    sampling_ratio = 0
                
                print(f"Batch {batch_idx}/{len(self.train_loader)}: "
                      f"Loss={loss.item():.4f}, "
                      f"Obj={loss_dict.get('obj_loss', 0):.4f}, "
                      f"BBox={loss_dict.get('bbox_loss', 0):.4f}, "
                      f"Class={loss_dict.get('class_loss', 0):.4f}")
                print(f"  ├─ Sampling: Pos={pos_samples}, Neg={neg_samples}, "
                      f"Ratio=1:{sampling_ratio:.1f}, Used={total_used}")
        
        avg_loss = total_loss / len(self.train_loader)
        avg_pos = num_positive_samples / len(self.train_loader)
        # ✅ ADD: Calculate average sampling statistics
        avg_neg = total_negative_samples / len(self.train_loader)
        avg_used = total_samples_used / len(self.train_loader)
        
        print(f"\n📊 Epoch Summary:")
        print(f"  ├─ Average Loss: {avg_loss:.4f}")
        print(f"  ├─ Average Positive Samples: {avg_pos:.2f}")
        print(f"  ├─ Average Negative Samples: {avg_neg:.2f}")
        print(f"  ├─ Average Total Used: {avg_used:.2f}")
        if avg_pos > 0:
            print(f"  └─ Average Pos:Neg Ratio: 1:{avg_neg/avg_pos:.1f}")
        '''
        avg_loss = total_loss / len(self.train_loader)
        print(f"\n📊 Epoch Summary: Average Loss = {avg_loss:.4f}")
        return avg_loss
    
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
            print('hi')
            start_epoch = self.load_checkpoint(resume_from)
            print('hi')
        
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