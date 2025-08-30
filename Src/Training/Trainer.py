import torch
from torch.optim import Adam
from Src.Model.Model import CaptchaSolverModel
from Src.Model.Loss import ModelLoss
from Src.Data.DataLoader import CaptchaDataLoader
from Src.Utils.TargetPreparer import TargetPreparer

class Trainer:
    def __init__(self, data_dir, num_classes=36, grid_height=20, grid_width=80, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = CaptchaSolverModel(num_classes=num_classes, grid_height=grid_height, grid_width=grid_width).to(self.device)
        self.criterion = ModelLoss(num_classes=num_classes, GridHeight=grid_height, GridWidth=grid_width).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.train_loader = CaptchaDataLoader(data_dir, batch_size=32, shuffle=True)
        
        # Initialize target preparer
        self.target_preparer = TargetPreparer(
            GridHeight=grid_height, 
            GridWidth=grid_width, 
            num_classes=num_classes,
            img_width=640, 
            img_height=160
        )
    
    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        
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
            self.optimizer.step()
            
            total_loss += loss_dict['total_loss'].item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss_dict['total_loss'].item():.4f}")
        
        return total_loss / len(self.train_loader)
    
    def train(self, epochs):
        """
        Train the model for multiple epochs.
        """
        for epoch in range(epochs):
            epoch_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")