import torch
import torch.nn as nn
import torch.nn.functional as F
from Src.Utils.WeightInitializer import WeightInitializer

class ModelHead(nn.Module):
    """
    CTC-style head for CAPTCHA sequence recognition.
    
    Instead of predicting bounding boxes per grid cell, this head
    collapses the height dimension and outputs a sequence of logits
    along the width axis, suitable for CTC decoding.
    """
    def __init__(self, InputChannels=256, Classes=36):
        super(ModelHead, self).__init__()
        # Add +1 for CTC blank symbol
        self.Classes = Classes
        self.OutputClasses = Classes + 1  

        # Reduce channel dimensions for efficiency
        self.conv1 = nn.Conv2d(InputChannels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Final linear layer for classification at each time step
        self.fc = nn.Linear(64, self.OutputClasses)

        # Initialize weights
        self.apply(WeightInitializer)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Feature maps from backbone (batch_size, in_channels, H, W)

        Returns:
            logits: (T, batch_size, OutputClasses)
                    where T = sequence length (usually width of feature map)
        """
        # Process features with conv layers
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 128, H, W)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 64, H, W)

        # Collapse height → average pooling across feature height
        x = torch.mean(x, dim=2)              # (B, 64, W)

        # Reshape to sequence format
        x = x.permute(0, 2, 1)                # (B, W, 64)

        # Apply linear layer per time step
        x = self.fc(x)                        # (B, W, OutputClasses)

        # CTC expects (T, N, C)
        x = x.permute(1, 0, 2)                # (W, B, OutputClasses)
        return x
