import torch
import torch.nn as nn
import torch.nn.functional as F
from Src.Utils.WeightInitializer import WeightInitializer

class ModelHead(nn.Module):
    """
    YOLOv8-style detection head for CAPTCHA character detection with rectangular grids.
    """
    def __init__(self, InputChannels=256, Classes=36, GridHeight=40, GridWidth=160):
        super(ModelHead, self).__init__()
        self.Classes = Classes
        self.GridHeight = GridHeight  # Height of the grid (e.g., 40)
        self.GridWidth = GridWidth    # Width of the grid (e.g., 160)
        self.OutputChannels = 5 + Classes  # bbox(4) + objectness(1) + number of classes(Classes)

        # Feature processing layers
        self.conv1 = nn.Conv2d(InputChannels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Final prediction layer
        self.pred_conv = nn.Conv2d(64, self.OutputChannels, kernel_size=1)

        # Adaptive pooling to ensure GridHeight × GridWidth output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((GridHeight, GridWidth))

        # Initialize weights
        self.apply(WeightInitializer)

    

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Feature maps from backbone (batch_size, in_channels, H, W)

        Returns:
            predictions: (batch_size, GridHeight*GridWidth*(5 + Classes))
        """
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 128, 40, 10)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, 64, 40, 10)
        x = self.pred_conv(x)                # (batch_size, 5+num_classes, 40, 10)
        x = self.adaptive_pool(x)            # Ensure (grid_height, GridWidth) output

        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)  # (batch_size, grid_height, GridWidth, 5+num_classes)
        x = x.reshape(batch_size, -1)  # Flatten to (batch_size, grid_height*GridWidth*(5+num_classes))
        return x


