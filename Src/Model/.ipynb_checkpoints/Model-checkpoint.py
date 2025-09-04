import torch
import torch.nn as nn
from Src.Model.Backbone import ModelBackbone
from Src.Model.CTC_Head import ModelHead

class CaptchaSolverModel(nn.Module):
    """
    YOLO-style model combining the backbone and head.
    """
    def __init__(self, num_classes=36, grid_height=10, grid_width=40):
        super(CaptchaSolverModel, self).__init__()
        self.backbone = ModelBackbone(in_ch=1)  # Grayscale input
        self.head = ModelHead(InputChannels=256, Classes=num_classess)

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x: Input images (batch_size, 1, H, W)

        Returns:
            predictions: (batch_size, GridHeight*GridWidth*(5 + num_classes))
        """
        features = self.backbone(x)  # Extract features using the backbone
        predictions = self.head(features)  # Generate predictions using the head
        return predictions