import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Weight Inititializeion with He-Method ) ---
def init_weights_kaiming(m):
    # use Kaiming init for conv/linear layers, set bias to zero
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # batch/group norm: start with weight=1 and bias=0
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

#  Stem 
class Stem(nn.Module):
    """
    This is a small-image friendly stem that mimics the res-net18 style stem.

    Tradition ResNet stem:
        # self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn   = nn.BatchNorm2d(out_ch)
        # self.relu = nn.ReLU(inplace=True)
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    Compared to traditional. resnet 18 this  replace 7x7 s2 + maxpool with 3×(3x3) conv stack and delay the first downsample.
    Goal: retain fineer details early in network
    """
    # def __init__(self, in_ch=3, out_ch=64):  # RGB input channels
    def __init__(self, in_ch=1, out_ch=64):    # Grayscale Input Channels
        super().__init__()

        # --- New small-image stem ---
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.act1  = nn.SiLU(inplace=True)  # see https://arxiv.org/pdf/1710.05941

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(32)
        self.act2  = nn.SiLU(inplace=True)

        self.conv3 = nn.Conv2d(32, out_ch, kernel_size=3, stride=2, padding=1, bias=False)  # first downsample here
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.act3  = nn.SiLU(inplace=True)

        # Initialize weights of conv layers via He_Init Strategy
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for b in [self.bn1, self.bn2, self.bn3]:
            nn.init.constant_(b.weight, 1.0)
            nn.init.constant_(b.bias, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        return x  # 160x640 -> 80x320 (H×W)

#  Residual Block used for resnet
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        """
        [IMPROVEMENT] allow dilation to grow RF without more downsamples
        [IMPROVEMENT] SiLU activations for smoother gradients on OCR edges
        """
        padding = dilation
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act   = nn.SiLU(inplace=True)  # was ReLU

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)


        # Init
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn1.weight, 1.0); nn.init.constant_(self.bn1.bias, 0.0)
        nn.init.constant_(self.bn2.weight, 1.0); nn.init.constant_(self.bn2.bias, 0.0)

    def forward(self, x):
        identity = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))


        out += identity
        out = self.act(out)
        return out

# ---------- ResNet-18 Backbone ----------
class ResNet18Backbone(nn.Module):
    """
    Backbone only (no GAP/FC).

    Old vanilla scales (for 640x160 W×H) were deeper OS; we keep higher spatial detail:
      stem  (s2) -> 80x320
      layer1 (s1) -> 80x320
      layer2 (s2) -> 40x160
      layer3 (s1, dil=2) -> 40x160
      layer4 (s1, dil=2) -> 40x160   (256 ch)  <-- feed to your head(s)

    Goal: better localization of small characters while keeping receptive field via dilation.
    """
    def __init__(self, in_ch=1, return_p3=True):
        super().__init__()
        self.stem = Stem(in_ch=in_ch, out_ch=64)
        self.in_ch = 64
        self.return_p3 = return_p3

        # layer1: stride=1
        self.layer1 = self._make_layer(ResidualBlock, 64,  blocks=2, stride=1, dilation=1)

        # layer2: stride=2 (downsample to ~1/4 overall)
        self.layer2 = self._make_layer(ResidualBlock, 128, blocks=2, stride=2, dilation=1)

        # layer3: keep resolution (stride=1) but expand RF with dilation
        self.layer3 = self._make_layer(ResidualBlock, 256, blocks=2, stride=1, dilation=2)

        # layer4: same resolution (stride=1) with dilation
        self.layer4 = self._make_layer(ResidualBlock, 256, blocks=2, stride=1, dilation=2)  # slimmer tail (256)
    def _make_layer(self, block, out_ch, blocks, stride, dilation):
        down = None
        in_ch = self.in_ch
    
        # Projection when spatial size or channels change
        if stride != 1 or in_ch != out_ch:
            down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
    
        layers = [block(in_ch, out_ch, stride=stride, dilation=dilation, downsample=down)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(block(out_ch, out_ch, stride=1, dilation=dilation, downsample=None))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x)         # -> 80x320
        x = self.layer1(x)      # -> 80x320
        x = self.layer2(x)     # -> 40x160
        x = self.layer3(x)     # -> 40x160
        x = self.layer4(x)     # -> 40x160
        return x
class YOLOv8Head(nn.Module):
    """
    YOLOv8-style detection head for CAPTCHA character detection
    
    Input: Feature maps from ResNet18 backbone (batch_size, 256, H, W)
    Output: Predictions (batch_size, height*width*(5 + num_classes))
           Format: [x, y, w, h, objectness, class_0, class_1, ...]
    """
    
    def __init__(self, in_channels=256, num_classes=37, height=10, width=40):
        super(YOLOv8Head, self).__init__()
        self.num_classes = num_classes
        self.height = height
        self.width = width
        
        # Output channels: bbox(4) + objectness(1) + classes(num_classes)
        self.output_channels = 5 + num_classes
        
        # Feature processing layers
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Final prediction layer
        self.pred_conv = nn.Conv2d(64, self.output_channels, kernel_size=1)
        
        # Adaptive pooling to ensure (height × width) output
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((height, width))
        # remove adaptive pool to consitent with resnet
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Feature maps from backbone (batch_size, 256, H, W)
            
        Returns:
            predictions: (batch_size, height*width*(5 + num_classes))
        """
        # Process features through conv layers
        x = F.relu(self.bn1(self.conv1(x)))  
        x = F.relu(self.bn2(self.conv2(x)))  
        x = F.relu(self.bn3(self.conv3(x)))  
        
        # Get predictions
        x = self.pred_conv(x)  # (batch_size, 5+num_classes, H, W)
        
        # Ensure fixed grid size
        # x = self.adaptive_pool(x)  # (batch_size, 5+num_classes, height, width)
        # remove adapative pool to make flexible and compatible with resnet
        
        # Reshape for loss function
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)  # (batch_size, height, width, 5+num_classes)
        x = x.view(batch_size, -1)  # (batch_size, height*width*(5+num_classes))
        
        return x

class ResNet18YOLO(nn.Module):
    """
    Full CAPTCHA solver backbone + YOLO head
    Hybrid approach: ResNet18-style backbone with dilations, YOLOv8-inspired detection head.
    
    Input:  (batch_size, 1, 160, 640)  grayscale CAPTCHA image
    Output: (batch_size, 10*10*(5 + num_classes))  detection grid
    """
    def __init__(self, num_classes=37, grid_size=10):
        super(ResNet18YOLO, self).__init__()
        # backbone
        self.backbone = ResNet18Backbone(in_ch=1, return_p3=True)
        # head
        self.head = YOLOv8Head(in_channels=256, num_classes=num_classes, S=grid_size)

    def forward(self, x):
        # Extract backbone features
        feats = self.backbone(x)              # (batch, 256, 40, 160)
        # YOLO-style detection
        preds = self.head(feats)              # (batch, 7*7*(5+num_classes))
        return preds

def decode_yolo_output(preds, num_classes=37, height=10, width=40, img_h=160, img_w=640, conf_thresh=0.5):
    """
    Decode YOLOv8Head predictions into bounding boxes and class labels.
    
    Args:
        preds: (batch_size, height*width*(5+num_classes)) flattened tensor
        num_classes: number of character classes
        height, width: grid dimensions (10x40 for your case)
        img_h, img_w: input image dimensions (160x640)
        conf_thresh: confidence threshold for objectness
        
    Returns:
        List[ List[dict] ] where each inner list is detections for one image:
        {
            "x_center": float,
            "y_center": float,
            "width": float,
            "height": float,
            "class": int,
            "confidence": float
        }
    """
    batch_size = preds.shape[0]
    cell_h = img_h / height
    cell_w = img_w / width
    
    detections = []
    
    for b in range(batch_size):
        img_preds = preds[b].view(height, width, 5 + num_classes)  # (H, W, 5+num_classes)
        
        boxes = []
        for i in range(height):
            for j in range(width):
                tx, ty, tw, th, obj_conf, *class_logits = img_preds[i, j]
                
                # Apply activations
                obj_conf = torch.sigmoid(obj_conf).item()
                class_probs = F.softmax(torch.tensor(class_logits), dim=0)
                cls_conf, cls_id = torch.max(class_probs, dim=0)
                
                # Skip low-confidence predictions
                if obj_conf * cls_conf.item() < conf_thresh:
                    continue
                
                # Decode box relative to grid cell
                x_center = (j + torch.sigmoid(tx).item()) * cell_w
                y_center = (i + torch.sigmoid(ty).item()) * cell_h
                width_box = torch.exp(tw).item() * cell_w
                height_box = torch.exp(th).item() * cell_h
                
                boxes.append({
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width_box,
                    "height": height_box,
                    "class": int(cls_id.item()),
                    "confidence": float(obj_conf * cls_conf.item())
                })
        
        detections.append(boxes)
    
    return detections

# --- Example usage ---
if __name__ == "__main__":
    model = ResNet18YOLO(num_classes=37, grid_size=7)
    dummy = torch.randn(2, 1, 160, 640)  # batch=2, grayscale CAPTCHA
    out = model(dummy)
    print("Output shape:", out.shape)   # Expected: (2, 7*7*(5+37))


