import torch
import torch.nn as nn
import torch.nn.functional as F
from Src.Utils.WeightInitializer import WeightInitializer



# --- Stem ---
class Stem(nn.Module):
    """
    This is a small-image friendly stem that mimics the ResNet-18 style stem.

    Compared to traditional ResNet-18, this replaces 7x7 s2 + maxpool with 3×(3x3) conv stack
    and delays the first downsample. Goal: retain finer details early in the network.
    """
    def __init__(self, in_ch=1, out_ch=64):  # Grayscale Input Channels
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.SiLU(inplace=True)

        self.conv3 = nn.Conv2d(32, out_ch, kernel_size=3, stride=2, padding=1, bias=False)  # First downsample here
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.act3 = nn.SiLU(inplace=True)

        # Apply weight initialization
        self.apply(WeightInitializer)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        return x  # Output shape: 160x640 -> 80x320 (H×W)

# --- Residual Block ---
class ResidualBlock(nn.Module):
    """
    Basic residual block with optional dilation.
    """
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        # Apply weight initialization
        self.apply(WeightInitializer)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.act(out)
        return out

# --- ResNet18 Backbone ---
class ModelBackbone(nn.Module):
    """
    ResNet-18 backbone with dilated convolutions for CAPTCHA detection.

    Old vanilla scales (for 640x160 W×H) were deeper OS; we keep higher spatial detail:
      stem  (s2) -> 80x320
      layer1 (s1) -> 80x320
      layer2 (s2) -> 40x160
      layer3 (s1, dil=2) -> 40x160
      layer4 (s1, dil=2) -> 40x160   (256 ch)  <-- feed to your head(s)

    Goal: better localization of small characters while keeping receptive field via dilation.
    """
    def __init__(self, in_ch=1):
        super().__init__()
        self.stem = Stem(in_ch=in_ch, out_ch=64)
        self.in_ch = 64

        # layer1: stride=1
        self.layer1 = self._make_layer(ResidualBlock, 64,  blocks=2, stride=1, dilation=1)

        # layer2: stride=2 (downsample to ~1/4 overall)
        self.layer2 = self._make_layer(ResidualBlock, 128, blocks=2, stride=2, dilation=1)

        # layer3: keep resolution (stride=1) but expand RF with dilation
        self.layer3 = self._make_layer(ResidualBlock, 256, blocks=2, stride=1, dilation=2)

        # layer4: same resolution (stride=1) with dilation
        self.layer4 = self._make_layer(ResidualBlock, 256, blocks=2, stride=1, dilation=2)

        # Apply weight initialization
        self.apply(WeightInitializer)

    def _make_layer(self, block, out_ch, blocks, stride, dilation):
        layers = []
        layers.append(block(self.in_ch, out_ch, stride=stride, dilation=dilation))
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(block(out_ch, out_ch, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)         # -> 80x320
        x = self.layer1(x)       # -> 80x320
        x = self.layer2(x)       # -> 40x160
        x = self.layer3(x)       # -> 40x160
        x = self.layer4(x)       # -> 40x160
        return x
    


''' TRADITIONAL RES-NET
# ---------- BasicBlock (ResNet-18 style) ----------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.apply(WeightInitializer)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ---------- Traditional ResNet-18 Stem ----------
class StemResNet(nn.Module):
    """7x7 s2 + BN + ReLU + 3x3 MaxPool s2 (classic ResNet-18)."""
    def __init__(self, in_ch=1, out_ch=64):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.apply(WeightInitializer)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))  # /2
        x = self.pool(x)                      # /4
        return x

# ---------- Traditional ResNet-18 Backbone ----------
class ModelBackbone(nn.Module):
    """
    Classic ResNet-18 layout:
      Stem: 7x7 s2 + 3x3 maxpool s2  -> /4
      Layer1: 64,  stride 1          -> /4
      Layer2: 128, stride 2          -> /8
      Layer3: 256, stride 2          -> /16
      Layer4: 512, stride 2          -> /32   (OUTPUT CHANNELS = 512)
    For 1×160×640 input -> spatial ~ 5×20 before your head's adaptive pooling.
    """
    def __init__(self, in_ch=1):
        super().__init__()
        self.stem = StemResNet(in_ch=in_ch, out_ch=64)
        self.inplanes = 64

        self.layer1 = self._make_layer(64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.apply(WeightInitializer)

    def _make_layer(self, out_ch, blocks, stride):
        layers = [BasicBlock(self.inplanes, out_ch, stride=stride)]
        self.inplanes = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)  # /4
        x = self.layer2(x)  # /8
        x = self.layer3(x)  # /16
        x = self.layer4(x)  # /32  -> (B, 512, ~5, ~20) for 160×640 input
        return x

'''
