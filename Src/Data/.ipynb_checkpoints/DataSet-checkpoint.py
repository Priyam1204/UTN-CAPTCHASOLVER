import os, json, random, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset

# torchvision v2
from torchvision import transforms, tv_tensors
from torchvision.transforms import v2 as T


# ---------- Noise Augmentations (only image, boxes unchanged) ----------
def add_noise_transforms(image):
    augments = T.Compose([
        T.RandomApply([T.RandomAutocontrast()], p=0.3),
        T.RandomApply([T.RandomEqualize()], p=0.15),
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.25),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.25),
    ])
    return augments(image)

def add_salt_pepper(image, prob=0.5, salt_prob=0.005, pepper_prob=0.005):
    if random.random() > prob:
        return image
    arr = np.array(image)
    H, W = arr.shape
    # salt
    num_salt = int(np.ceil(salt_prob * H * W))
    coords = (np.random.randint(0, H, num_salt), np.random.randint(0, W, num_salt))
    arr[coords] = 255
    # pepper
    num_pepper = int(np.ceil(pepper_prob * H * W))
    coords = (np.random.randint(0, H, num_pepper), np.random.randint(0, W, num_pepper))
    arr[coords] = 0
    return Image.fromarray(arr)

'''old augmentation, lines are too small
def add_lines(image, prob=0.3, min_ratio=0.5, max_ratio=1.2, width_range=(1, 3), max_lines=3):
    if random.random() > prob:
        return image
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(random.randint(1, max_lines)):
        length = random.uniform(min_ratio, max_ratio) * math.hypot(width, height)
        x0, y0 = random.uniform(0, width), random.uniform(0, height)
        angle = random.uniform(0, 2 * math.pi)
        x1, y1 = x0 + length * math.cos(angle), y0 + length * math.sin(angle)
        line_width = random.randint(width_range[0], width_range[1])
        draw.line([(x0, y0), (x1, y1)], fill=0, width=line_width)
    return image
'''

def add_lines(image, prob=0.3, min_ratio=0.3, max_ratio=1.0, width_range=(1, 4), max_lines=4):
    if random.random() > prob:
        return image
    draw = ImageDraw.Draw(image)
    width, height = image.size
    diag = math.hypot(width, height)
    for _ in range(random.randint(1, max_lines)):
        if random.random() < 0.5:
            # 50% chance to draw long line through center
            length = random.uniform(0.8, 1.2) * diag
            x0, y0 = width / 2, height / 2
        else:
            # draw shorter shorter clutter line anywhere
            length = random.uniform(min_ratio, max_ratio) * diag * 0.5
            x0, y0 = random.uniform(0, width), random.uniform(0, height)
        angle = random.choice([random.uniform(-0.5, 0.5), random.uniform(2.5, 3.5)])  # ~45° or ~135°
        x1, y1 = x0 + length * math.cos(angle), y0 + length * math.sin(angle)
        line_width = random.randint(width_range[0], width_range[1])
        draw.line([(x0, y0), (x1, y1)], fill=0, width=line_width)
    return image

def add_circles(image, prob=0.3, radius_range=(3, 10), max_circles=5):
    """
    Adds small black circular distractors to the image.
    """
    if random.random() > prob:
        return image
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(random.randint(1, max_circles)):
        r = random.randint(radius_range[0], radius_range[1])
        x0 = random.uniform(0, width - r)
        y0 = random.uniform(0, height - r)
        x1, y1 = x0 + 2 * r, y0 + 2 * r
        draw.ellipse([(x0, y0), (x1, y1)], fill=0)
    return image

def add_symbol_distractors(image,
                           prob=0.4,
                           symbols="*#?✓",
                           max_symbols=5,
                           size_range=(14, 28),
                           angle_range=(-35, 35),
                           alpha_range=(200, 255),
                           fonts=None):
    """
    Adds small symbol distractors (e.g., '*#?✓') as grayscale overlays.
    If `fonts` is a list of TTF paths, picks randomly; otherwise uses PIL default.
    """
    if random.random() > prob:
        return image

    W, H = image.size
    base = image.convert("L")

    for _ in range(random.randint(1, max_symbols)):
        ch = random.choice(symbols)

        # font selection
        if fonts and len(fonts) > 0:
            try:
                fp = random.choice(fonts)
                fsz = random.randint(*size_range)
                font = ImageFont.truetype(fp, fsz)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        # render glyph to its own small mask
        # get bbox for sizing
        bbox = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), ch, font=font)
        gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        glyph = Image.new("L", (max(1, gw), max(1, gh)), 0)
        gdraw = ImageDraw.Draw(glyph)

        fill_val = 0 if random.random() < 0.5 else 255                 # black or white
        alpha = random.randint(*alpha_range)                            # opacity
        gdraw.text((0, 0), ch, font=font, fill=alpha)                   # alpha in mask

        # rotate glyph and paste
        ang = random.uniform(*angle_range)
        glyph = glyph.rotate(ang, expand=True)

        # position: mildly biased toward center
        cx = random.uniform(W * 0.15, W * 0.85)
        cy = random.uniform(H * 0.25, H * 0.75)
        x0 = int(cx - glyph.width / 2)
        y0 = int(cy - glyph.height / 2)

        # paste uses 'glyph' as mask; fill_val defines distractor intensity
        patch = Image.new("L", glyph.size, color=fill_val)
        base.paste(patch, (x0, y0), mask=glyph)

    return base


def apply_noise_policy(image,
                       clean_prob=0.4,
                       photometric_prob=0.2,
                       clutter_prob=0.25,
                       occlusion_prob=0.15):
    """
    One-shot augmentation policy:
      - clean_prob:        return image as-is
      - photometric_prob:  photometric jitter only
      - clutter_prob:      short lines OR small circles (+ optional light salt/pepper)
      - occlusion_prob:    long diagonal line(s) OR bold circles (+ optional light salt/pepper)
    Photometric (if used) is applied BEFORE overlays; overlays are applied last.
    """
    print("noise")
    r = random.random()
    if r < clean_prob:
        return image

    # pick mode
    r2 = random.random()
    thresholds = (photometric_prob, photometric_prob + clutter_prob, photometric_prob + clutter_prob + occlusion_prob)

    # base (can be added to any non-clean mode)
    def maybe_salt(img, p=0.25):
        return add_salt_pepper(img, prob=p, salt_prob=0.003, pepper_prob=0.003)

    # ---- photometric-only ----
    if r2 < thresholds[0]:
        img = add_noise_transforms(image)
        return img

    # ---- clutter: short lines OR small circles ----
    if r2 < thresholds[1]:
        img = add_noise_transforms(image) if random.random() < 0.5 else image
        if random.random() < 0.5:
            # shorter lines scattered anywhere
            img = add_lines(img, prob=1.0, min_ratio=0.2, max_ratio=0.6, width_range=(1, 3), max_lines=3)
        else:
            # small circles
            img = add_circles(img, prob=1.0, radius_range=(3, 10), max_circles=5)
        img = maybe_salt(img, p=0.20)
        return img

    # ---- occlusion: long diagonal line(s) OR bold circles ----
    img = add_noise_transforms(image) if random.random() < 0.5 else image
    if random.random() < 0.6:
        # long lines through center: bias to diagonals by calling your add_lines with longer ratios
        # (angle bias handled inside your function as you set earlier)
        img = add_lines(img, prob=1.0, min_ratio=0.8, max_ratio=1.2, width_range=(2, 4), max_lines=2)
    else:
        # fewer but larger circles
        img = add_circles(img, prob=1.0, radius_range=(8, 18), max_circles=3)
    img = maybe_salt(img, p=0.15)
    return img

# ---------- Geometric Augmentations (image + boxes together) ----------
def apply_geo_transforms(image, target):
    H, W = image.height, image.width
    geom = T.Compose([
        T.ToImage(),
        T.RandomAffine(
            degrees=7, translate=(0.02, 0.02), scale=(0.95, 1.05), shear=5,
            interpolation=T.InterpolationMode.BILINEAR, fill=0
        ),
        T.SanitizeBoundingBoxes(min_size=1),
        T.ToDtype(torch.float32, scale=True),
    ])
    boxes = tv_tensors.BoundingBoxes(target["boxes"], format="XYXY", canvas_size=(H, W))
    tgt = {"boxes": boxes, "labels": target["labels"]}
    img_t, tgt = geom(image, tgt)
    return img_t, tgt


class CaptchaDataset(Dataset):
    """
    A PyTorch Dataset for loading CAPTCHA images and its metadata from labels.json
    """
    
    def __init__(self, data_dir, transform=None, use_geo_aug=False):
    
        #Set directories
        self.DataDirectory = data_dir
        self.ImagesDirectory = os.path.join(self.DataDirectory, 'images')
        self.Transform = transform
        self.use_geo_aug = use_geo_aug   # <--- NEW flag
        self.ImageList = sorted([f for f in os.listdir(self.ImagesDirectory) if f.endswith('.png')])

        #Load labels
        LabelsFile = os.path.join(self.DataDirectory, 'labels.json')
        self.LabelsDict = {}
        if os.path.exists(LabelsFile):
            with open(LabelsFile, 'r') as f:
                LabelsJS = json.load(f)
                self.LabelsDict = {item['image_id']: item for item in LabelsJS}

    def __len__(self):

        return len(self.ImageList)

    def __getitem__(self, idx):

        ImageName = self.ImageList[idx]
        ImagePath = os.path.join(self.ImagesDirectory, ImageName)

        #Load image as grayscale
        image = Image.open(ImagePath).convert('L')
        ImageSize = image.size  

        #Get metadata
        ImageID = os.path.splitext(ImageName)[0]
        LabelsInfo = self.LabelsDict.get(ImageID, {})
        CaptchaString = LabelsInfo.get('captcha_string', '')
        Annotations = LabelsInfo.get('annotations', [])

        #Process bounding boxes
        BoundingBoxes = [ann.get('bbox', []) for ann in Annotations]
        OrientedBoundingBoxes = [ann.get('oriented_bbox', []) for ann in Annotations]
        CategoryIDs = [ann.get('category_id', -1) for ann in Annotations]

        #Convert to tensors
        BoundingBoxes = torch.tensor(BoundingBoxes, dtype=torch.float32) if BoundingBoxes else torch.empty((0, 4))
        OrientedBoundingBoxes = torch.tensor(OrientedBoundingBoxes, dtype=torch.float32) if OrientedBoundingBoxes else torch.empty((0, 8))
        CategoryIDs = torch.tensor(CategoryIDs, dtype=torch.long) if CategoryIDs else torch.empty((0,), dtype=torch.long)

        # ---- Apply geometric augmentation if enabled ----
        if self.use_geo_aug and BoundingBoxes.numel() > 0:
            image, tgt = apply_geo_transforms(image, {"boxes": BoundingBoxes, "labels": CategoryIDs})
            BoundingBoxes = tgt["boxes"].as_subclass(torch.Tensor)
            CategoryIDs = tgt["labels"]
            from torchvision.transforms import functional as F
            image = F.to_pil_image(image)

        # ---- Apply noise augmentations (always) ----
        #image = apply_noise_policy(image)

        # Apply transforms
        if self.Transform:
            image = self.Transform(image)

        return {
            'Image': image,
            'ImageID': ImageID,
            'CaptchaString': CaptchaString,
            'BoundingBoxes': BoundingBoxes,
            'OrientedBoundingBoxes': OrientedBoundingBoxes,
            'CategoryIDs': CategoryIDs,
            'NumberofObjects': len(Annotations),
            'ImageSize': ImageSize
        }