import os, json, random, math
import numpy as np
from PIL import Image, ImageDraw
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

def add_lines(image, prob=0.3, min_ratio=0.1, max_ratio=0.6, width_range=(1, 3), max_lines=3):
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
        image = add_noise_transforms(image)
        image = add_lines(image)
        image = add_salt_pepper(image)

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