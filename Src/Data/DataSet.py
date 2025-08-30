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


# ---------- Dataset ----------
class CaptchaDataset(Dataset):
    def __init__(self, data_dir, transform=None, use_geo_aug=False):
        self.DataDirectory = data_dir
        self.ImagesDirectory = os.path.join(self.DataDirectory, "images")
        self.Transform = transform
        self.use_geo_aug = use_geo_aug

        self.ImageList = sorted([f for f in os.listdir(self.ImagesDirectory) if f.endswith(".png")])
        LabelsFile = os.path.join(self.DataDirectory, "labels.json")
        self.LabelsDict = {}
        if os.path.exists(LabelsFile):
            with open(LabelsFile, "r") as f:
                labels = json.load(f)
                self.LabelsDict = {item["image_id"]: item for item in labels}

    def __len__(self):
        return len(self.ImageList)

    def __getitem__(self, idx):
        img_name = self.ImageList[idx]
        img_path = os.path.join(self.ImagesDirectory, img_name)

        image = Image.open(img_path).convert("L")
        image_id = os.path.splitext(img_name)[0]

        labels_info = self.LabelsDict.get(image_id, {})
        captcha_string = labels_info.get("captcha_string", "")
        annotations = labels_info.get("annotations", [])

        bboxes = [ann.get("bbox", []) for ann in annotations]
        cats = [ann.get("category_id", -1) for ann in annotations]

        bboxes = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.empty((0, 4))
        cats = torch.tensor(cats, dtype=torch.long) if cats else torch.empty((0,), dtype=torch.long)

        if self.use_geo_aug and len(bboxes) > 0:
            image, tgt = apply_geo_transforms(image, {"boxes": bboxes, "labels": cats})
            bboxes = tgt["boxes"].as_subclass(torch.Tensor)
            cats = tgt["labels"]
            from torchvision.transforms import functional as F
            image = F.to_pil_image(image)   # <-- ensures PIL for the next steps

        # noise augments
        image = add_noise_transforms(image)
        image = add_lines(image)
        image = add_salt_pepper(image)

        W, H = image.size  # PIL gives (width, height)

        # final transform (ToTensor, Normalize)
        if self.Transform:
            image = self.Transform(image)

        if bboxes.numel() > 0:
            x1, y1, x2, y2 = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = (x2 - x1).clamp(min=0)
            h  = (y2 - y1).clamp(min=0)
            oriented = torch.stack([cx, cy, w, h, torch.zeros_like(cx)], dim=1)  # angle=0
        else:
            oriented = torch.empty((0, 5), dtype=torch.float32)

        return {
            "Image": image,
            "ImageID": image_id,
            "CaptchaString": captcha_string,
            "BoundingBoxes": bboxes,              # (N,4) XYXY
            "OrientedBoundingBoxes": oriented,    # (N,5) (cx,cy,w,h,theta)
            "CategoryIDs": cats,                  # (N,)
            "NumberofObjects": len(annotations),
            "ImageSize": (H, W),                  # store (height, width) consistently
        }
