import os, json, random, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset
from .NoisePolicy import Apply_Geo_Augmentations, Apply_Noise_Policy

class CaptchaDataset(Dataset):
    """
    A PyTorch Dataset for loading CAPTCHA images and its metadata from labels.json
    """
    
    def __init__(self, data_dir, transform=None, use_geo_aug=False):
    
        self.DataDirectory = data_dir
        self.ImagesDirectory = os.path.join(self.DataDirectory, 'images')
        self.Transform = transform
        self.use_geo_aug = use_geo_aug   # <--- NEW flag
        self.ImageList = sorted([f for f in os.listdir(self.ImagesDirectory) if f.endswith('.png')])

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

        ImageID = os.path.splitext(ImageName)[0]
        LabelsInfo = self.LabelsDict.get(ImageID, {})
        CaptchaString = LabelsInfo.get('captcha_string', '')
        Annotations = LabelsInfo.get('annotations', [])

        BoundingBoxes = [ann.get('bbox', []) for ann in Annotations]
        OrientedBoundingBoxes = [ann.get('oriented_bbox', []) for ann in Annotations]
        CategoryIDs = [ann.get('category_id', -1) for ann in Annotations]

        BoundingBoxes = torch.tensor(BoundingBoxes, dtype=torch.float32) if BoundingBoxes else torch.empty((0, 4))
        OrientedBoundingBoxes = torch.tensor(OrientedBoundingBoxes, dtype=torch.float32) if OrientedBoundingBoxes else torch.empty((0, 8))
        CategoryIDs = torch.tensor(CategoryIDs, dtype=torch.long) if CategoryIDs else torch.empty((0,), dtype=torch.long)

        if self.use_geo_aug and BoundingBoxes.numel() > 0:
            image, tgt = Apply_Geo_Augmentations(image, {"boxes": BoundingBoxes, "labels": CategoryIDs})
            BoundingBoxes = tgt["boxes"].as_subclass(torch.Tensor)
            CategoryIDs = tgt["labels"]
            from torchvision.transforms import functional as F
            image = F.to_pil_image(image)

        # Noise is commented out for inference
        # image = Apply_Noise_Policy(image)

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