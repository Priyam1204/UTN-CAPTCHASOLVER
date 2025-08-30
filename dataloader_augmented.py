import os
import json
import math
import random
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw

# torchvision
from torchvision import transforms, tv_tensors
from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T  # v2 augmentations




''' BELOW IS ADDED BY NOAS'''
def add_noise_transforms(image):
    # assumes image in pillow form, returns tensor
    augments = T.Compose([
        T.RandomApply([T.RandomAutocontrast()], p=0.3),
        T.RandomApply([T.RandomEqualize()], p=0.15),
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.25),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.25),
    ])
    return augments(image)   # apply transform pipeline to image


def apply_geo_transforms(image, target):
    """
    target['boxes'] is a tensor in (cx, cy, w, h).
    target['labels'] is a tensor of class ids.
    """
    H, W = image.height, image.width
    # Build one pipeline that updates both image and boxes
    geom = T.Compose([
        T.ToImage(),  # PIL -> tensor (C,H,W)
        T.RandomAffine(
            degrees=7, translate=(0.02, 0.02), scale=(0.95, 1.05), shear=5,
            interpolation=T.InterpolationMode.BILINEAR, fill=0
        ),
        T.SanitizeBoundingBoxes(min_size=1),  # drop/clip degenerate boxes
        T.ToDtype(torch.float32, scale=True),
    ])
    # Wrap boxes so v2 transforms know how to update them

    # Function Below can adjust the bounding boxes based on the same transforms supposedely
    boxes = tv_tensors.BoundingBoxes(
        target["boxes"], format="CXCYWH", canvas_size=(H, W)
    )
    tgt = {"boxes": boxes, "labels": target.get("labels", None)}

    img_t, tgt = geom(image, tgt)

    # (optional) keep your preferred box format
    tgt["boxes"] = tgt["boxes"].to_format("CXCYWH")

    return img_t, tgt


def add_salt_pepper(image, prob=0.5, salt_prob=0.005, pepper_prob=0.005):
    """Add salt & pepper to a PIL grayscale image."""
    if random.random() > prob:
        return image

    arr = np.array(image)  # HxW, uint8
    H, W = arr.shape

    # add salt (white) 
    num_salt = int(np.ceil(salt_prob * H * W))
    coords = (np.random.randint(0, H, num_salt), np.random.randint(0, W, num_salt))
    arr[coords] = 255

    # add pepper (black)
    num_pepper = int(np.ceil(pepper_prob * H * W))
    coords = (np.random.randint(0, H, num_pepper), np.random.randint(0, W, num_pepper))
    arr[coords] = 0

    return Image.fromarray(arr)

def add_lines(image,prob=.3,min_ratio=0.1, max_ratio=0.6, width_range=(1, 3), max_lines=3):
    '''
    adds 1-3 lines to 30% of images, uses ratio to decide length. 
    '''
    if random.random() > prob:
        return image # only adds lines to 30% of images
    drawing = ImageDraw.draw(image)
    width,height = image.size
    num_lines = random.randint(1,max_lines)
    for n in range(num_lines):
        length = random.uniform(min_ratio, max_ratio) * math.hypot(width, height) # chooses length of line randomly
        x0,y0 = random.uniform(0,width), random.uniform(0,height)
        rot_angle=  random.uniform(0,2*math.pi)
        x1, y1 = x0 + length * math.cos(rot_angle), y0 + length * math.sin(rot_angle)
        line_width = random.randint(width_range[0],width_range[1])
        drawing.line([(x0, y0), (x1, y1)], fill=(0, 0, 0), width=line_width)
    return image
''' ABOVE IS ADDED BY NOAS'''



class CaptchaDatasetWithBBoxes(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Direct path to the specific data folder (train, val, or test)
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(self.data_dir, 'images')
        self.transform = transform
        self.image_list = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])
        
        # Load labels if available
        self.labels_dict = {}
        labels_file = os.path.join(self.data_dir, 'labels.json')
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                labels = json.load(f)
                self.labels_dict = {item['image_id']: item for item in labels}
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.image_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load image as grayscale
        image = Image.open(img_path).convert('L')
        original_size = image.size  # (width, height)
        
        # Get image_id without extension
        image_id = os.path.splitext(img_name)[0]
        
        # Get labels if available
        label_info = self.labels_dict.get(image_id, {})
        
        # Extract captcha string and annotations
        captcha_string = label_info.get('captcha_string', '')
        annotations = label_info.get('annotations', [])
        
        # Process bounding boxes
        bboxes = []
        oriented_bboxes = []
        category_ids = []
        
        for annotation in annotations:
            # Regular bounding box [x1, y1, x2, y2]
            bbox = annotation.get('bbox', [])
            if bbox:
                bboxes.append(bbox)
            
            # Oriented bounding box [x1,y1,x2,y2,x3,y3,x4,y4]
            oriented_bbox = annotation.get('oriented_bbox', [])
            if oriented_bbox:
                oriented_bboxes.append(oriented_bbox)
            
            # Category ID (character class)
            category_id = annotation.get('category_id', -1)
            category_ids.append(category_id)
        
        # Convert to tensors
        bboxes = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.empty((0, 4))
        oriented_bboxes = torch.tensor(oriented_bboxes, dtype=torch.float32) if oriented_bboxes else torch.empty((0, 8))
        category_ids = torch.tensor(category_ids, dtype=torch.long) if category_ids else torch.empty((0,), dtype=torch.long)
        
        # Apply transforms if specified

        # BELOW IS ADDED by NOAS
        if self.transform:
            image = self.apply_augments(image)
            # image = self.apply_geo_transforms(image,bbox) 
            # Add in above line if desired to test geometrics for training
        # ABOVE IS ADDED BY NOAS
        sample = {
            'image': image,
            'image_id': image_id,
            'captcha_string': captcha_string,
            'bboxes': bboxes,  # Regular bounding boxes
            'oriented_bboxes': oriented_bboxes,  # Oriented bounding boxes
            'category_ids': category_ids,  # Character category IDs
            'num_objects': len(annotations),  # Number of characters
            'original_size': original_size  # (width, height)
        }
        
        return sample
    def apply_augments(image):
        new_img = add_noise_transforms(image)
        new_img = add_lines(new_img)
        new_img = add_salt_pepper(new_img)
        return image

# Custom collate function for bounding boxes
def collate_fn_with_bboxes(batch):
    """Custom collate function to handle variable-length bounding boxes"""
    images = torch.stack([item['image'] for item in batch])
    image_ids = [item['image_id'] for item in batch]
    captcha_strings = [item['captcha_string'] for item in batch]
    
    # Keep bounding boxes as lists since they have variable lengths
    bboxes = [item['bboxes'] for item in batch]
    oriented_bboxes = [item['oriented_bboxes'] for item in batch]
    category_ids = [item['category_ids'] for item in batch]
    num_objects = [item['num_objects'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    
    return {
        'image': images,
        'image_id': image_ids,
        'captcha_string': captcha_strings,
        'bboxes': bboxes,
        'oriented_bboxes': oriented_bboxes,
        'category_ids': category_ids,
        'num_objects': num_objects,
        'original_size': original_sizes
    }

# Helper function to create dataloaders with bboxes
def get_dataloader_with_bboxes(data_folder, batch_size=32, shuffle=True):
    """
    Create dataloader that includes bounding box information
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Create dataset
    dataset = CaptchaDatasetWithBBoxes(
        data_dir=data_folder,
        transform=transform
    )
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn_with_bboxes
    )
    
    return dataloader

# Visualization function with bounding boxes
def visualize_with_bboxes(batch, idx=0):
    """Visualize image with bounding boxes"""
    # Get image and denormalize
    img = batch['image'][idx].squeeze()
    img = img * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
    
    # Get bounding boxes for this image
    bboxes = batch['bboxes'][idx]
    category_ids = batch['category_ids'][idx]
    captcha_string = batch['captcha_string'][idx]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.imshow(img, cmap='gray')
    ax.set_title(f"CAPTCHA: {captcha_string}")
    
    # Draw bounding boxes
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle
        rect = plt.Rectangle((x1, y1), width, height, 
                           linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add category ID as text
        if i < len(category_ids):
            ax.text(x1, y1-5, f'ID: {category_ids[i].item()}', 
                   color='red', fontsize=10, weight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"Number of characters: {len(bboxes)}")
    print(f"Bounding boxes shape: {bboxes.shape}")
    print(f"Category IDs: {category_ids.tolist()}")