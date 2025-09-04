import os, json, random, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset
from torchvision import transforms, tv_tensors
from torchvision.transforms import v2 as T

def Apply_Geo_Augmentations(image, target):
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


def add_noise_transforms(image):
    # Injects photometric noise into image
    augments = T.Compose([
        T.RandomApply([T.RandomAutocontrast()], p=0.3),
        T.RandomApply([T.RandomEqualize()], p=0.15),
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.25),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.25),
    ])
    return augments(image)

def add_salt_pepper(image, prob=0.5, salt_prob=0.005, pepper_prob=0.005):
    # injects salt and pepper pixels into image
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

def add_lines(image, prob=0.3, min_ratio=0.3, max_ratio=1.0, width_range=(1, 4), max_lines=4):
    # adds lines of varying length into image
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
    # adds circles into image
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
    # adds symbol glyphs into image
    if random.random() > prob:
        return image

    W, H = image.size
    base = image.convert("L")

    for _ in range(random.randint(1, max_symbols)):
        ch = random.choice(symbols)

        if fonts and len(fonts) > 0:
            try:
                fp = random.choice(fonts)
                fsz = random.randint(*size_range)
                font = ImageFont.truetype(fp, fsz)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        bbox = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), ch, font=font)
        gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        glyph = Image.new("L", (max(1, gw), max(1, gh)), 0)
        gdraw = ImageDraw.Draw(glyph)

        fill_val = 0 if random.random() < 0.5 else 255                 
        alpha = random.randint(*alpha_range)                           
        gdraw.text((0, 0), ch, font=font, fill=alpha)                   

        ang = random.uniform(*angle_range)
        glyph = glyph.rotate(ang, expand=True)

        cx = random.uniform(W * 0.15, W * 0.85)
        cy = random.uniform(H * 0.25, H * 0.75)
        x0 = int(cx - glyph.width / 2)
        y0 = int(cy - glyph.height / 2)

        patch = Image.new("L", glyph.size, color=fill_val)
        base.paste(patch, (x0, y0), mask=glyph)

    return base


def Apply_Noise_Policy(image,
                       clean_prob=0.4,
                       photometric_prob=0.2,
                       clutter_prob=0.25,
                       occlusion_prob=0.15):
    """
    One-shot augmentation policy:
      - clean_prob:        return clean unedited image
      - photometric_prob:  photometric noise only
      - clutter_prob:      short lines, small circles w/ salt/pepper
      - occlusion_prob:    long diagonal line(s), bold circles w/ heavy salt/pepper
    """
    r = random.random()
    if r < clean_prob:
        return image

    r2 = random.random()
    thresholds = (photometric_prob, photometric_prob + clutter_prob, photometric_prob + clutter_prob + occlusion_prob)

    def maybe_salt(img, p=0.25):
        return add_salt_pepper(img, prob=p, salt_prob=0.003, pepper_prob=0.003)

    if r2 < thresholds[0]:
        img = add_noise_transforms(image)
        return img

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

    img = add_noise_transforms(image) if random.random() < 0.5 else image
    if random.random() < 0.6:
        img = add_lines(img, prob=1.0, min_ratio=0.8, max_ratio=1.2, width_range=(2, 4), max_lines=2)
    else:
        img = add_circles(img, prob=1.0, radius_range=(8, 18), max_circles=3)
    img = maybe_salt(img, p=0.15)
    return img