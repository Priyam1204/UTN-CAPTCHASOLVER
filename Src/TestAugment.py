import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from torch.utils.data import DataLoader
from Src.Data.DataSet import CaptchaDataset
from Src.Data.Collate import CaptchaCollateFn
from Src.Data.Transform import Transform

def denorm(img_t):
    # inverse of Normalize(mean=[0.5], std=[0.5]) -> x = x*0.5 + 0.5
    return (img_t * 0.5 + 0.5).clamp(0, 1)

def visualize(img, bboxes, title="Image", save_path=None, show=False):
    """Visualize one grayscale image with bounding boxes."""
    if torch.is_tensor(img):
        img = denorm(img).squeeze().cpu().numpy()   # de-normalize and to numpy

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.imshow(img, cmap="gray")
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu()
    for box in bboxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, linewidth=2, edgecolor="red"))
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    data_dir = "/home/utn/abap44us/Downloads/UTN-CV25-Captcha-Dataset/part2/train"

    dataset = CaptchaDataset(data_dir, transform=Transform(), use_geo_aug=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        collate_fn=CaptchaCollateFn, num_workers=0)

    batch = next(iter(loader))
    img = batch["Image"][0]         # (1,H,W), normalized
    bboxes = batch["BoundingBoxes"][0]

    print("captcha:", batch["CaptchaString"][0])
    print("num boxes:", len(bboxes))

    # Save to file so it works on headless servers; set show=True if you have a GUI
    visualize(img, bboxes, "After Augmentations",
              save_path="viz/augmented_sample.png", show=False)
