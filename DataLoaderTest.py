import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(sys.path)
from Src.Data.DataLoader import CaptchaDataLoader
from Src.Utils.BoundingBoxVisualization import BoundingBoxVisualization



if __name__ == "__main__":
    # Path to the dataset
    data_dir = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/train"

    # Create dataloader
    dataloader = CaptchaDataLoader(data_dir, batch_size=4, shuffle=True)

    # Print dataset size
    print(f"Dataset size: {len(dataloader.dataset)}")

    # Visualize a batch with bounding boxes
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Batch structure:", batch)
        BoundingBoxVisualization(batch, idx=0, output_path="visualization_sample.png")
        break