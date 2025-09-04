import matplotlib.pyplot as plt
import torch

def BoundingBoxVisualization(Batch, idx=0, output_path="output_visualization.png"):
    """
    Visualizes a single sample from the batch with bounding boxes and saves it to a file.

    Args:
        Batch (dict): A batch containing 'Image', 'BoundingBoxes', etc.
        idx (int): Index of the sample in the batch to visualize.
        output_path (str): Path to save the output visualization.
    """
    print("BoundingBoxVisualization called")  # Debug print

    # Extract the image and denormalize it
    Image = Batch['Image'][idx]  # Shape: [C, H, W]
    Image = Image * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
    Image = Image.permute(1, 2, 0)  # Convert [C, H, W] to [H, W, C]

    # Handle grayscale images
    if Image.shape[-1] == 1:  # Grayscale image
        Image = Image.squeeze(-1)  # Remove the channel dimension
        plt.imshow(Image, cmap='gray')  # Use grayscale colormap
    else:
        plt.imshow(Image)  # For RGB images

    # Extract annotations
    bounding_boxes = Batch['BoundingBoxes'][idx]

    # Extract the CAPTCHA string
    captcha_string = Batch['CaptchaString'][idx]

    # Plot the image
    plt.title(f"CAPTCHA: {captcha_string}")
    plt.axis('off')

    # Draw bounding boxes
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)

    # Save the plot to a file
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.close()  # Close the figure to free memory