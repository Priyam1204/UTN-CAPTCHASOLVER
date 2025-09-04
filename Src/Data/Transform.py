from torchvision import transforms

def Transform():
    """
    Transform image to tensor and normalize.
    """
    return transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5], std=[0.5])  
    ])