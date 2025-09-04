import torch

def CaptchaCollateFn(batch):
    """
    Custom collate function to handle variable-length bounding boxes.
    """
    images = torch.stack([item['Image'] for item in batch])
    ImageIDs = [item['ImageID'] for item in batch]
    CaptchaStrings = [item['CaptchaString'] for item in batch]

    # Keep bounding boxes as lists since they have variable lengths
    BoundingBoxes = [item['BoundingBoxes'] for item in batch]
    OrientedBoundingBoxes = [item['OrientedBoundingBoxes'] for item in batch]
    CategoryIDs = [item['CategoryIDs'] for item in batch]
    NumberofObjects = [item['NumberofObjects'] for item in batch]
    ImageSize = [item['ImageSize'] for item in batch]

    return {
        'Image': images,
        'ImageID': ImageIDs,
        'CaptchaString': CaptchaStrings,
        'BoundingBoxes': BoundingBoxes,
        'OrientedBoundingBoxes': OrientedBoundingBoxes,
        'CategoryIDs': CategoryIDs,
        'NumberofObjects': NumberofObjects,
        'ImageSize': ImageSize
    }