def iou(box1, box2):
    """
    Calculate Intersection-over-Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] - First bounding box
        box2: [x1, y1, x2, y2] - Second bounding box
    
    Returns:
        IoU value (float)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return inter_area / union_area