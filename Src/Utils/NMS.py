from .IoU import iou

def ApplyNMS(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.
    
    Args:
        detections: List of detections per image. Each detection is a dictionary:
                    {'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': int}
        iou_threshold: IoU threshold for suppressing overlapping boxes.
    
    Returns:
        List of filtered detections after NMS.
    """
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    filtered_detections = []

    while detections:
        best = detections.pop(0)
        filtered_detections.append(best)
        detections = [
            det for det in detections
            if iou(best['bbox'], det['bbox']) < iou_threshold
        ]
    return filtered_detections