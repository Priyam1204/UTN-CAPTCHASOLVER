def CalculateIoU(Box1, Box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = Box1
    x1_2, y1_2, x2_2, y2_2 = Box2
    
    #Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    InterSection = (x2_i - x1_i) * (y2_i - y1_i)
    Area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    Area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    Union = Area1 + Area2 - InterSection
    
    return InterSection / Union if Union > 0 else 0.0

def ApplyNMS(Detections, IoUThreshold=0.3):
    """Apply Non-Maximum Suppression to remove overlapping detections"""
    if len(Detections) == 0:
        return []
    
    #Sort by confidence
    Detections = sorted(Detections, key=lambda x: x['confidence'], reverse=True)
    
    FilteredDetections = []
    while Detections:
        BestDetection = Detections.pop(0)
        FilteredDetections.append(BestDetection)
        
        #Remove overlapping Detections
        Detections = [
            det for det in Detections
            if CalculateIoU(BestDetection['bbox'], det['bbox']) < IoUThreshold
        ]
    
    return FilteredDetections

