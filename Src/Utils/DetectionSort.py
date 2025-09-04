def SortDetectionsLeftToRight(Detections):
    """Sort detections from left to right based on bounding box left edge"""
    return sorted(Detections, key=lambda det: det['bbox'][0]) if Detections else []