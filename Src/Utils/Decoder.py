import torch

def decode_yolo_output(predictions, num_classes, S, img_width, img_height, conf_thresh=0.5):
    """
    Decode YOLO head output into bounding boxes and class predictions.
    
    Args:
        predictions: (batch_size, S*S*(5 + num_classes)) - Raw YOLO head output
        num_classes: Number of classes
        S: Grid size (height and width assumed equal)
        img_width: Width of the original image
        img_height: Height of the original image
        conf_thresh: Confidence threshold for filtering predictions
    
    Returns:
        List of detections per image: [bbox, confidence, class_id]
    """
    batch_size = predictions.size(0)
    predictions = predictions.view(batch_size, S, S, 5 + num_classes)

    batch_detections = []
    for b in range(batch_size):
        detections = []
        for i in range(S):
            for j in range(S):
                cell_pred = predictions[b, i, j]
                x_rel, y_rel, w, h = cell_pred[:4]
                objectness = torch.sigmoid(cell_pred[4])
                class_scores = torch.softmax(cell_pred[5:], dim=0)
                class_conf, class_id = torch.max(class_scores, dim=0)

                # Final confidence score
                final_conf = objectness * class_conf
                if final_conf > conf_thresh:
                    # Convert relative coordinates to absolute
                    center_x = (j + x_rel.item()) / S * img_width
                    center_y = (i + y_rel.item()) / S * img_height
                    width = w.item() * img_width
                    height = h.item() * img_height

                    # Convert to corner format
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': final_conf.item(),
                        'class_id': class_id.item()
                    })
        batch_detections.append(detections)
    return batch_detections