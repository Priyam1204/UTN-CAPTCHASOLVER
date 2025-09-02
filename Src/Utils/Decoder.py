import torch

def decode_yolo_output(predictions, num_classes, grid_height, grid_width, img_width, img_height, conf_thresh=0.5):
    """
    Decode YOLO head output into bounding boxes and class predictions for rectangular grids.
    
    Args:
        predictions: (batch_size, grid_height*grid_width*(5 + num_classes)) - Raw YOLO head output
        num_classes: Number of classes
        grid_height: Grid height (e.g., 20)
        grid_width: Grid width (e.g., 80)
        img_width: Width of the original image (e.g., 640)
        img_height: Height of the original image (e.g., 160)
        conf_thresh: Confidence threshold for filtering predictions
    
    Returns:
        List of detections per image: [bbox, confidence, class_id]
    """
    batch_size = predictions.size(0)
    predictions = predictions.view(batch_size, grid_height, grid_width, 5 + num_classes)

    batch_detections = []
    for b in range(batch_size):
        detections = []
        for i in range(grid_height):  # Height dimension
            for j in range(grid_width):   # Width dimension
                cell_pred = predictions[b, i, j]
                x_rel, y_rel, w, h = cell_pred[:4]
                objectness = torch.sigmoid(cell_pred[4])
                class_scores = torch.softmax(cell_pred[5:], dim=0)
                class_conf, class_id = torch.max(class_scores, dim=0)

                # Final confidence score
                final_conf = objectness * class_conf
                if final_conf > conf_thresh:
                    # Convert relative coordinates to absolute (rectangular grid)
                    center_x = (j + x_rel.item()) / grid_width * img_width   # Use grid_width for x
                    center_y = (i + y_rel.item()) / grid_height * img_height # Use grid_height for y
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

# Backward compatibility function (for square grids)
def decode_yolo_output_square(predictions, num_classes, S, img_width, img_height, conf_thresh=0.5):
    """
    Backward compatibility function for square grids.
    """
    return decode_yolo_output(predictions, num_classes, S, S, img_width, img_height, conf_thresh)