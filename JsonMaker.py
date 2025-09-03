import json
import torch
from Src.Inference.SimpleEvaluator import SimpleEvaluator
from Src.Data.DataLoader import CaptchaDataLoader

def make_predictions():
    model_path = "./checkpoints_Priyam/continue_training_20250903_011045/original_best_model.pth"
    data_dir = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/val"
    
    evaluator = SimpleEvaluator(
        model_path=model_path,
        num_classes=36,
        grid_height=10,
        grid_width=40,
        device='cuda',
        default_conf_thresh=0.35
    )
    
    eval_loader = CaptchaDataLoader(data_dir, batch_size=1, shuffle=False)
    raw_predictions = []
    
    print("Running inference for all images...")
    
    for batch_idx, batch in enumerate(eval_loader):
        if batch_idx >= 20000:
            break
            
        image_tensor = batch['Image'][0].unsqueeze(0).to(evaluator.device)
        
        with torch.no_grad():
            raw_output = evaluator.model(image_tensor)
            
        decoded_predictions = evaluator.decode_predictions(raw_output, conf_thresh=0.1)  # Very low threshold
        
        raw_predictions.append({
            'all_detections': decoded_predictions[0],  # All possible detections
            'image_id': batch['ImageID'][0]
        })
        
        if batch_idx % 1000 == 0:
            print(f"Processed {batch_idx} images...")
    
    print(f"Inference complete! Processing {len(raw_predictions)} images with different thresholds...")
    
    for thresh in [0.3, 0.35, 0.4, 0.45]:
        process_threshold(raw_predictions, thresh, evaluator)

def process_threshold(raw_predictions, thresh, evaluator):
    predictions_data = []
    
    for pred_data in raw_predictions:
        # Just filter existing detections by confidence
        filtered_dets = [det for det in pred_data['all_detections'] if det['confidence'] >= thresh]
        
        nms_results = evaluator.simple_nms(filtered_dets, iou_thresh=0.3)
        sorted_detections = evaluator.sort_detections_left_to_right(nms_results)
        
        pred_string = "".join([det['char'] for det in sorted_detections])
        
        annotations = []
        for det in sorted_detections:
            x1, y1, x2, y2 = det['bbox']
            annotations.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "category_id": int(det['class_id'])
            })
        
        predictions_data.append({
            "height": 160,
            "width": 640,
            "image_id": pred_data['image_id'],
            "captcha_string": pred_string,
            "annotations": annotations
        })
    
    output_file = f"predictions_conf_{thresh:.2f}.json"
    with open(output_file, 'w') as f:
        json.dump(predictions_data, f, indent=4)
    
    print(f"Saved {len(predictions_data)} predictions to {output_file} (threshold: {thresh})")

if __name__ == "__main__":
    make_predictions()