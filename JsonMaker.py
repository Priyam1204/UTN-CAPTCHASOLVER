import json
from Src.Inference.SimpleEvaluator import SimpleEvaluator
from Src.Data.DataLoader import CaptchaDataLoader

def make_predictions_json(model_path, data_dir, conf_thresh=0.35, num_images=10, output_file='predictions.json'):
    """Create predictions.json in README format"""
    
    evaluator = SimpleEvaluator(
        model_path=model_path,
        num_classes=36,
        grid_height=10,
        grid_width=40,
        device='cuda',
        default_conf_thresh=conf_thresh
    )
    
    eval_loader = CaptchaDataLoader(data_dir, batch_size=1, shuffle=False)
    predictions_data = []
    
    for batch_idx, batch in enumerate(eval_loader):
        if batch_idx >= num_images:
            break
        
        image = batch['Image'][0]
        image_tensor = image.unsqueeze(0).to(evaluator.device)
        image_id = batch['ImageID'][0]
        
        # Get left-to-right sorted predictions
        pred_string, detections = evaluator.extract_captcha_string(image_tensor)
        
        # Create annotations
        annotations = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']  # Already in corner format from decode_predictions
            
            annotations.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "category_id": int(det['class_id'])
            })
        
        # Create image entry
        predictions_data.append({
            "height": 160,
            "width": 640,
            "image_id": image_id,
            "captcha_string": pred_string,
            "annotations": annotations
        })
    
    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(predictions_data, f, indent=4)
    
    print(f"Saved {len(predictions_data)} predictions to {output_file} (threshold: {conf_thresh})")

def make_multiple_predictions():
    """Create predictions with multiple confidence thresholds"""
    model_path = "./checkpoints_Priyam/continue_training_20250903_011045/best_model.pth"
    data_dir = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/val"
    
    # Specific confidence thresholds to test
    thresholds = [0.35, 0.4, 0.45, 0.5, 0.55]
    
    print("🚀 Creating predictions with multiple thresholds...")
    print("="*60)
    
    for thresh in thresholds:
        output_file = f"predictions_conf_{thresh:.2f}.json"
        make_predictions_json(
            model_path=model_path,
            data_dir=data_dir,
            conf_thresh=thresh,
            num_images=10,
            output_file=output_file
        )
    
    print("="*60)
    print("✅ All prediction files created!")
    print("Files generated:")
    for thresh in thresholds:
        print(f"   - predictions_conf_{thresh:.2f}.json")

if __name__ == "__main__":
    make_multiple_predictions()