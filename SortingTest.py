from Src.Inference.SimpleEvaluator import SimpleEvaluator
from Src.Data.DataLoader import CaptchaDataLoader

def test_detailed_predictions():
    """Test predictions with detailed confidence and sorting for 10 images"""
    
    model_path = "./checkpoints/continue_training_20250903_011045/best_model.pth"
    val_data_dir = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/val"
    
    # ✅ Initialize with custom threshold
    evaluator = SimpleEvaluator(
        model_path=model_path,
        num_classes=36,
        grid_height=10,
        grid_width=40,
        device='cuda',
        default_conf_thresh=0.4  # ✅ Single place to control threshold
    )
    
    print("="*80)
    print(f"🚀 DETAILED PREDICTION TEST - 10 IMAGES")
    print(f"📊 Using threshold: {evaluator.default_conf_thresh}")
    print("="*80)
    
    # Test on 10 images
    eval_loader = CaptchaDataLoader(val_data_dir, batch_size=1, shuffle=False)
    
    correct_predictions = 0
    total_detections = 0
    confidence_scores = []
    
    for batch_idx, batch in enumerate(eval_loader):
        if batch_idx >= 10:
            break
        
        image = batch['Image'][0]
        image_tensor = image.unsqueeze(0).to(evaluator.device)
        ground_truth = batch['CaptchaString'][0]
        image_id = batch['ImageID'][0]
        
        print(f"\n📷 Image {batch_idx + 1}/10 (ID: {image_id})")
        print(f"🎯 Ground Truth: '{ground_truth}'")
        
        # Get detailed predictions with sorting
        pred_string, detections = evaluator.extract_captcha_string(image_tensor)
        
        # Show unsorted vs sorted comparison
        unsorted_preds = evaluator.predict(image_tensor, sort_left_to_right=False)
        unsorted_chars = [det['char'] for det in unsorted_preds[0]]
        unsorted_string = ''.join(unsorted_chars)
        
        print(f"🔄 Unsorted Prediction: '{unsorted_string}'")
        print(f"✅ Sorted Prediction:   '{pred_string}'")
        
        # Check if prediction is correct
        is_correct = pred_string == ground_truth
        if is_correct:
            correct_predictions += 1
        
        status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        print(f"🎯 Result: {status}")
        
        # Show detailed detection info
        if detections:
            print(f"📋 Detections ({len(detections)} found):")
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                char = det['char']
                conf = det['confidence']
                class_id = det['class_id']
                
                # Calculate center and position info
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                print(f"   {i+1}. '{char}' (ID:{class_id}) - Conf: {conf:.4f}")
                print(f"      📍 Position: x={center_x:.0f}, y={center_y:.0f}")
                print(f"      📏 Size: {width:.0f}x{height:.0f}")
                print(f"      🔲 BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                
                confidence_scores.append(conf)
                total_detections += 1
        else:
            print("📋 No detections found!")
        
        # Show sorting effect
        if len(detections) > 1:
            positions = [f"{det['char']}@{int(det['bbox'][0])}" for det in detections]
            print(f"🔀 Left-to-Right Order: {' → '.join(positions)}")
            
            if unsorted_string != pred_string:
                print(f"🔄 Sorting Effect: '{unsorted_string}' → '{pred_string}'")
            else:
                print(f"🔄 Sorting Effect: No change needed")
        
        print("-" * 60)
    
    # Summary statistics
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    print(f"🎯 Accuracy: {correct_predictions}/10 ({correct_predictions/10*100:.1f}%)")
    print(f"🔍 Total Detections: {total_detections}")
    print(f"📈 Average Detections per Image: {total_detections/10:.1f}")
    
    if confidence_scores:
        import numpy as np
        conf_array = np.array(confidence_scores)
        print(f"📊 Confidence Statistics:")
        print(f"   Mean: {conf_array.mean():.4f}")
        print(f"   Std:  {conf_array.std():.4f}")
        print(f"   Min:  {conf_array.min():.4f}")
        print(f"   Max:  {conf_array.max():.4f}")
        print(f"   Median: {np.median(conf_array):.4f}")
        
        # Confidence distribution
        high_conf = (conf_array > 0.8).sum()
        med_conf = ((conf_array > 0.5) & (conf_array <= 0.8)).sum()
        low_conf = (conf_array <= 0.5).sum()
        
        print(f"📈 Confidence Distribution:")
        print(f"   High (>0.8): {high_conf}/{len(conf_array)} ({high_conf/len(conf_array)*100:.1f}%)")
        print(f"   Medium (0.5-0.8): {med_conf}/{len(conf_array)} ({med_conf/len(conf_array)*100:.1f}%)")
        print(f"   Low (≤0.5): {low_conf}/{len(conf_array)} ({low_conf/len(conf_array)*100:.1f}%)")
    
    print("="*80)

def test_threshold_comparison():
    """Compare different thresholds on the same images"""
    
    model_path = "./checkpoints/continue_training_20250903_011045/best_model.pth"
    val_data_dir = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/val"
    
    evaluator = SimpleEvaluator(
        model_path=model_path,
        num_classes=36,
        grid_height=10,
        grid_width=40,
        device='cuda',
        default_conf_thresh=0.3
    )
    
    thresholds = [0.2, 0.3, 0.4, 0.5]
    
    print("\n" + "="*80)
    print("🔄 THRESHOLD COMPARISON TEST")
    print("="*80)
    
    eval_loader = CaptchaDataLoader(val_data_dir, batch_size=1, shuffle=False)
    
    for batch_idx, batch in enumerate(eval_loader):
        if batch_idx >= 3:  # Test on 3 images
            break
        
        image = batch['Image'][0]
        image_tensor = image.unsqueeze(0).to(evaluator.device)
        ground_truth = batch['CaptchaString'][0]
        image_id = batch['ImageID'][0]
        
        print(f"\n📷 Image {batch_idx + 1}/3 (ID: {image_id})")
        print(f"🎯 Ground Truth: '{ground_truth}'")
        print()
        
        for thresh in thresholds:
            evaluator.set_thresholds(conf_thresh=thresh)
            pred_string, detections = evaluator.extract_captcha_string(image_tensor)
            
            status = "✅" if pred_string == ground_truth else "❌"
            conf_info = ""
            if detections:
                avg_conf = sum(det['confidence'] for det in detections) / len(detections)
                conf_info = f" (avg conf: {avg_conf:.3f})"
            
            print(f"   Thresh {thresh:.1f}: '{pred_string}' {status} - {len(detections)} dets{conf_info}")
        
        print("-" * 60)

if __name__ == "__main__":
    # Run detailed predictions test
    test_detailed_predictions()
    
    # Optional: Run threshold comparison
    print("\n" + "🔄" * 40)
    response = input("Run threshold comparison test? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        test_threshold_comparison()