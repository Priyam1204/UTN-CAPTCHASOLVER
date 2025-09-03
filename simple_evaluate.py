import os
from Src.Inference.SimpleEvaluator import SimpleEvaluator
from Src.Data.DataLoader import CaptchaDataLoader  # ADD THIS LINE

def main():
    # Paths
    model_path = "./checkpoints/continue_training_20250903_011045/best_model.pth"
    val_data_dir = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/val"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}")
        print("Please train the model first or check the checkpoint path.")
        return
    
    # Check if validation directory exists
    if not os.path.exists(val_data_dir):
        print(f"Validation directory not found: {val_data_dir}")
        print("Please check the dataset path.")
        return
    
    # Initialize evaluator
    evaluator = SimpleEvaluator(
        model_path=model_path,
        num_classes=36,
        grid_height=10,   # FIX: Change from 10 to 40
        grid_width=40,   # FIX: Change from 40 to 160  
        device='cuda'
    )
    
    # DEBUG: Test with one image first
    eval_loader = CaptchaDataLoader(val_data_dir, batch_size=1, shuffle=False)
    for batch_idx, batch in enumerate(eval_loader):
        if batch_idx >= 1:  # Just test one image
            break
        
        image = batch['Image'][0]
        image_tensor = image.unsqueeze(0).to(evaluator.device)
        
        # Debug the model output
        evaluator.debug_model_output(image_tensor)
        
        # Try prediction with very low threshold
        predictions = evaluator.predict(image_tensor)
        print(f"\nPredictions found: {len(predictions[0])}")
        
        break  # Exit after first image
    
    # Evaluate on validation set
    print("=== Evaluating on Validation Set ===")
    evaluator.visualize_comparison(
        data_dir=val_data_dir,
        num_images=10,
        save_dir='./simple_evaluation'
    )
    
    print("\nEvaluation completed!")
    print("Check './simple_evaluation' for results")

if __name__ == "__main__":
    main()