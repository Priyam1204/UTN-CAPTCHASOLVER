import os
from Src.Inference.VisualEvaluator import VisualEvaluator

def main():
    # Paths
    model_path = "./checkpoints/best_model.pth"
    val_data_dir = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/val"
    test_data_dir = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/test"
    
    # Initialize evaluator
    evaluator = VisualEvaluator(
        model_path=model_path,
        num_classes=36,
        grid_height=20,
        grid_width=80,
        device='cuda'
    )
    
    # Evaluate on validation set (with ground truth comparison)
    print("=== Evaluating on Validation Set (with Ground Truth) ===")
    evaluator.evaluate_on_validation(
        val_data_dir=val_data_dir,
        num_images=10,
        save_dir='./visual_evaluation_validation'
    )
    
    # Evaluate on test set (predictions only)
    print("\n=== Evaluating on Test Set (Predictions Only) ===")
    evaluator.evaluate_on_test(
        test_data_dir=test_data_dir,
        num_images=10,
        save_dir='./visual_evaluation_test'
    )
    
    print("\nEvaluation completed!")
    print("Check './visual_evaluation_validation' for validation results with ground truth")
    print("Check './visual_evaluation_test' for test predictions")

if __name__ == "__main__":
    main()