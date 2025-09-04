import os
from Src.Inference.Inference import ModelInference

def main():
    # Paths
    ModelPath = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/checkpoints_Priyam/continue_training_20250903_011045/best_model.pth"
    ValDataDir = "/home/utn/omul36yx/git/UTN-CAPTCHASOLVER/UTN-CV25-Captcha-Dataset/part2/val/images"
    
    # Check if model exists
    if not os.path.exists(ModelPath):
        print(f"Model checkpoint not found: {ModelPath}")
        return
    
    # Check if validation directory exists
    if not os.path.exists(ValDataDir):
        print(f"Validation directory not found: {ValDataDir}")
        return
    
    # Initialize evaluator
    evaluator = ModelInference(
        ModelPath=ModelPath,
        NumClasses=36,
        GridHeight=10,
        GridWidth=40,
        Device='cuda'
    )
    
    # Run inference
    Results = evaluator.PredictFolder(
        FolderPath=ValDataDir,
        NumImages=10,
        SaveResults=True
    )

if __name__ == "__main__":
    main()