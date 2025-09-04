import os
import argparse
from Src.Inference.Inference import ModelInference

def main():
    
    parser = argparse.ArgumentParser(description='CAPTCHA Model Inference')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to directory containing images')
    parser.add_argument('--num_images', type=int, default=None,
                       help='Number of images to process (default: all images)')
    
    args = parser.parse_args()
    
    #Model checkpoint path
    ModelPath = "./ModelWeights/30EpochsBestModel.pth"
    
    if not os.path.exists(ModelPath):
        print(f"Model checkpoint not found: {ModelPath}")
        return
    
        #Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Data directory not found: {args.data_dir}")
        return
    
    # Initialize evaluator with fixed settings
    evaluator = ModelInference(
        ModelPath=ModelPath,
        NumClasses=36,
        GridHeight=10,
        GridWidth=40,
        Device='cuda',
        ConfThresh=0.4,
        IoUThresh=0.3
    )
    
    # Run inference
    Results = evaluator.PredictFolder(
        FolderPath=args.data_dir,
        NumImages=args.num_images,
        SaveResults=True
    )

if __name__ == "__main__":
    main()