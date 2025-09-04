import torch
import os

def LoadModelWeights(model, model_path, device, verbose=True):
    """Load model weights from checkpoint"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if verbose:
        print(f"Loaded model from {model_path}")
        print(f"Model was trained for {checkpoint['epoch']} epochs")
        print(f"Best training loss: {checkpoint['best_loss']:.4f}")
    
    return checkpoint