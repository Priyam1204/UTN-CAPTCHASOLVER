import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Src.Model.Model import CaptchaSolverModel
from Src.Inference.PredictionDecoder import PredictionDecoder
from Src.Utils.CharacterMapping import CreateCharacterMapping
from Src.Utils.LoadModel import LoadModelWeights
from Src.Utils.NMS import ApplyNMS
from Src.Utils.DetectionSort import SortDetectionsLeftToRight
import json

class ModelInference:
    def __init__(self, ModelPath, NumClasses=36, GridHeight=10, GridWidth=40, 
                 Device='cuda', ConfThresh=0.4, IoUThresh=0.3):
        
        self.ConfThresh = ConfThresh
        self.IoUThresh = IoUThresh
        self.Device = torch.device(Device if torch.cuda.is_available() else 'cpu')
        
        # Model setup
        self.Model = CaptchaSolverModel(NumClasses, GridHeight, GridWidth).to(self.Device)
        LoadModelWeights(self.Model, ModelPath, self.Device)
        
        # Utilities
        self.IdxToChar = CreateCharacterMapping()
        self.Decoder = PredictionDecoder(GridHeight, GridWidth, 640, 160, NumClasses)
        
        print(f"ModelInference ready | Conf: {ConfThresh} | IoU: {IoUThresh}")
    
    def PredictImage(self, ImagePath):
        """Predict CAPTCHA string from image file"""
        # Load image
        Image_PIL = Image.open(ImagePath).convert('L')
        ImageTensor = torch.from_numpy(np.array(Image_PIL)).unsqueeze(0).float() / 255.0

        # Resize to 640x160
        ImageTensor = torch.nn.functional.interpolate(
            ImageTensor.unsqueeze(0), size=(160, 640), mode='bilinear'
        ).to(self.Device)
        
        return self._ExtractCaptcha(ImageTensor)
    
    def PredictFolder(self, FolderPath, NumImages=None, SaveResults=True):
        """Predict images in folder
        
        Args:
            FolderPath: Path to folder containing images
            NumImages: Number of images to process (None = all images)
            SaveResults: Whether to save visualizations and summary
        """
        print("Inference started")
        
        # Get image files (filter out result files)
        ImageFiles = []
        for f in os.listdir(FolderPath):
            if (f.lower().endswith(('.png', '.jpg', '.jpeg')) and 
                not f.endswith('_result.png') and 
                not f.startswith('result_')):
                ImageFiles.append(f)
        
        # Sort for consistent order
        ImageFiles.sort()
        
        # Limit number of images if specified
        TotalAvailable = len(ImageFiles)
        if NumImages is not None:
            if NumImages <= 0:
                print("NumImages must be positive")
                return []
            
            if NumImages > TotalAvailable:
                NumImages = TotalAvailable
            
            ImageFiles = ImageFiles[:NumImages]
        
        Results = []
        
        for i, FileName in enumerate(ImageFiles):
            try:
                ImagePath = os.path.join(FolderPath, FileName)
                CaptchaString, Detections = self.PredictImage(ImagePath)
                
                # Create annotations for each detection
                Annotations = []
                for Detection in Detections:
                    x1, y1, x2, y2 = Detection['bbox']
                    Annotations.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "category_id": int(Detection.get('class_id', 0))
                    })
                
                Result = {
                    "height": 160,
                    "width": 640,
                    "image_id": os.path.splitext(FileName)[0],  # Remove file extension
                    "captcha_string": CaptchaString,
                    "annotations": Annotations
                }
                Results.append(Result)
                
                print(f"Processing [{i+1}/{len(ImageFiles)}] {FileName}: '{CaptchaString}'")
                
                # Save visualization
                if SaveResults:
                    self._SaveVisualization(ImagePath, Detections, CaptchaString)
                    
            except Exception as e:
                print(f"Error processing {FileName}: {e}")
                Result = {
                    "height": 160,
                    "width": 640,
                    "image_id": os.path.splitext(FileName)[0],
                    "captcha_string": "",
                    "annotations": []
                }
                Results.append(Result)
        
        # Save summary
        if SaveResults:
            self._SaveSummary(Results, len(ImageFiles), TotalAvailable)
        
        print("Inference completed")
        return Results
    
    def _ExtractCaptcha(self, ImageTensor):
        """Internal method to extract CAPTCHA from tensor"""
        self.Model.eval()
        with torch.no_grad():
            # Get predictions
            RawPredictions = self.Model(ImageTensor)
            DecodedPredictions = self.Decoder.Decode(RawPredictions, self.ConfThresh, self.IdxToChar)
            
            if not DecodedPredictions or not DecodedPredictions[0]:
                return "", []
            
            # Apply NMS and sort
            Detections = ApplyNMS(DecodedPredictions[0], self.IoUThresh)
            Detections = SortDetectionsLeftToRight(Detections)
            
            # Extract string
            CaptchaString = "".join([det['char'] for det in Detections])
            return CaptchaString, Detections
    
    def _SaveVisualization(self, ImagePath, Detections, CaptchaString):
        """Save annotated image to results folder"""
        # Create results directory
        ResultsDir = 'inference_results'
        os.makedirs(ResultsDir, exist_ok=True)
        
        # Load image
        Image_PIL = Image.open(ImagePath)
        Fig, Ax = plt.subplots(figsize=(12, 4))
        Ax.imshow(Image_PIL, cmap='gray')
        Ax.set_title(f'CAPTCHA: "{CaptchaString}"', fontsize=16, fontweight='bold')
        
        # Draw boxes
        for Detection in Detections:
            x1, y1, x2, y2 = Detection['bbox']
            Rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            Ax.add_patch(Rect)
            Ax.text(x1, y1-5, Detection['char'], color='red', fontsize=12, fontweight='bold')
        
        Ax.axis('off')
        
        # Save to results folder
        FileName = os.path.basename(ImagePath)
        OutputPath = os.path.join(ResultsDir, f"result_{FileName}")
        plt.savefig(OutputPath, bbox_inches='tight', dpi=100)
        plt.close()
    
    def _SaveSummary(self, Results, ProcessedCount, TotalAvailable):
        """Save results summary"""
        OutputFile = f"predictions_conf_{self.ConfThresh:.2f}.json"
        
        with open(OutputFile, 'w') as f:
            json.dump(Results, f, indent=4)
        
        SuccessCount = sum(1 for r in Results if r.get('captcha_string', '') != '')
        
        print(f"Results saved to: {OutputFile}")