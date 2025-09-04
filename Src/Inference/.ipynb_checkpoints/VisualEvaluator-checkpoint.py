import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Src.Model.Model import CaptchaSolverModel
from Src.Data.DataLoader import CaptchaDataLoader
from Src.Utils.TargetPreparer import TargetPreparer
from Src.Utils.Decoder import decode_yolo_output
from Src.Utils.NMS import ApplyNMS
import json

class VisualEvaluator:
    def __init__(self, model_path, num_classes=36, grid_height=20, grid_width=80, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.img_width = 640    # Image width
        self.img_height = 160   # Image height
        self._json_records = []   # collects one dict per image

        # Initialize model
        self.model = CaptchaSolverModel(
            num_classes=num_classes,
            grid_height=grid_height,
            grid_width=grid_width
        ).to(self.device)
        
        # Load trained weights
        self.load_model(model_path)
        
        
        
        # Character mapping (adjust based on your dataset)
        self.idx_to_char = self._create_character_mapping()

    def _preds_to_json_entry(self, image_id, predictions, img_h=None, img_w=None):
        """Build one JSON entry in your target schema from model predictions."""
        img_h = int(img_h or self.img_height)
        img_w = int(img_w or self.img_width)

        anns = []
        chars = []

        preds = predictions[0] if (predictions and len(predictions[0]) > 0) else []
        # left→right for a stable captcha_string
        for x, y, w, h, conf, class_id in sorted(preds, key=lambda p: p[0]):
            x1, y1 = float(x), float(y)
            x2, y2 = float(x + w), float(y + h)
            anns.append({
                "bbox": [x1, y1, x2, y2],
                "oriented_bbox": [x1, y1, x2, y1, x2, y2, x1, y2],
                "category_id": int(class_id)
            })
            chars.append(self.idx_to_char.get(int(class_id), "?"))

        return {
            "height": img_h,
            "width": img_w,
            "image_id": str(image_id),
            "captcha_string": "".join(chars),
            "annotations": anns
        }
    # --- ADD near the top of the file ---
    def ctc_greedy_decode(logits, blank=0):
        """
        logits: (T, B, C) or (B, T, C) torch tensor
        returns list[str] length B (no mapping applied)
        """
        if logits.dim() != 3:
            raise ValueError("CTC logits must be 3D (T,B,C) or (B,T,C)")
        # force (T,B,C)
        if logits.shape[0] != logits.size(0) or logits.size(0) == logits.size(1):
            pass  # shape check is noisy; we handle both layouts below
        if logits.shape[0] == logits.size(0) and logits.shape[1] != logits.size(0):
            # assume (T,B,C)
            T, B, C = logits.shape
            path = logits.argmax(-1)          # (T,B)
            path = path.detach().cpu().numpy()
            seqs = []
            for b in range(B):
                prev = None
                out = []
                for t in range(T):
                    k = int(path[t, b])
                    if k != prev and k != blank:
                        out.append(k)
                    prev = k
                seqs.append(out)
            return seqs
        else:
            # assume (B,T,C)
            B, T, C = logits.shape
            path = logits.argmax(-1).detach().cpu().numpy()  # (B,T)
            seqs = []
            for b in range(B):
                prev = None
                out = []
                for t in range(T):
                    k = int(path[b, t])
                    if k != prev and k != blank:
                        out.append(k)
                    prev = k
                seqs.append(out)
            return seqs

    def ler(pred_strs, gt_strs):
        """character-level Levenshtein Error Rate"""
        try:
            from Levenshtein import distance as lev
        except Exception:
            # fallback: simple DP if python-Levenshtein isn't installed
            def lev(a,b):
                m,n=len(a),len(b)
                dp=list(range(n+1))
                for i in range(1,m+1):
                    prev,dp[0]=dp[0],i
                    for j in range(1,n+1):
                        cur=dp[j]
                        dp[j]=min(dp[j]+1, dp[j-1]+1, prev + (a[i-1]!=b[j-1]))
                        prev=cur
                return dp[n]
        tot=0; L=0
        for p,g in zip(pred_strs, gt_strs):
            tot += lev(p,g)
            L   += max(1,len(g))
        return tot/float(L)

    def _append_and_optionally_write_json(self, entry, out_path=None):
        """Append to in-memory list and write labels.json if a path is given."""
        if not hasattr(self, "_json_records"):
            self._json_records = []
        self._json_records.append(entry)
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self._json_records, f, ensure_ascii=False, indent=4)

    # Optional alias (lets you call either name)
    def _append_and_write_json(self, entry, json_path):
        return self._append_and_optionally_write_json(entry, out_path=json_path)


    # --- REPLACE your _create_character_mapping with this ---
    def _create_character_mapping(self, blank_idx=36):
        # 0-9 + A-Z
        chars = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)]
        self.blank_idx = blank_idx
        self.idx_to_char = {i:c for i,c in enumerate(chars)}
        # blank is not a printable char; skip mapping for it

    def load_model(self, model_path):
        """Load model weights from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Model was trained for {checkpoint['epoch']} epochs")
        print(f"Best training loss: {checkpoint['best_loss']:.4f}")
    
    def predict(self, image_tensor):
        """
        Make prediction on a single image tensor
        
        Args:
            image_tensor: torch tensor of shape (1, 1, H, W)
            
        Returns:
            predictions: List of detected characters with bounding boxes
        """
        self.model.eval()
    
        with torch.no_grad():
            # Get model predictions
            raw_predictions = self.model(image_tensor)
            
            # Decode YOLO output to bounding boxes using rectangular grid
            decoded_predictions = decode_yolo_output(
                predictions=raw_predictions,
                num_classes=self.num_classes,
                grid_height=self.grid_height,  # 20
                grid_width=self.grid_width,    # 80
                img_width=self.img_width,      # 640
                img_height=self.img_height,    # 160
                conf_thresh=0.05
            )
            
            # Apply Non-Maximum Suppression to each batch
            final_predictions = []
            for batch_detections in decoded_predictions:
                # ApplyNMS expects the format from decode_yolo_output
                # which is: {'bbox': [x1, y1, x2, y2], 'confidence': conf, 'class_id': id}
                if len(batch_detections) > 0:
                    nms_results = ApplyNMS(batch_detections, iou_threshold=0.0)
                    
                    # Convert to the format expected by the rest of the code
                    # [x, y, w, h, conf, class_id] format
                    batch_formatted = []
                    for detection in nms_results:
                        x1, y1, x2, y2 = detection['bbox']
                        w = x2 - x1
                        h = y2 - y1
                        x = x1  # Top-left corner
                        y = y1  # Top-left corner
                        
                        batch_formatted.append([
                            x, y, w, h, 
                            detection['confidence'], 
                            detection['class_id']
                        ])
                    final_predictions.append(batch_formatted)
                else:
                    final_predictions.append([])
        
        return final_predictions
    
    def visualize_comparison(self, data_dir, num_images=10, save_dir='./visual_evaluation'):
        """
        Create side-by-side comparison of ground truth vs predictions
        
        Args:
            data_dir: Path to evaluation dataset
            num_images: Number of images to evaluate
            save_dir: Directory to save visualizations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create data loader
        eval_loader = CaptchaDataLoader(
            data_dir,
            batch_size=1,
            shuffle=False
        )
        
        print(f"Creating visual evaluation for {num_images} images...")
        
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx >= num_images:
                break
                
            image = batch['Image'][0]  # Single image tensor (1, H, W)
            image_path = batch.get('ImagePath', [f'image_{batch_idx}'])[0]
            
            # Get ground truth
            gt_bboxes = batch['BoundingBoxes'][0] if len(batch['BoundingBoxes'][0]) > 0 else []
            gt_categories = batch['CategoryIDs'][0] if len(batch['CategoryIDs'][0]) > 0 else []
            
            # Make prediction
            image_tensor = image.unsqueeze(0).to(self.device)  # Add batch dimension
            predictions = self.predict(image_tensor)

                        # ---- NEW: derive an image_id and write JSON for predictions ----
            # use filename stem as image_id when available
            image_filename = os.path.basename(str(image_path))
            image_id = os.path.splitext(image_filename)[0]

            json_entry = self._preds_to_json_entry(
                image_id=image_id,
                predictions=predictions,
                img_h=self.img_height,
                img_w=self.img_width
            )
            json_out = os.path.join(save_dir, "labels.json")
            self._append_and_write_json(json_entry, json_out)
            # ---- end NEW ----

            
            # Create visualization
            self._create_side_by_side_plot(
                image, gt_bboxes, gt_categories, predictions, 
                image_path, batch_idx, save_dir
            )
            
            print(f"Processed image {batch_idx + 1}/{num_images}")
        
        print(f"Visual evaluation completed! Results saved to {save_dir}")
    
    def _create_side_by_side_plot(self, image, gt_bboxes, gt_categories, predictions, 
                                  image_path, idx, save_dir):
        """Create side-by-side comparison plot"""
        
        # Convert tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            image_np = image.squeeze().cpu().numpy()
        else:
            image_np = np.array(image)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Ground Truth (Left)
        ax1.imshow(image_np, cmap='gray')
        ax1.set_title(f'Ground Truth - Image {idx + 1}', fontsize=16, fontweight='bold')
        
        # Draw ground truth bounding boxes
        gt_text = []
        if len(gt_bboxes) > 0 and len(gt_categories) > 0:
            for i, (bbox, cat_id) in enumerate(zip(gt_bboxes, gt_categories)):
                x, y, w, h = bbox.tolist()
                char = self.idx_to_char.get(int(cat_id.item()), '?')
                
                # Draw bounding box
                rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                       edgecolor='lime', facecolor='none')
                ax1.add_patch(rect)
                
                # Add text with character and ID
                ax1.text(x, y-8, f'{char} (ID:{int(cat_id.item())})', 
                        color='lime', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                
                gt_text.append(f'{char}(ID:{int(cat_id.item())})')
        
        ax1.axis('off')
        
        # Add ground truth sequence at bottom
        gt_sequence = ' '.join(gt_text) if gt_text else 'No characters'
        ax1.text(0.5, -0.15, f'GT Sequence: {gt_sequence}', 
                transform=ax1.transAxes, ha='center', va='top',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        
        # Predictions (Right)
        ax2.imshow(image_np, cmap='gray')
        ax2.set_title(f'Predictions - Image {idx + 1}', fontsize=16, fontweight='bold')
        
        # Draw prediction bounding boxes
        pred_text = []
        if len(predictions) > 0 and len(predictions[0]) > 0:
            for pred in predictions[0]:
                x, y, w, h, conf, class_id = pred
                char = self.idx_to_char.get(int(class_id), '?')
                
                # Draw bounding box
                rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                       edgecolor='red', facecolor='none')
                ax2.add_patch(rect)
                
                # Add text with character, ID, and confidence
                ax2.text(x, y-8, f'{char} (ID:{int(class_id)}) {conf:.2f}', 
                        color='red', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                
                pred_text.append(f'{char}(ID:{int(class_id)},{conf:.2f})')
        
        ax2.axis('off')
        
        # Add prediction sequence at bottom
        pred_sequence = ' '.join(pred_text) if pred_text else 'No detections'
        ax2.text(0.5, -0.15, f'Pred Sequence: {pred_sequence}', 
                transform=ax2.transAxes, ha='center', va='top',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8))
        
        # Add accuracy indicator
        gt_chars = [self.idx_to_char.get(int(cat.item()), '?') for cat in gt_categories] if len(gt_categories) > 0 else []
        pred_chars = [self.idx_to_char.get(int(pred[5]), '?') for pred in predictions[0]] if len(predictions) > 0 and len(predictions[0]) > 0 else []
        
        gt_string = ''.join(gt_chars)
        pred_string = ''.join(pred_chars)
        is_correct = gt_string == pred_string
        
        accuracy_color = 'green' if is_correct else 'red'
        accuracy_text = 'CORRECT ✓' if is_correct else 'INCORRECT ✗'
        
        fig.suptitle(f'{accuracy_text} | GT: "{gt_string}" | Pred: "{pred_string}"', 
                    fontsize=18, fontweight='bold', color=accuracy_color,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'comparison_{idx+1:02d}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
        # Print summary for this image
        print(f"Image {idx + 1}: {accuracy_text}")
        print(f"  Ground Truth: {gt_string}")
        print(f"  Prediction: {pred_string}")
        print(f"  GT Count: {len(gt_chars)}, Pred Count: {len(pred_chars)}")
        print(f"  Saved: {save_path}")
        print("-" * 50)

    def evaluate_on_validation(self, val_data_dir, num_images=10, save_dir='./visual_evaluation_val'):
        """
        Create visual evaluation using validation dataset (which has labels.json)
        
        Args:
            val_data_dir: Path to validation dataset directory
            num_images: Number of images to evaluate
            save_dir: Directory to save visualizations
        """
        # Ensure the validation directory has labels.json
        labels_path = os.path.join(val_data_dir, 'labels.json')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.json not found in {val_data_dir}. Please use validation dataset.")
        
        print(f"Found labels.json in validation directory: {labels_path}")
        return self.visualize_comparison(val_data_dir, num_images, save_dir)

    def evaluate_on_test(self, test_data_dir, num_images=10, save_dir='./visual_evaluation_test'):
        """
        Create predictions-only visualization for test dataset (no ground truth)
        
        Args:
            test_data_dir: Path to test dataset directory
            num_images: Number of images to evaluate
            save_dir: Directory to save visualizations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create data loader for test set
        eval_loader = CaptchaDataLoader(
            test_data_dir,
            batch_size=1,
            shuffle=False
        )
        
        print(f"Creating predictions for {num_images} test images...")
        
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx >= num_images:
                break
                
            image = batch['Image'][0]  # Single image tensor (1, H, W)
            image_id = batch['ImageID'][0]
            
            # Make prediction
            image_tensor = image.unsqueeze(0).to(self.device)  # Add batch dimension
            predictions = self.predict(image_tensor)
            
            # Create prediction-only visualization
            self._create_prediction_plot(
                image, predictions, image_id, batch_idx, save_dir
            )

                        # Make prediction
            image_tensor = image.unsqueeze(0).to(self.device)
            predictions = self.predict(image_tensor)

            # ---- NEW: build & save JSON entry for this image ----
            json_entry = self._preds_to_json_entry(
                image_id=image_id,
                predictions=predictions,
                img_h=self.img_height,
                img_w=self.img_width
            )
            # Append and keep an always-up-to-date labels.json in save_dir
            json_out = os.path.join(save_dir, "labels.json")
            self._append_and_optionally_write_json(json_entry, out_path=json_out)

            
            print(f"Processed test image {batch_idx + 1}/{num_images}")
        
        print(f"Test predictions completed! Results saved to {save_dir}")

    def _create_prediction_plot(self, image, predictions, image_id, idx, save_dir):
        """Create prediction-only plot for test images (no ground truth)"""
        
        # Convert tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            image_np = image.squeeze().cpu().numpy()
        else:
            image_np = np.array(image)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        
        # Display image
        ax.imshow(image_np, cmap='gray')
        ax.set_title(f'Test Predictions - {image_id}', fontsize=16, fontweight='bold')
        
        # Draw prediction bounding boxes
        pred_text = []
        if len(predictions) > 0 and len(predictions[0]) > 0:
            for pred in predictions[0]:
                x, y, w, h, conf, class_id = pred
                char = self.idx_to_char.get(int(class_id), '?')
                
                # Draw bounding box
                rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                       edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # Add text with character, ID, and confidence
                ax.text(x, y-8, f'{char} (ID:{int(class_id)}) {conf:.2f}', 
                        color='red', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                
                pred_text.append(char)
        
        ax.axis('off')
        
        # Add prediction sequence at bottom
        pred_sequence = ''.join(pred_text) if pred_text else 'No detections'
        ax.text(0.5, -0.1, f'Predicted CAPTCHA: "{pred_sequence}"', 
                transform=ax.transAxes, ha='center', va='top',
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'test_prediction_{idx+1:02d}_{image_id}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
        # Print summary for this image
        print(f"Test Image {idx + 1} ({image_id}): Predicted '{pred_sequence}'")
        print(f"  Detected {len(pred_text)} characters")
        print(f"  Saved: {save_path}")
        print("-" * 50)

def create_summary_grid(save_dir, num_images):
    """Create a summary grid showing all comparisons"""
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    
    # Calculate grid dimensions
    cols = 2  # Always 2 columns for this layout
    rows = min(5, (num_images + 1) // 2)  # Max 5 rows to keep readable
    
    fig, axes = plt.subplots(rows, cols, figsize=(30, 6*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_images, rows * cols)):
        row = i // cols
        col = i % cols
        
        img_path = os.path.join(save_dir, f'comparison_{i+1:02d}.png')
        if os.path.exists(img_path):
            img = imread(img_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Image {i+1}', fontsize=14)
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Visual Evaluation Summary', fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    summary_path = os.path.join(save_dir, 'summary_grid.png')
    plt.savefig(summary_path, bbox_inches='tight', dpi=100, facecolor='white')
    plt.close()
    
    print(f"Summary grid saved to: {summary_path}")