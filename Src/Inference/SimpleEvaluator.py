import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Src.Model.Model import CaptchaSolverModel
from Src.Data.DataLoader import CaptchaDataLoader
import json

class SimpleEvaluator:
    def __init__(self, model_path, num_classes=36, 
                 grid_height=10, grid_width=40,
                 device='cuda', 
                 default_conf_thresh=0.35):  # ✅ NEW: Add as parameter
        
        # ✅ DECLARE SINGLE CONFIDENCE THRESHOLD VARIABLE
        self.default_conf_thresh = default_conf_thresh
        self.default_iou_thresh = 0.3  # ✅ Also standardize IoU threshold
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.img_width = 640
        self.img_height = 160
        
        # Initialize model
        self.model = CaptchaSolverModel(
            num_classes=num_classes,
            grid_height=grid_height,
            grid_width=grid_width
        ).to(self.device)
        
        # Load trained weights
        self.load_model(model_path)
        
        # Character mapping
        self.idx_to_char = self._create_character_mapping()
        
        print(f"✅ SimpleEvaluator initialized with default thresholds:")
        print(f"   Confidence: {self.default_conf_thresh}")
        print(f"   IoU: {self.default_iou_thresh}")
    
    def _create_character_mapping(self):
        """Create mapping from class indices to characters"""
        # Digits 0-9
        chars = [str(i) for i in range(10)]
        # Uppercase letters A-Z
        chars.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        return {i: char for i, char in enumerate(chars)}
    
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
    
    def decode_predictions(self, raw_predictions, conf_thresh=None):  # ✅ CHANGED: Use None as default
        """Simple decoder for YOLO predictions"""
        if conf_thresh is None:
            conf_thresh = self.default_conf_thresh  # ✅ USE CLASS VARIABLE
            
        batch_size = raw_predictions.size(0)
        predictions = raw_predictions.view(batch_size, self.grid_height, self.grid_width, 5 + self.num_classes)
        
        batch_detections = []
        for b in range(batch_size):
            detections = []
            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    cell_pred = predictions[b, i, j]
                    x_rel, y_rel, w, h = cell_pred[:4]
                    objectness = torch.sigmoid(cell_pred[4])
                    class_scores = torch.softmax(cell_pred[5:], dim=0)
                    class_conf, class_id = torch.max(class_scores, dim=0)
                    
                    # Final confidence
                    final_conf = objectness * class_conf
                    
                    if final_conf > conf_thresh:
                        # Convert to absolute coordinates
                        center_x = (j + x_rel.item()) / self.grid_width * self.img_width
                        center_y = (i + y_rel.item()) / self.grid_height * self.img_height
                        width = w.item() * self.img_width
                        height = h.item() * self.img_height
                        
                        # Convert to corner format
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': final_conf.item(),
                            'class_id': class_id.item(),
                            'char': self.idx_to_char.get(class_id.item(), '?')
                        })
            
            # Sort by confidence
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            batch_detections.append(detections)
        
        return batch_detections
    
    def simple_nms(self, detections, iou_thresh=None):  # ✅ CHANGED: Use None as default
        """Simple Non-Maximum Suppression"""
        if iou_thresh is None:
            iou_thresh = self.default_iou_thresh  # ✅ USE CLASS VARIABLE
            
        if len(detections) == 0:
            return []
        
        def calculate_iou(box1, box2):
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        filtered_detections = []
        while detections:
            best = detections.pop(0)
            filtered_detections.append(best)
            
            # Remove overlapping detections
            detections = [
                det for det in detections
                if calculate_iou(best['bbox'], det['bbox']) < iou_thresh
            ]
        
        return filtered_detections
    
    def predict(self, image_tensor, conf_thresh=None, iou_thresh=None, sort_left_to_right=True):  # ✅ CHANGED
        """Make prediction on a single image tensor with optional sorting"""
        if conf_thresh is None:
            conf_thresh = self.default_conf_thresh  # ✅ USE CLASS VARIABLE
        if iou_thresh is None:
            iou_thresh = self.default_iou_thresh  # ✅ USE CLASS VARIABLE
            
        self.model.eval()
        
        with torch.no_grad():
            # Get model predictions
            raw_predictions = self.model(image_tensor)
            
            # Debug prints
            print(f"Raw predictions shape: {raw_predictions.shape}")
            print(f"Raw predictions range: [{raw_predictions.min().item():.3f}, {raw_predictions.max().item():.3f}]")
            
            # Decode predictions
            decoded_predictions = self.decode_predictions(raw_predictions, conf_thresh=conf_thresh)
            
            # Apply NMS
            final_predictions = []
            for batch_detections in decoded_predictions:
                nms_results = self.simple_nms(batch_detections, iou_thresh=iou_thresh)
                
                # ✅ SORT LEFT-TO-RIGHT if requested
                if sort_left_to_right and nms_results:
                    nms_results = self.sort_detections_left_to_right(nms_results)
                
                final_predictions.append(nms_results)
            
            return final_predictions
    
    def load_ground_truth(self, data_dir):
        """Load ground truth from labels.json"""
        labels_path = os.path.join(data_dir, 'labels.json')
        if not os.path.exists(labels_path):
            return {}
        
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        # Convert list format to dictionary format
        gt_dict = {}
        if isinstance(labels_data, list):
            # If labels.json is a list of items
            for item in labels_data:
                image_id = item.get('image_id', '')
                if image_id:
                    # Extract bboxes and labels from annotations
                    annotations = item.get('annotations', [])
                    bboxes = [ann.get('bbox', []) for ann in annotations]
                    labels = [ann.get('category_id', -1) for ann in annotations]
                    
                    gt_dict[image_id] = {
                        'bboxes': bboxes,
                        'labels': labels,
                        'captcha_string': item.get('captcha_string', '')
                    }
        elif isinstance(labels_data, dict):
            # If labels.json is already a dictionary
            gt_dict = labels_data
        
        return gt_dict

    def visualize_comparison(self, data_dir, num_images=10, save_dir='./visual_evaluation'):
        """Create visual comparison with ground truth"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create data loader
        eval_loader = CaptchaDataLoader(data_dir, batch_size=1, shuffle=False)
        
        print(f"Creating visual evaluation for {num_images} images...")
        print(f"Using thresholds: conf={self.default_conf_thresh}, iou={self.default_iou_thresh}")
        
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx >= num_images:
                break
            
            image = batch['Image'][0]  # Single image tensor
            image_id = batch['ImageID'][0]
            
            # Get ground truth from the batch
            gt_bboxes = batch['BoundingBoxes'][0].tolist() if len(batch['BoundingBoxes'][0]) > 0 else []
            gt_labels = batch['CategoryIDs'][0].tolist() if len(batch['CategoryIDs'][0]) > 0 else []
            
            # ✅ CHANGED: Use class variables (no need to specify)
            image_tensor = image.unsqueeze(0).to(self.device)
            predictions = self.predict(image_tensor, sort_left_to_right=True)
            
            # Create visualization
            self._create_comparison_plot(
                image, gt_bboxes, gt_labels, predictions[0], 
                image_id, batch_idx, save_dir
            )
            
            print(f"Processed image {batch_idx + 1}/{num_images}")
        
        print(f"Visual evaluation completed! Results saved to {save_dir}")
    
    def _create_comparison_plot(self, image, gt_bboxes, gt_labels, predictions, 
                              image_id, idx, save_dir):
        """Create side-by-side comparison plot with proper sorting"""
        
        # Convert tensor to numpy
        if isinstance(image, torch.Tensor):
            image_np = image.squeeze().cpu().numpy()
        else:
            image_np = np.array(image)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Ground Truth (Left)
        ax1.imshow(image_np, cmap='gray')
        ax1.set_title(f'Ground Truth - {image_id}', fontsize=16, fontweight='bold')
        
        # ✅ SORT GROUND TRUTH LEFT-TO-RIGHT
        gt_with_positions = []
        for bbox, label in zip(gt_bboxes, gt_labels):
            x, y, w, h = bbox
            char = self.idx_to_char.get(label, '?')
            gt_with_positions.append({
                'bbox': bbox,
                'char': char,
                'label': label,
                'x_left': x  # Left edge for sorting
            })
        
        # Sort ground truth by left edge
        gt_sorted = sorted(gt_with_positions, key=lambda x: x['x_left'])
        
        # Draw ground truth boxes in sorted order
        gt_text = []
        for i, gt_item in enumerate(gt_sorted):
            bbox = gt_item['bbox']
            char = gt_item['char']
            label = gt_item['label']
            x, y, w, h = bbox
            
            # Draw bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                   edgecolor='lime', facecolor='none')
            ax1.add_patch(rect)
            
            # Add text with position number
            ax1.text(x, y-8, f'{char} ({i+1})', 
                    color='lime', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            gt_text.append(char)
        
        ax1.axis('off')
        
        # Ground truth sequence (sorted)
        gt_sequence = ''.join(gt_text) if gt_text else 'No labels'
        ax1.text(0.5, -0.15, f'GT Sequence (L→R): {gt_sequence}', 
                transform=ax1.transAxes, ha='center', va='top',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        
        # Predictions (Right) - Already sorted by predict method
        ax2.imshow(image_np, cmap='gray')
        ax2.set_title(f'Predictions - {image_id}', fontsize=16, fontweight='bold')
        
        # Draw prediction boxes (should already be sorted)
        pred_text = []
        for i, pred in enumerate(predictions):
            x1, y1, x2, y2 = pred['bbox']
            w = x2 - x1
            h = y2 - y1
            char = pred['char']
            conf = pred['confidence']
            class_id = pred['class_id']
            
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), w, h, linewidth=3, 
                                   edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
            
            # Add text with position number
            ax2.text(x1, y1-8, f'{char} ({i+1}) {conf:.2f}', 
                    color='red', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            pred_text.append(char)
        
        ax2.axis('off')
        
        # Prediction sequence (sorted)
        pred_sequence = ''.join(pred_text) if pred_text else 'No detections'
        ax2.text(0.5, -0.15, f'Pred Sequence (L→R): {pred_sequence}', 
                transform=ax2.transAxes, ha='center', va='top',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8))
        
        # Accuracy indicator
        is_correct = gt_sequence == pred_sequence
        accuracy_color = 'green' if is_correct else 'red'
        accuracy_text = 'CORRECT ✓' if is_correct else 'INCORRECT ✗'
        
        fig.suptitle(f'{accuracy_text} | GT: "{gt_sequence}" | Pred: "{pred_sequence}"', 
                    fontsize=18, fontweight='bold', color=accuracy_color,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(save_dir, f'comparison_{idx+1:02d}_{image_id}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
        # Print summary - ✅ FIXED: Simplified the f-string formatting
        print(f"Image {idx + 1} ({image_id}): {accuracy_text}")
        print(f"  Ground Truth (sorted): {gt_sequence}")
        print(f"  Prediction (sorted): {pred_sequence}")
        print(f"  GT Count: {len(gt_text)}, Pred Count: {len(pred_text)}")
        
        # ✅ FIXED: Simplified position printing to avoid nested f-string issues
        gt_positions = [f"{char}@{int(item['x_left'])}" for char, item in zip(gt_text, gt_sorted)]
        pred_positions = [f"{char}@{int(pred['bbox'][0])}" for char, pred in zip(pred_text, predictions)]
        
        print(f"  Character positions GT: {gt_positions}")
        print(f"  Character positions Pred: {pred_positions}")
        print(f"  Saved: {save_path}")
        print("-" * 50)
    
    def debug_model_output(self, image_tensor):
        """Debug what the model is actually outputting"""
        self.model.eval()
        
        with torch.no_grad():
            # Get raw model output
            raw_predictions = self.model(image_tensor)
            
            print("=== MODEL OUTPUT DEBUG ===")
            print(f"Input shape: {image_tensor.shape}")
            print(f"Raw output shape: {raw_predictions.shape}")
            # FIX: Use correct grid dimensions (40x160 not 10x40)
            expected_size = self.grid_height * self.grid_width * (5 + self.num_classes)
            expected_shape = f"{image_tensor.shape[0]} × {expected_size}"
            grid_info = f"{self.grid_height}×{self.grid_width}×{5+self.num_classes}"
            print(f"Expected shape: {expected_shape} = {image_tensor.shape[0]} × {grid_info}")
            
            # Check output statistics
            print(f"\nRaw output statistics:")
            print(f"  Min: {raw_predictions.min().item():.6f}")
            print(f"  Max: {raw_predictions.max().item():.6f}")
            print(f"  Mean: {raw_predictions.mean().item():.6f}")
            print(f"  Std: {raw_predictions.std().item():.6f}")
            
            # Reshape and analyze
            try:
                batch_size = raw_predictions.size(0)
                # FIX: Use correct grid dimensions
                reshaped = raw_predictions.view(batch_size, self.grid_height, self.grid_width, 5 + self.num_classes)
                print(f"Reshape successful: {reshaped.shape}")
                
                # Check objectness scores
                objectness_raw = reshaped[0, :, :, 4]  # Raw objectness
                objectness_sigmoid = torch.sigmoid(objectness_raw)
                
                print(f"\nObjectness analysis:")
                print(f"  Raw objectness - Min: {objectness_raw.min():.6f}, Max: {objectness_raw.max():.6f}")
                print(f"  After sigmoid - Min: {objectness_sigmoid.min():.6f}, Max: {objectness_sigmoid.max():.6f}")
                print(f"  Mean sigmoid: {objectness_sigmoid.mean():.6f}")
                
                # Check how many cells have reasonable objectness
                thresholds = [0.001, 0.01, 0.1, 0.3, 0.5]
                for thresh in thresholds:
                    count = (objectness_sigmoid > thresh).sum().item()
                    total_cells = self.grid_height * self.grid_width
                    # ✅ FIXED: Simplified f-string to avoid nesting issues
                    print(f"  Cells with objectness > {thresh}: {count}/{total_cells}")
                
                # Check coordinate predictions
                coords = reshaped[0, :, :, :4]
                print(f"\nCoordinate predictions:")
                print(f"  X coords - Min: {coords[:,:,0].min():.3f}, Max: {coords[:,:,0].max():.3f}")
                print(f"  Y coords - Min: {coords[:,:,1].min():.3f}, Max: {coords[:,:,1].max():.3f}")
                print(f"  W coords - Min: {coords[:,:,2].min():.3f}, Max: {coords[:,:,2].max():.3f}")
                print(f"  H coords - Min: {coords[:,:,3].min():.3f}, Max: {coords[:,:,3].max():.3f}")
                
                # Check class predictions
                class_raw = reshaped[0, :, :, 5:]
                class_softmax = torch.softmax(class_raw, dim=-1)
                class_max_prob, class_max_id = torch.max(class_softmax, dim=-1)
                
                print(f"\nClass predictions:")
                print(f"  Max class probability: {class_max_prob.max():.6f}")
                print(f"  Mean class probability: {class_max_prob.mean():.6f}")
                print(f"  Predicted classes range: {class_max_id.min().item()} to {class_max_id.max().item()}")
                
            except Exception as e:
                print(f"Error reshaping: {e}")
                print("This suggests wrong grid size or output format!")

    def sort_detections_left_to_right(self, detections):
        """
        Sort detections from left to right based on x-coordinate of bounding box center
        
        Args:
            detections: List of detection dictionaries with 'bbox' key
        
        Returns:
            List of detections sorted left-to-right
        """
        if not detections:
            return detections
        
        # Sort by left edge of bounding box for more robust sorting
        def get_left_edge(det):
            x1, y1, x2, y2 = det['bbox']
            return x1  # Left edge of bounding box
        
        sorted_detections = sorted(detections, key=get_left_edge)
        
        return sorted_detections

    def extract_captcha_string(self, image_tensor, conf_thresh=None, iou_thresh=None):  # ✅ CHANGED
        """Extract CAPTCHA string with characters arranged left-to-right"""
        if conf_thresh is None:
            conf_thresh = self.default_conf_thresh  # ✅ USE CLASS VARIABLE
        if iou_thresh is None:
            iou_thresh = self.default_iou_thresh  # ✅ USE CLASS VARIABLE
            
        self.model.eval()
        
        with torch.no_grad():
            # Get model predictions
            raw_predictions = self.model(image_tensor)
            
            # Decode predictions
            decoded_predictions = self.decode_predictions(raw_predictions, conf_thresh=conf_thresh)
            
            # Apply NMS
            if not decoded_predictions or not decoded_predictions[0]:
                return "", []
            
            nms_results = self.simple_nms(decoded_predictions[0], iou_thresh=iou_thresh)
            
            if not nms_results:
                return "", []
            
            # ✅ SORT DETECTIONS LEFT-TO-RIGHT
            sorted_detections = self.sort_detections_left_to_right(nms_results)
            
            # Extract character string
            captcha_string = "".join([det['char'] for det in sorted_detections])
            
            return captcha_string, sorted_detections

    def predict_with_sorted_string(self, image_tensor, conf_thresh=None, iou_thresh=None):  # ✅ CHANGED
        """Get both detections and the ordered CAPTCHA string"""
        if conf_thresh is None:
            conf_thresh = self.default_conf_thresh  # ✅ USE CLASS VARIABLE
        if iou_thresh is None:
            iou_thresh = self.default_iou_thresh  # ✅ USE CLASS VARIABLE
            
        captcha_string, sorted_detections = self.extract_captcha_string(
            image_tensor, conf_thresh, iou_thresh
        )
        
        return sorted_detections, captcha_string

    def predict_captcha_batch(self, image_tensors, conf_thresh=None, iou_thresh=None):  # ✅ CHANGED
        """Process a batch of images for CAPTCHA prediction with consistent thresholds"""
        if conf_thresh is None:
            conf_thresh = self.default_conf_thresh  # ✅ USE CLASS VARIABLE
        if iou_thresh is None:
            iou_thresh = self.default_iou_thresh  # ✅ USE CLASS VARIABLE
            
        results = []
        
        for i in range(image_tensors.size(0)):
            single_image = image_tensors[i:i+1]  # Keep batch dimension
            captcha_string, detection_info = self.extract_captcha_string(
                single_image, conf_thresh, iou_thresh
            )
            results.append((captcha_string, detection_info))
        
        return results

    # ✅ NEW: Method to change thresholds at runtime
    def set_thresholds(self, conf_thresh=None, iou_thresh=None):
        """Update default thresholds"""
        if conf_thresh is not None:
            self.default_conf_thresh = conf_thresh
            print(f"✅ Updated confidence threshold to: {self.default_conf_thresh}")
        
        if iou_thresh is not None:
            self.default_iou_thresh = iou_thresh
            print(f"✅ Updated IoU threshold to: {self.default_iou_thresh}")
    
    # ✅ NEW: Method to get current thresholds
    def get_thresholds(self):
        """Get current threshold settings"""
        return {
            'confidence': self.default_conf_thresh,
            'iou': self.default_iou_thresh
        }

    # ... [rest of the methods remain the same] ...