import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Src.Model.CTC_Model import CaptchaSolverModel
from Src.Data.DataLoader import CaptchaDataLoader
from Src.Utils.TargetPreparer import TargetPreparer
from Src.Utils.Decoder import decode_yolo_output
from Src.Utils.NMS import ApplyNMS
import json

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
        self._create_character_mapping(blank_idx=36)   # adjust if your training used a different blank    # --- ADD near the top of the file ---

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

        # --- REPLACE the whole predict() ---
    def predict(self, image_tensor):
        """
        image_tensor: (B,1,H,W)
        returns list[str]
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image_tensor)  # (T,B,C) or (B,T,C)
            seqs = ctc_greedy_decode(logits, blank=self.blank_idx)  # list[list[int]]
            def map_seq(ids):
                return "".join(self.idx_to_char.get(k, "?") for k in ids)
            return [map_seq(s) for s in seqs]

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
                # --- ADD in visualize_comparison loop, after loading batch ---
                # Try to extract GT string from loader (common keys)
                gt_string = batch.get('CaptchaString', [''])[0] if isinstance(batch.get('CaptchaString'), list) \
                            else batch.get('captcha_string', '')
                self._last_gt_string = gt_string  # pass implicitly

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


                # --- CHANGE in visualize_comparison ---
                image_tensor = image.unsqueeze(0).to(self.device)
                pred_strings = self.predict(image_tensor)     # list[str]
                ...
                self._create_side_by_side_plot(
                    image, None, batch.get('CategoryIDs',[torch.empty(0)])[0], pred_strings,
                    image_path, batch_idx, save_dir
                )

                print(f"Processed image {batch_idx + 1}/{num_images}")

            print(f"Visual evaluation completed! Results saved to {save_dir}")
    
    def _create_side_by_side_plot(self, image, _gt_bboxes, gt_categories, predictions, 
                                  image_path, idx, save_dir):
        pred_string = predictions[0] if predictions else ""
        # prefer explicit GT string captured earlier
        if isinstance(gt_categories, torch.Tensor) and gt_categories.numel() > 0:
            gt_chars = [self.idx_to_char.get(int(c.item()), '?') for c in gt_categories]
            gt_string = ''.join(gt_chars)
        else:
            gt_string = getattr(self, "_last_gt_string", "")

        image_np = image.squeeze().cpu().numpy() if isinstance(image, torch.Tensor) else np.array(image)
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.imshow(image_np, cmap='gray'); ax.axis('off')

        ok = (gt_string == pred_string)
        title = f'GT: "{gt_string}" | Pred: "{pred_string}"'
        ax.set_title(title, fontsize=16, color=('green' if ok else 'red'), fontweight='bold')

        save_path = os.path.join(save_dir, f'comparison_{idx+1:02d}.png')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()

        print(f"Image {idx + 1}: {'CORRECT ✓' if ok else 'INCORRECT ✗'}")
        print(f"  GT:   {gt_string}")
        print(f"  Pred: {pred_string}")
        print(f"  Saved: {save_path}")
        print("-"*50)


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
        image_np = image.squeeze().cpu().numpy() if isinstance(image, torch.Tensor) else np.array(image)
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        ax.imshow(image_np, cmap='gray'); ax.axis('off')

        pred_string = predictions[0] if predictions else ""
        ax.set_title(f'Test Prediction - {image_id}\n"{pred_string}"',
                     fontsize=16, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'test_prediction_{idx+1:02d}_{image_id}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()

        print(f"Test Image {idx + 1} ({image_id}): Predicted '{pred_string}'")
        print(f"  Saved: {save_path}")
        print("-" * 50)
        
                # --- ADD method ---
    @torch.no_grad()
    def validate_ctc(self, val_dir, max_batches=None):
        loader = CaptchaDataLoader(val_dir, batch_size=1, shuffle=False)
        preds, gts = [], []
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches: break
            image = batch['Image'][0]
            gt = batch.get('CaptchaString', [''])[0] if isinstance(batch.get('CaptchaString'), list) \
                 else batch.get('captcha_string','')
            p = self.predict(image.unsqueeze(0).to(self.device))[0]
            preds.append(p); gts.append(gt)
        score = ler(preds, gts)
        print(f"Validation LER (CTC): {score:.4f}")
        return score


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
    
    