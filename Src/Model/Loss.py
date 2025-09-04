import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in objectness detection.
    Focuses learning on hard negative examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # Calculate binary cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Calculate pt (probability of correct class)
        pt = torch.exp(-ce_loss)
        
        #Apply focal loss formula: α(1-pt)^γ * CE
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.sum()

class ModelLoss(nn.Module):
    """YOLOv8-style loss: Smooth L1 + Focal Loss + Cross Entropy"""

    def __init__(self, NumClasses=36, GridHeight=10, GridWidth=40, 
                 LambdaBoundingBox=5.0, LambdaObjectness=0.1, LambdaClassification=2.0):
        super(ModelLoss, self).__init__()
        self.NumClasses = NumClasses
        self.GridHeight = GridHeight
        self.GridWidth = GridWidth
        self.LambdaBoundingBox = LambdaBoundingBox
        self.LambdaObjectness = LambdaObjectness
        self.LambdaClassification = LambdaClassification
        
        # Initialize Focal Loss
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    def forward(self, Predictions, GroundTruth):
        
        BatchSize = Predictions.size(0)
        
        #Reshape ground truth to (BatchSize, GridHeight, GridWidth, 5 + num of classes)
        GroundTruth = GroundTruth.view(BatchSize, self.GridHeight, self.GridWidth, 5 + self.NumClasses)
        
        # Split into components
        BoundingBoxPredictions = Predictions[..., :4]    # bounding boxes
        ObjectnessPredictions = Predictions[..., 4:5]    # objectness
        ClassPredictions = Predictions[..., 5:]   # classes
        
        GroundTruthBoundingBox = GroundTruth[..., :4]
        GroundTruthObjectness = GroundTruth[..., 4:5]
        GroundTruthClass = GroundTruth[..., 5:]
        
        # Create masks
        obj_mask = (GroundTruthObjectness > 0)         # cells with objects
        noobj_mask = (GroundTruthObjectness == 0)      # cells without objects
        
        #Bounding Box Loss (Smooth L1) 
        bbox_loss = 0
        if obj_mask.sum() > 0:
            bbox_loss = F.smooth_l1_loss(
                BoundingBoxPredictions[obj_mask.expand_as(BoundingBoxPredictions)],
                GroundTruthBoundingBox[obj_mask.expand_as(GroundTruthBoundingBox)],
                reduction='sum'
            )
        
        #Balanced Sampling with reshape by removing last dimension
        combined_targets = GroundTruthObjectness.squeeze(-1)  
        combined_preds = ObjectnessPredictions.squeeze(-1)    
        flat_targets = combined_targets.reshape(-1)
        flat_preds = combined_preds.reshape(-1)
        
        # Get positive and negative masks
        pos_mask = (flat_targets > 0)
        neg_mask = (flat_targets == 0)
        
        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()
        
        # Apply balanced sampling (1:8 ratio - 1 positive : 8 negatives)
        if num_pos > 0:
            # Limit negatives to 8x positives
            max_negatives = min(num_pos * 8, num_neg)
            
            if num_neg > max_negatives:
                # Randomly sample negatives
                neg_indices = torch.where(neg_mask)[0]
                
                # Random sampling of negatives
                perm = torch.randperm(len(neg_indices), device=neg_indices.device)
                selected_neg_idx = neg_indices[perm[:max_negatives]]
                
                # Create balanced negative mask
                balanced_neg_mask = torch.zeros_like(neg_mask)
                balanced_neg_mask[selected_neg_idx] = True
            else:
                balanced_neg_mask = neg_mask
            
            # Combine positive and balanced negative masks
            balanced_mask = pos_mask | balanced_neg_mask
        else:
            # If no positives, use all negatives (shouldn't happen in practice)
            balanced_mask = neg_mask
        
        # Apply focal loss only to balanced samples
        if balanced_mask.sum() > 0:
            total_obj_loss = self.focal_loss(
                flat_preds[balanced_mask],
                flat_targets[balanced_mask]
            )
        else:
            total_obj_loss = torch.tensor(0.0, device=flat_preds.device, requires_grad=True)
        
        #Classification Loss (Cross Entropy)
        class_loss = 0
        if obj_mask.sum() > 0:
            GroundTruthClass_indices = torch.argmax(GroundTruthClass, dim=-1)
            obj_mask_flat = obj_mask.squeeze(-1)
            
            pred_class_obj = ClassPredictions[obj_mask_flat]
            GroundTruthClass_obj = GroundTruthClass_indices[obj_mask_flat]
            
            if pred_class_obj.numel() > 0:
                class_loss = F.cross_entropy(pred_class_obj, GroundTruthClass_obj, reduction='sum')
        
        # Combine losses
        total_loss = (
            self.LambdaBoundingBox * bbox_loss +
            self.LambdaObjectness * total_obj_loss +
            self.LambdaClassification * class_loss
        )
        
        # Normalize by batch size
        total_loss = total_loss / BatchSize
        bbox_loss = bbox_loss / BatchSize
        total_obj_loss = total_obj_loss / BatchSize
        class_loss = class_loss / BatchSize
        
        return {
            'total_loss': total_loss,
            'bbox_loss': bbox_loss,
            'obj_loss': total_obj_loss,
            'class_loss': class_loss,
            'num_pos': num_pos,
            'num_neg': balanced_mask.sum().item() - num_pos,  
            'total_samples_used': balanced_mask.sum().item()
        }