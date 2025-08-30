import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelLoss(nn.Module):
    """YOLOv8-style loss: Smooth L1 + Binary CE + Cross Entropy"""

    def __init__(self, num_classes=36, GridHeight=10, GridWidth=40, LambdaBoundingBox=5.0, LambdaObjectness=1.0, LambdaClassification=1.0):
        super(ModelLoss, self).__init__()
        self.num_classes = num_classes
        self.GridHeight = GridHeight  # Rectangular grid height
        self.GridWidth = GridWidth    # Rectangular grid width
        self.LambdaBoundingBox = LambdaBoundingBox
        self.LambdaObjectness = LambdaObjectness
        self.LambdaClassification = LambdaClassification
    
    def forward(self, Predictions, GroundTruth):
        """
        Predictions: (batch_size, GridHeight*grid_width*(5 + num_classes)) - [x,y,w,h,obj,classes...]
        GroundTruth: (batch_size, GridHeight, grid_width, 5 + num_classes) - [x,y,w,h,conf,classes...]
        """
        BatchSize = Predictions.size(0)
        
        # Reshape Predictions to match GroundTruth
        Predictions = Predictions.view(BatchSize, self.GridHeight, self.GridWidth, 5 + self.num_classes)
        
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
        
        #Objectness Loss (Binary CE)
        obj_loss = 0
        if obj_mask.sum() > 0:
            obj_loss = F.binary_cross_entropy_with_logits(
                ObjectnessPredictions[obj_mask], GroundTruthObjectness[obj_mask], reduction='sum'
            )
        
        noobj_loss = 0
        if noobj_mask.sum() > 0:
            noobj_loss = F.binary_cross_entropy_with_logits(
                ObjectnessPredictions[noobj_mask], torch.zeros_like(ObjectnessPredictions[noobj_mask]), reduction='sum'
            )
        
        total_obj_loss = obj_loss + noobj_loss
        
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
            'num_pos': obj_mask.sum().item(),
            'num_neg': noobj_mask.sum().item()
        }