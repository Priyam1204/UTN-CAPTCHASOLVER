import torch

class PredictionDecoder:
    def __init__(self, GridHeight, GridWidth, ImageWidth, ImageHeight, NumberofClasses):
        self.GridHeight = GridHeight
        self.GridWidth = GridWidth
        self.ImageWidth = ImageWidth
        self.ImageHeight = ImageHeight
        self.NumberofClasses = NumberofClasses
    
    def Decode(self, RawPredictions, ConfidenceThreshold, ID2Character):
        """
        Decode YOLO Predictions into detection objects
        
        Args:
            RawPredictions: Raw model output tensor
            ConfidenceThreshold: Confidence threshold for filtering
            ID2Character: Dictionary mapping class indices to characters
            
        Returns:
            List of batch Detections
        """
        BatchSize = RawPredictions.size(0)
        Predictions = RawPredictions.view(BatchSize, self.GridHeight, self.GridWidth, 5 + self.NumberofClasses)
        
        BatchDetections = []
        for b in range(BatchSize):
            Detections = []
            for i in range(self.GridHeight):
                for j in range(self.GridWidth):
                    CellPrediction = Predictions[b, i, j]
                    x_rel, y_rel, w, h = CellPrediction[:4]
                    Objectness = torch.sigmoid(CellPrediction[4])
                    ClassScores = torch.softmax(CellPrediction[5:], dim=0)
                    ClassConfidence, ClassID = torch.max(ClassScores, dim=0)
                    
                    final_conf = Objectness * ClassConfidence
                    
                    if final_conf > ConfidenceThreshold:
                        center_x = (j + x_rel.item()) / self.GridWidth * self.ImageWidth
                        center_y = (i + y_rel.item()) / self.GridHeight * self.ImageHeight
                        width = w.item() * self.ImageWidth
                        height = h.item() * self.ImageHeight
                        
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2
                        
                        Detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': final_conf.item(),
                            'class_id': ClassID.item(),
                            'char': ID2Character.get(ClassID.item(), '?')
                        })
            
            Detections = sorted(Detections, key=lambda x: x['confidence'], reverse=True)
            BatchDetections.append(Detections)
        
        return BatchDetections



    def ConfidenceFilter(self, Detections, ConfidenceThreshold):
        """
        Filter existing Detections by confidence threshold
        
        Args:
            Detections: List of detection dictionaries
            ConfidenceThreshold: Confidence threshold
            
        Returns:
            Filtered list of Detections
        """
        return [det for det in Detections if det['confidence'] >= ConfidenceThreshold]