import torch

class TargetPreparer:
    """
    Converts raw dataset annotations into YOLO target format.
    """
    def __init__(self, GridHeight=20, GridWidth=80, NumClasses=36, ImageWidth=640, ImageHeight=160):
        self.GridHeight = GridHeight
        self.GridWidth = GridWidth
        self.NumClasses = NumClasses
        self.ImageWidth = ImageWidth
        self.ImageHeight = ImageHeight

    def __call__(self, batch):
        """
        Converts a batch of raw annotations into YOLO target format.

        Args:
            batch (dict): A batch containing:
                - 'Image': Tensor of images.
                - 'BoundingBoxes': List of tensors with bounding boxes for each image.
                - 'CategoryIDs': List of tensors with category IDs for each image.

        Returns:
            Tensor: YOLO target format (batch_size, GridHeight, GridWidth, 5 + NumClasses).
        """
        batch_size = len(batch['BoundingBoxes'])
        targets = torch.zeros(batch_size, self.GridHeight, self.GridWidth, 5 + self.NumClasses)

        for b in range(batch_size):
            bboxes = batch['BoundingBoxes'][b]  # Tensor of shape (N, 4)
            category_ids = batch['CategoryIDs'][b]  # Tensor of shape (N,)
            
            if len(bboxes) == 0:
                continue

            for i in range(len(bboxes)):
                # Extract bounding box and class information
                x, y, w, h = bboxes[i].tolist()  # [x, y, width, height] format
                class_id = int(category_ids[i].item())
                
                # Convert to center coordinates
                center_x = x + w / 2.0
                center_y = y + h / 2.0
                
                # Normalize to [0, 1]
                center_x_norm = center_x / self.ImageWidth
                center_y_norm = center_y / self.ImageHeight
                width_norm = w / self.ImageWidth
                height_norm = h / self.ImageHeight

                # Find grid cell
                grid_x = min(int(center_x_norm * self.GridWidth), self.GridWidth - 1)
                grid_y = min(int(center_y_norm * self.GridHeight), self.GridHeight - 1)

                # Relative position within grid cell
                rel_x = center_x_norm * self.GridWidth - grid_x
                rel_y = center_y_norm * self.GridHeight - grid_y

                # Set target values
                targets[b, grid_y, grid_x, 0] = rel_x
                targets[b, grid_y, grid_x, 1] = rel_y
                targets[b, grid_y, grid_x, 2] = width_norm
                targets[b, grid_y, grid_x, 3] = height_norm
                targets[b, grid_y, grid_x, 4] = 1.0  # Confidence

                # One-hot encode class label
                if 0 <= class_id < self.NumClasses:
                    targets[b, grid_y, grid_x, 5 + class_id] = 1.0

        return targets