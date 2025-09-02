import torch

class TargetPreparer:
    """
    Converts raw dataset annotations into YOLO target format.
    """
    def __init__(self, GridHeight=40, GridWidth=160, num_classes=36, img_width=640, img_height=160):
        # Change from 20×80 to 10×40 to match your model
        self.GridHeight = GridHeight
        self.GridWidth = GridWidth
        self.num_classes = num_classes
        self.img_width = img_width
        self.img_height = img_height

    def __call__(self, batch):
        """
        Converts a batch of raw annotations into YOLO target format.

        Args:
            batch (dict): A batch containing:
                - 'Image': Tensor of images.
                - 'BoundingBoxes': List of tensors with bounding boxes for each image.
                - 'CategoryIDs': List of tensors with category IDs for each image.

        Returns:
            Tensor: YOLO target format (batch_size, GridHeight, GridWidth, 5 + num_classes).
        """
        batch_size = len(batch['BoundingBoxes'])
        targets = torch.zeros(batch_size, self.GridHeight, self.GridWidth, 5 + self.num_classes)

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
                center_x_norm = center_x / self.img_width
                center_y_norm = center_y / self.img_height
                width_norm = w / self.img_width
                height_norm = h / self.img_height

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
                if 0 <= class_id < self.num_classes:
                    targets[b, grid_y, grid_x, 5 + class_id] = 1.0

        #  Use view() for flattening
        batch_size = targets.size(0)
        return targets.view(batch_size, -1)  # Shape: (batch_size, 40*160*41)