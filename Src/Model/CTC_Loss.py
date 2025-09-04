import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class ModelLoss(nn.Module):
    """
    CTC Loss wrapper for CAPTCHA sequence recognition.
    
    Uses PyTorch's built-in nn.CTCLoss, configured with:
    - blank index = last class (num_classes)
    - reduction = mean
    - zero_infinity = True (ignore infinite losses when targets are longer than inputs)
    """
    def __init__(self, Classes=36):
        super(ModelLoss, self).__init__()
        # +1 for the CTC blank symbol, which is always the last index
        self.BlankIndex = Classes  
        self.criterion = nn.CTCLoss(blank=self.BlankIndex, reduction='mean', zero_infinity=True)

    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        Compute CTC loss.

        Args:
            logits: (T, N, C) from ModelHead (time, batch, num_classes+1)
            targets: 1D concatenated tensor of target labels
                     (e.g., torch.tensor([c1, c2, ..., cM]) for all batch items)
            input_lengths: lengths of input sequences (list/tensor of size N)
            target_lengths: lengths of each target sequence (list/tensor of size N)

        Returns:
            loss: scalar tensor
        """
        # Convert logits to log-probabilities
        log_probs = logits.log_softmax(2)
        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
        return loss
