"""
Loss functions for contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Standard InfoNCE Loss (NT-Xent Loss) for contrastive learning.
    This is the standard contrastive loss used in CLIP and SimCLR.

    Args:
        temperature: Temperature parameter to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features, batch_size=None):
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Infer batch size from features if not provided
        if batch_size is None:
            batch_size = image_features.shape[0]

        # Compute cosine similarity between image and text features
        # logits shape: [batch_size, batch_size]
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        # Create labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=image_features.device)

        # Compute cross entropy loss for both directions
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        # Average the two losses
        loss = (loss_i2t + loss_t2i) / 2

        return loss
