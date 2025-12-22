"""
Segmentation loss for VFMKD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List


class SegmentationLoss(nn.Module):
    """
    Segmentation loss combining dice loss and cross-entropy loss.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize segmentation loss.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.num_classes = config.get('num_classes', 1)
        self.dice_weight = config.get('dice_weight', 1.0)
        self.ce_weight = config.get('ce_weight', 1.0)
        self.smooth = config.get('smooth', 1e-6)
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute dice loss.
        
        Args:
            pred: Predicted segmentation logits
            target: Ground truth segmentation masks
            
        Returns:
            Dice loss
        """
        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation loss.
        
        Args:
            predictions: Dictionary containing model predictions
            targets: Dictionary containing ground truth targets
            
        Returns:
            Dictionary containing loss components
        """
        # Extract predictions and targets
        pred_masks = predictions.get('masks', None)
        target_masks = targets.get('masks', None)
        
        if pred_masks is None or target_masks is None:
            return {'total_loss': torch.tensor(0.0, device=next(self.parameters()).device)}
        
        losses = {}
        total_loss = 0.0
        
        # Dice loss
        dice_loss = self.dice_loss(pred_masks, target_masks)
        losses['dice_loss'] = dice_loss
        total_loss += self.dice_weight * dice_loss
        
        # Cross-entropy loss
        ce_loss = self.ce_loss(pred_masks, target_masks.long())
        losses['ce_loss'] = ce_loss
        total_loss += self.ce_weight * ce_loss
        
        losses['total_loss'] = total_loss
        return losses