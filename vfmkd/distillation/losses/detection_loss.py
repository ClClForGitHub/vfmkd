"""
Detection loss for VFMKD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List


class DetectionLoss(nn.Module):
    """
    Detection loss combining classification and regression losses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize detection loss.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.num_classes = config.get('num_classes', 80)
        self.box_loss_weight = config.get('box_loss_weight', 1.0)
        self.cls_loss_weight = config.get('cls_loss_weight', 1.0)
        self.obj_loss_weight = config.get('obj_loss_weight', 1.0)
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.
        
        Args:
            predictions: Dictionary containing model predictions
            targets: Dictionary containing ground truth targets
            
        Returns:
            Dictionary containing loss components
        """
        # Extract predictions
        pred_boxes = predictions.get('boxes', None)
        pred_cls = predictions.get('cls', None)
        pred_obj = predictions.get('obj', None)
        
        # Extract targets
        target_boxes = targets.get('boxes', None)
        target_cls = targets.get('cls', None)
        target_obj = targets.get('obj', None)
        
        losses = {}
        total_loss = 0.0
        
        # Box regression loss
        if pred_boxes is not None and target_boxes is not None:
            box_loss = self.mse_loss(pred_boxes, target_boxes)
            losses['box_loss'] = box_loss
            total_loss += self.box_loss_weight * box_loss
        
        # Classification loss
        if pred_cls is not None and target_cls is not None:
            cls_loss = self.bce_loss(pred_cls, target_cls)
            losses['cls_loss'] = cls_loss
            total_loss += self.cls_loss_weight * cls_loss
        
        # Objectness loss
        if pred_obj is not None and target_obj is not None:
            obj_loss = self.bce_loss(pred_obj, target_obj)
            losses['obj_loss'] = obj_loss
            total_loss += self.obj_loss_weight * obj_loss
        
        losses['total_loss'] = total_loss
        return losses