"""
Feature distillation loss for VFMKD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union


class FeatureLoss(nn.Module):
    """
    Feature distillation loss for knowledge distillation.
    
    Supports multiple loss types: MSE, KL divergence, cosine similarity, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature loss.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.loss_type = config.get('loss_type', 'mse')
        self.temperature = config.get('temperature', 1.0)
        self.alpha = config.get('alpha', 1.0) # Weight for distillation loss
        
        self.alignment_layers = nn.ModuleList() # For channel alignment
        
        print(f"Initialized FeatureLoss with type: {self.loss_type}, temperature: {self.temperature}, alpha: {self.alpha}")

    def _align_features(self, student_features: List[torch.Tensor], teacher_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Align student features to match teacher features in terms of channels and spatial dimensions.
        
        Args:
            student_features: List of student feature tensors
            teacher_features: List of teacher feature tensors
        
        Returns:
            List of aligned student feature tensors
        """
        aligned_student_features = []
        
        # Assuming student_features is a single tensor and teacher_features is a list
        # We need to align the single student feature to the last teacher feature
        if not isinstance(student_features, list):
            student_features = [student_features]
        
        if len(student_features) == 1 and len(teacher_features) > 1:
            # Align the single student feature to the last teacher feature
            student_feat = student_features[0]
            teacher_feat = teacher_features[-1] # Align to the deepest teacher feature
            
            student_dim = student_feat.size(1)
            teacher_dim = teacher_feat.size(1)
            
            # Spatial alignment (upsample/downsample if needed)
            if student_feat.shape[2:] != teacher_feat.shape[2:]:
                student_feat = F.interpolate(student_feat, size=teacher_feat.shape[2:], mode='bilinear', align_corners=False)
            
            # Channel alignment (1x1 convolution)
            if student_dim != teacher_dim:
                # Check if an alignment layer already exists for this dimension pair
                # For simplicity, we'll create a new one each time for now, or manage a dict of layers
                # For a single student feature, we only need one alignment layer
                if not self.alignment_layers:
                    self.alignment_layers.append(nn.Conv2d(student_dim, teacher_dim, kernel_size=1, bias=False))
                    self.alignment_layers[-1].to(student_feat.device) # Move to device
                
                student_feat = self.alignment_layers[0](student_feat)
            
            aligned_student_features.append(student_feat)
            
        elif len(student_features) == len(teacher_features):
            # If both are multi-scale, align pair-wise
            for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
                student_dim = s_feat.size(1)
                teacher_dim = t_feat.size(1)
                
                # Spatial alignment
                if s_feat.shape[2:] != t_feat.shape[2:]:
                    s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
                
                # Channel alignment
                if student_dim != teacher_dim:
                    if len(self.alignment_layers) <= i:
                        self.alignment_layers.append(nn.Conv2d(student_dim, teacher_dim, kernel_size=1, bias=False))
                        self.alignment_layers[-1].to(s_feat.device)
                    s_feat = self.alignment_layers[i](s_feat)
                
                aligned_student_features.append(s_feat)
        else:
            raise ValueError("Feature lists must be either single student to multi-teacher, or matching lengths.")
            
        return aligned_student_features

    def forward(self, student_features: Union[torch.Tensor, List[torch.Tensor]], 
                teacher_features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Compute the distillation loss.
        
        Args:
            student_features: Student feature tensor or list of tensors
            teacher_features: Teacher feature tensor or list of tensors
        
        Returns:
            Computed distillation loss
        """
        
        # Ensure features are lists for consistent processing
        if not isinstance(student_features, list):
            student_features = [student_features]
        if not isinstance(teacher_features, list):
            teacher_features = [teacher_features]
        
        # Align student features to match teacher features
        # For simple distillation, we align the single student output to the last teacher output
        aligned_student_features = self._align_features(student_features, teacher_features)
        
        # For simple distillation, we only consider the last teacher feature
        # and the (now aligned) single student feature
        if len(aligned_student_features) == 1 and len(teacher_features) > 1:
            teacher_features_to_use = [teacher_features[-1]]
        else:
            teacher_features_to_use = teacher_features
        
        total_loss = 0.0
        for s_feat, t_feat in zip(aligned_student_features, teacher_features_to_use):
            if self.loss_type == 'mse':
                loss = F.mse_loss(s_feat, t_feat)
            elif self.loss_type == 'kl_div':
                # KL divergence for logits, assuming features are pre-softmax or log-softmax
                # Need to apply softmax/log_softmax if not already done
                s_log_softmax = F.log_softmax(s_feat / self.temperature, dim=1)
                t_softmax = F.softmax(t_feat / self.temperature, dim=1)
                loss = F.kl_div(s_log_softmax, t_softmax, reduction='batchmean') * (self.temperature ** 2)
            elif self.loss_type == 'cosine_similarity':
                # Cosine similarity loss (maximize similarity, so minimize 1 - similarity)
                s_norm = F.normalize(s_feat, p=2, dim=1)
                t_norm = F.normalize(t_feat, p=2, dim=1)
                loss = 1 - (s_norm * t_norm).sum(dim=1).mean()
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")
            
            total_loss += loss
        
        return self.alpha * total_loss