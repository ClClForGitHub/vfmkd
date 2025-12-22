"""
Base teacher interface for VFMKD.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn


class BaseTeacher(ABC, nn.Module):
    """
    Abstract base class for all teacher models.
    
    All teacher models should inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the teacher model.
        
        Args:
            config: Configuration dictionary containing teacher-specific parameters
        """
        super().__init__()
        self.config = config
        self.model_name = self._get_model_name()
        self.feature_types = self._get_feature_types()
        self.frozen = True  # Teachers are typically frozen
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the teacher model.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing teacher features
        """
        pass
    
    @abstractmethod
    def get_feature_types(self) -> List[str]:
        """
        Get the types of features this teacher can provide.
        
        Returns:
            List of feature type strings
        """
        pass
    
    @abstractmethod
    def get_feature_dims(self) -> Dict[str, int]:
        """
        Get the dimensions of each feature type.
        
        Returns:
            Dictionary mapping feature types to dimensions
        """
        pass
    
    def get_model_name(self) -> str:
        """
        Get the name of this teacher model.
        
        Returns:
            Model name string
        """
        return self.model_name
    
    def _get_model_name(self) -> str:
        """Get model name - to be implemented by subclasses."""
        return self.get_model_name()
    
    def _get_feature_types(self) -> List[str]:
        """Get feature types - to be implemented by subclasses."""
        return self.get_feature_types()
    
    def load_pretrained(self, pretrained_path: Optional[str] = None) -> None:
        """
        Load pretrained weights.
        
        Args:
            pretrained_path: Path to pretrained weights file
        """
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys in teacher model: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in teacher model: {unexpected_keys}")
    
    def freeze(self) -> None:
        """Freeze all teacher parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.frozen = True
    
    def unfreeze(self) -> None:
        """Unfreeze all teacher parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.frozen = False
    
    def is_frozen(self) -> bool:
        """Check if teacher is frozen."""
        return self.frozen
    
    def extract_features(self, x: torch.Tensor, 
                        feature_types: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract specific features from the teacher model.
        
        Args:
            x: Input tensor
            feature_types: List of feature types to extract (None for all)
            
        Returns:
            Dictionary of extracted features
        """
        if feature_types is None:
            feature_types = self.feature_types
        
        with torch.no_grad():
            outputs = self.forward(x)
        
        # Filter outputs based on requested feature types
        features = {}
        for feature_type in feature_types:
            if feature_type in outputs:
                features[feature_type] = outputs[feature_type]
            else:
                print(f"Warning: Feature type '{feature_type}' not found in teacher outputs")
        
        return features
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'feature_types': self.feature_types,
            'feature_dims': self.get_feature_dims(),
            'num_parameters': self.get_num_parameters(),
            'num_trainable_parameters': self.get_num_trainable_parameters(),
            'frozen': self.frozen,
        }