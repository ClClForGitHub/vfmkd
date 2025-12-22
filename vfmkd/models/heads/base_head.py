"""
Base head interface for VFMKD.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import torch.nn as nn
import torch


class BaseHead(nn.Module, ABC):
    """
    Abstract base class for all head networks.
    
    All heads should inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the head.
        
        Args:
            config: Configuration dictionary containing head-specific parameters
        """
        super().__init__()
        self.config = config
        self.task_type = config.get('task_type', 'detection')
        
    @abstractmethod
    def forward(self, features: List[torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the head.
        
        Args:
            features: List of feature tensors from backbone
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing head outputs
        """
        pass
    
    @abstractmethod
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for the head.
        
        Args:
            predictions: Dictionary of predictions from forward pass
            targets: Dictionary of ground truth targets
            
        Returns:
            Dictionary containing loss components
        """
        pass
    
    def get_output_channels(self) -> int:
        """
        Get the number of output channels for this head.
        
        Returns:
            Number of output channels
        """
        return self.config.get('output_channels', 1)
    
    def get_task_type(self) -> str:
        """
        Get the task type for this head.
        
        Returns:
            Task type string (e.g., 'detection', 'segmentation')
        """
        return self.task_type
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get basic information about the head model.
        
        Returns:
            Dictionary containing head information
        """
        return {
            "head_name": self.__class__.__name__,
            "task_type": self.task_type,
            "output_channels": self.get_output_channels(),
        }