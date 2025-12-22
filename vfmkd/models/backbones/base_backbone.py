"""
Base backbone interface for VFMKD.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import torch.nn as nn
import torch
import os


class BaseBackbone(nn.Module, ABC):
    """
    Abstract base class for all backbone networks.
    
    All backbones should inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backbone.
        
        Args:
            config: Configuration dictionary containing backbone-specific parameters
        """
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature tensors at different scales
        """
        pass
    
    @abstractmethod
    def get_feature_dims(self) -> List[int]:
        """
        Get the output feature dimensions of each stage.
        
        Returns:
            List of integers representing feature dimensions
        """
        pass
        
    @abstractmethod
    def get_feature_strides(self) -> List[int]:
        """
        Get the output feature strides (downsampling factors) of each stage.
        
        Returns:
            List of integers representing feature strides
        """
        pass
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get basic information about the backbone model.
        
        Returns:
            Dictionary containing model information (e.g., name, size)
        """
        return {
            "model_name": self.__class__.__name__,
            "feature_dims": self.get_feature_dims(),
            "feature_strides": self.get_feature_strides(),
        }
        
    def freeze(self, freeze_at: int = -1):
        """
        Freeze backbone parameters up to a certain stage.
        
        Args:
            freeze_at: Number of stages to freeze (0 for all, -1 for none)
        """
        # This is a placeholder. Subclasses should implement specific freezing logic.
        for i, param in enumerate(self.parameters()):
            if i < freeze_at:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print(f"Frozen {freeze_at} stages of {self.__class__.__name__}")

    def unfreeze(self):
        """
        Unfreeze all backbone parameters.
        """
        for param in self.parameters():
            param.requires_grad = True
        print(f"Unfrozen all parameters of {self.__class__.__name__}")
    
    def save_weights(self, filepath: str):
        """
        Save backbone weights to file.
        
        Args:
            filepath: Path to save the weights
        """
        import torch
        import os
        
        # 创建目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存权重和配置
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': self.config,
            'model_info': self.get_model_info()
        }
        
        torch.save(checkpoint, filepath)
        print(f"Backbone weights saved to: {filepath}")
    
    def load_weights(self, filepath: str, strict: bool = True):
        """
        Load backbone weights from file.
        
        Args:
            filepath: Path to load the weights from
            strict: Whether to strictly enforce that the keys in state_dict match
        """
        import torch
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weights file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # 加载权重
        self.load_state_dict(checkpoint['state_dict'], strict=strict)
        
        # 更新配置
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        print(f"Backbone weights loaded from: {filepath}")
        
        # 验证模型信息
        if 'model_info' in checkpoint:
            current_info = self.get_model_info()
            loaded_info = checkpoint['model_info']
            print(f"Model info: {current_info}")
            print(f"Loaded info: {loaded_info}")
    
    @classmethod
    def from_pretrained(cls, filepath: str, config: Dict[str, Any] = None):
        """
        Create backbone instance from pretrained weights.
        
        Args:
            filepath: Path to pretrained weights
            config: Configuration dictionary (optional, will use saved config)
            
        Returns:
            Backbone instance with loaded weights
        """
        import torch
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pretrained weights not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # 使用保存的配置或提供的配置
        if config is None and 'config' in checkpoint:
            config = checkpoint['config']
        elif config is None:
            config = {}
        
        # 创建实例
        instance = cls(config)
        
        # 加载权重
        instance.load_state_dict(checkpoint['state_dict'])
        
        print(f"Backbone loaded from pretrained weights: {filepath}")
        return instance