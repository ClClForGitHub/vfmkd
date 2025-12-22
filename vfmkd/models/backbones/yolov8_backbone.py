"""
YOLOv8 backbone implementation for VFMKD.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

from .base_backbone import BaseBackbone
from .yolov8_components import CSPDarknet


class YOLOv8Backbone(BaseBackbone):
    """
    YOLOv8 backbone implementation.
    
    This backbone implements the CSPDarknet architecture used in YOLOv8,
    providing multi-scale feature extraction for object detection and segmentation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize YOLOv8 backbone.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Get configuration parameters
        self.model_size = config.get('model_size', 'n')
        self.pretrained = config.get('pretrained', True)
        self.freeze_backbone = config.get('freeze_backbone', False)
        self.freeze_at = config.get('freeze_at', -1)
        self.external_weight_path: Optional[str] = config.get('external_weight_path', None)
        
        # Initialize CSPDarknet
        self.csp_darknet = CSPDarknet(model_size=self.model_size)
        
        # Get feature information
        self.feature_dims = self.csp_darknet.get_feature_dims()
        self.feature_strides = self.csp_darknet.get_feature_strides()
        
        # Freeze backbone if specified
        if self.freeze_backbone:
            self.freeze(self.freeze_at)

        # Optional: load external weights (partial non-strict load to backbone only)
        if self.external_weight_path:
            self._load_external_backbone_weights(self.external_weight_path)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through YOLOv8 backbone.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of feature tensors at different scales:
            [feat_s4, feat_s8, feat_s16, feat_s32]
            - feat_s4: 4x downsampled features (256×256 for 1024 input)
            - feat_s8: 8x downsampled features (128×128 for 1024 input)
            - feat_s16: 16x downsampled features (64×64 for 1024 input)
            - feat_s32: 32x downsampled features (32×32 for 1024 input)
        """
        return self.csp_darknet(x)
    
    def get_feature_dims(self) -> List[int]:
        """
        Get the number of channels for each feature level.
        
        Returns:
            List of channel dimensions [dim_s8, dim_s16, dim_s32]
        """
        return self.feature_dims
    
    def get_feature_strides(self) -> List[int]:
        """
        Get the stride (downsampling factor) for each feature level.
        
        Returns:
            List of strides [stride_s8, stride_s16, stride_s32]
        """
        return self.feature_strides
    
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
                print(f"Missing keys in YOLOv8 backbone: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in YOLOv8 backbone: {unexpected_keys}")

    def _load_external_backbone_weights(self, weight_path: str) -> None:
        """
        尝试从外部YOLOv8权重文件中，尽可能加载到本backbone的CSPDarknet部分。
        - 非严格加载；
        - 仅匹配本模块存在的参数键；
        - 允许键名不完全一致时按后缀近似匹配（保守策略）。
        """
        try:
            # 允许反序列化自定义类（Ultralytics DetectionModel）
            ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"[WARN] load external weights failed: {e}")
            return

        # 提取 state_dict
        state_dict = None
        if isinstance(ckpt, dict):
            obj = ckpt.get('model', ckpt.get('state_dict', ckpt))
            if isinstance(obj, dict):
                state_dict = obj
            else:
                # 可能是DetectionModel，取其state_dict()
                try:
                    state_dict = obj.state_dict()
                except Exception:
                    state_dict = None
        else:
            try:
                state_dict = ckpt.state_dict()
            except Exception:
                state_dict = None
        if state_dict is None:
            print("[WARN] external weights: unable to extract state_dict, skip")
            return

        # 仅提取与 csp_darknet 匹配的键
        own_state = self.csp_darknet.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            # 直接键名匹配
            if k in own_state and own_state[k].shape == v.shape:
                filtered[k] = v
                continue
            # 尝试按后缀匹配（如去掉前缀 'model.backbone.' 等）
            for ok in own_state.keys():
                if ok.endswith(k) and own_state[ok].shape == v.shape:
                    filtered[ok] = v
                    break

        missing, unexpected = self.csp_darknet.load_state_dict(filtered, strict=False)
        if missing:
            print(f"[INFO] external load missing keys (ignored): {len(missing)}")
        if unexpected:
            print(f"[INFO] external load unexpected keys (ignored): {len(unexpected)}")
    
    def get_num_parameters(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': f'YOLOv8-{self.model_size}',
            'model_size': self.model_size,
            'feature_dims': self.feature_dims,
            'feature_strides': self.feature_strides,
            'num_parameters': self.get_num_parameters(),
            'num_trainable_parameters': self.get_num_trainable_parameters(),
            'frozen': self.frozen if hasattr(self, 'frozen') else False,
        }


def create_yolov8_backbone(config: Dict[str, Any]) -> YOLOv8Backbone:
    """
    Create YOLOv8 backbone from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        YOLOv8Backbone instance
    """
    return YOLOv8Backbone(config)
