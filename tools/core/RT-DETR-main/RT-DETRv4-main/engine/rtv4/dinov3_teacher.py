"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core import register
import logging
from torchvision.transforms import v2 as transforms
import os

_logger = logging.getLogger(__name__)


@register()
class DINOv3TeacherModel(nn.Module):
    def __init__(self,
                 dinov3_repo_path: str = None,
                 dinov3_weights_path: str = None,
                 dinov3_model_type: str = "dinov3_vitb16",
                 patch_size: int = 16,
                 use_huggingface: bool = None,  # Auto-detect if None
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.dinov3_repo_path = dinov3_repo_path
        self.dinov3_weights_path = dinov3_weights_path
        self.dinov3_model_type = dinov3_model_type
        self.patch_size = patch_size
        
        # Auto-detect HuggingFace format: check if weights_path is a directory with config.json
        if use_huggingface is None:
            if dinov3_weights_path and os.path.isdir(dinov3_weights_path):
                config_path = os.path.join(dinov3_weights_path, "config.json")
                use_huggingface = os.path.exists(config_path)
            else:
                use_huggingface = False
        
        self.use_huggingface = use_huggingface

        if self.use_huggingface:
            _logger.info(f"[Teacher Model] Loading DINOv3 teacher from HuggingFace format...")
            _logger.info(f"[Teacher Model] DINOv3 weights path: {dinov3_weights_path}")
            
            try:
                from transformers import AutoModel
                
                # Note: We don't use AutoImageProcessor because we handle normalization manually
                # This avoids version compatibility issues and gives us more control
                self.model = AutoModel.from_pretrained(
                    dinov3_weights_path,
                    trust_remote_code=True
                )
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                
                # Get feature dimension from model config
                if hasattr(self.model.config, 'hidden_size'):
                    self.teacher_feature_dim = self.model.config.hidden_size
                elif hasattr(self.model.config, 'embed_dim'):
                    self.teacher_feature_dim = self.model.config.embed_dim
                else:
                    # Default for ViT-B/16
                    self.teacher_feature_dim = 768
                    _logger.warning(f"[Teacher Model] Could not determine feature dim from config, using default 768")
                
                # DINOv3 has 1 class token + 4 register tokens = 5 tokens to skip
                self.num_special_tokens = 5
                
                _logger.info(f"[Teacher Model] Successfully loaded DINOv3 teacher from HuggingFace format.")
                _logger.info(f"[Teacher Model] Feature dimension: {self.teacher_feature_dim}")
                
            except Exception as e:
                _logger.error(f"[Teacher Model] Failed to load DINOv3 from HuggingFace format: {e}")
                raise
        else:
            _logger.info(f"[Teacher Model] Attempting to load DINOv3 teacher via torch.hub.load...")
            _logger.info(f"[Teacher Model] DINOv3 repo path: {dinov3_repo_path}")
            _logger.info(f"[Teacher Model] DINOv3 weights path: {dinov3_weights_path}")

            try:
                self.model = torch.hub.load(
                    dinov3_repo_path,
                    dinov3_model_type,
                    source='local',
                    weights=dinov3_weights_path
                )
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False

                _logger.info(f"[Teacher Model] Successfully loaded DINOv3 teacher from local repo and weights.")
                self.teacher_feature_dim = self.model.embed_dim
                self.num_special_tokens = 0  # Original format already provides patch tokens

            except Exception as e:
                _logger.error(f"[Teacher Model] Failed to load DINOv3: {e}")
                raise

        self.normalize_transform = transforms.Normalize(mean=mean, std=std)
        self.avgpool_2x2 = nn.AvgPool2d(kernel_size=2, stride=2)

        _logger.info(f"[Teacher Model] DINOv3 initialized. Feature dimension: {self.teacher_feature_dim}.")
        _logger.info(
            f"[Teacher Model] Teacher model is configured to output features at a resolution that is 2x2 of the student's highest-level features after 2x downsampling.")

    def forward(self, images: torch.Tensor):
        # Images from dataloader are in [0, 1] range (float32, scale=True)
        # For both formats, we downsample first, then handle normalization differently
        processed_images = self.avgpool_2x2(images)

        with torch.no_grad():
            if self.use_huggingface:
                # HuggingFace format: model expects pixel_values
                # HF DINOv3's processor does: rescale (1/255) + ImageNet normalize
                # Since our images are already in [0, 1], we only need ImageNet normalization
                # Apply ImageNet normalization manually (same as original format)
                normalized_images = self.normalize_transform(processed_images)
                
                outputs = self.model(pixel_values=normalized_images)
                
                # Extract last_hidden_state: [B, N_tokens, C]
                # N_tokens = 1 (class) + 4 (registers) + N_patches
                last_hidden_state = outputs.last_hidden_state  # [B, N_tokens, C]
                
                B, N_tokens, C_teacher = last_hidden_state.shape
                
                # Skip first 5 tokens (class + 4 registers), get patch tokens
                patch_tokens = last_hidden_state[:, self.num_special_tokens:, :]  # [B, N_patches, C]
                N_patches = patch_tokens.shape[1]
                
            else:
                # Original torch.hub format
                dinov3_output_dict = self.model(processed_images, is_training=True, masks=None)
                patch_tokens = dinov3_output_dict["x_norm_patchtokens"]
                B, N_patches, C_teacher = patch_tokens.shape

            if patch_tokens.dim() != 3:
                _logger.error(
                    f"[Teacher Model] Expected 3D patch tokens, but got {patch_tokens.dim()}D tensor. Shape: {patch_tokens.shape}")
                raise ValueError("DINOv3 model's output patch tokens is not in expected 3D format.")

            H_patches_out = W_patches_out = int(N_patches ** 0.5)
            if H_patches_out * W_patches_out != N_patches:
                _logger.error(
                    f"[Teacher Model] Number of patches {N_patches} is not a perfect square for spatial reshape. Input image size: {processed_images.shape[2:]}. Patch size: {self.patch_size}.")
                raise ValueError(
                    f"Number of patches {N_patches} is not a perfect square, cannot reshape to HxW. Check DINOv3 model output or input image size vs patch_size.")

            teacher_feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C_teacher, H_patches_out, W_patches_out)

            _logger.debug(
                f"[Teacher Model] Spatial size: {teacher_feature_map.shape[2:]}")

            return teacher_feature_map.detach()
