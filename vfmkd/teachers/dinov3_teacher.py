"""
DinoV3教师模型实现
基于 DinoV3 vendor 的真实 DinoV3 模型加载和特征提取
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional, Union, List
import warnings

from .base_teacher import BaseTeacher


class DinoV3Teacher(BaseTeacher):
    """真实 DinoV3 教师模型（基于 DinoV3 vendor 实现）"""
    
    def __init__(self, config):
        # 设置模型名称
        self.model_name = 'DinoV3'
        
        # 从配置获取参数
        self.checkpoint_path = config.get('checkpoint_path', 'weights/dinov3_vitb14_pretrain.pth')
        self.device = config.get('device', 'cuda')
        self.model_type = config.get('model_type', 'vit_base')  # vit_small, vit_base, vit_large, vit_giant2
        self.patch_size = config.get('patch_size', 14)  # 14 是标准 patch size
        
        # 特征提取配置
        self.extract_cls_token = config.get('extract_cls_token', True)
        self.extract_patch_tokens = config.get('extract_patch_tokens', True)
        
        # 可视化配置
        self.enable_visualization = config.get('enable_visualization', False)
        self.vis_output_dir = config.get('vis_output_dir', 'teacher_features/dino/visualizations')
        
        # 特征存储配置
        self.feature_output_dir = config.get('feature_output_dir', 'teacher_features/dino/')
        
        super().__init__(config)
        
        # 加载真实 DinoV3 模型
        self.model = self._load_dinov3()
        
        # 冻结模型参数（教师模型通常冻结）
        if config.get('freeze_teacher', True):
            self.freeze()
    
    def _load_dinov3(self):
        """加载真实 DinoV3 模型"""
        try:
            # 添加 DinoV3 vendor 路径到 sys.path
            dinov3_path = Path(__file__).parent.parent / "dinov3"
            if str(dinov3_path) not in sys.path:
                sys.path.insert(0, str(dinov3_path))
            
            # 导入 DinoV3 模块
            # DinoV3 通常通过 torch.hub 或直接导入使用
            # 这里假设 DinoV3 有标准的模型构建接口
            try:
                from dinov3.models import build_model_from_cfg
                from dinov3.utils.config import setup
                use_dinov3_native = True
            except ImportError:
                # 如果上述导入失败，尝试使用 torch.hub 或 transformers
                use_dinov3_native = False
            
            print(f"[INFO] 正在加载真实 DinoV3 模型: {self.model_type}")
            print(f"[INFO] 权重路径: {self.checkpoint_path}")
            print(f"[INFO] 设备: {self.device}")
            
            # 检查权重文件是否存在
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.exists():
                print(f"[WARNING] 权重文件不存在: {checkpoint_path}")
                print("[INFO] 将尝试从官方仓库加载预训练权重")
                checkpoint_path = None
            else:
                print(f"[INFO] 找到权重文件: {checkpoint_path}")
            
            # 设置设备
            dev = torch.device(self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu")
            
            # GPU 内存优化
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.95)
                total_memory = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
                allocated_memory = total_memory * 0.95
                print(f"[GPU] DinoV3 显存分配: 95% (约{allocated_memory:.1f}GB/{total_memory:.1f}GB)")
            
            # 加载模型
            if use_dinov3_native:
                # 使用 DinoV3 原生 API
                # 这里需要根据实际的 DinoV3 API 调整
                model = build_model_from_cfg(
                    model_type=self.model_type,
                    pretrained=checkpoint_path is not None,
                    pretrained_path=str(checkpoint_path) if checkpoint_path else None
                )
            else:
                # 使用 torch.hub 或 transformers
                # DinoV3 可能通过 transformers 库提供
                try:
                    from transformers import AutoModel
                    model_name = f"facebook/dinov3-{self.model_type.replace('vit_', '').replace('giant2', 'g')}"
                    print(f"[INFO] 尝试从 HuggingFace 加载: {model_name}")
                    model = AutoModel.from_pretrained(model_name)
                except Exception as e:
                    print(f"[WARNING] 无法从 HuggingFace 加载: {e}")
                    # 最后的回退：直接加载权重文件
                    if checkpoint_path and checkpoint_path.exists():
                        print(f"[INFO] 直接从权重文件加载...")
                        # 这里需要根据 DinoV3 的实际架构创建模型
                        # 暂时抛出一个有意义的错误
                        raise NotImplementedError(
                            "需要实现 DinoV3 模型的直接加载。"
                            "请确保 DinoV3 vendor 目录正确或使用 transformers 库。"
                        )
                    else:
                        raise FileNotFoundError(
                            f"无法找到 DinoV3 权重文件: {self.checkpoint_path}"
                        )
            
            model = model.to(dev)
            model.eval()
            
            # 获取模型信息
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[SUCCESS] 真实 DinoV3 模型加载成功")
            print(f"   - 模型类型: {self.model_type}")
            print(f"   - 设备: {dev}")
            print(f"   - 参数数量: {n_params:,}")
            
            return model
            
        except ImportError as e:
            print(f"[ERROR] 无法导入 DinoV3 模块: {e}")
            print("[INFO] 请确保:")
            print("  1. 已运行 tools/vendor_dinov3.py vendor DinoV3 代码")
            print("  2. 已安装 DinoV3 的依赖项")
            print("  3. DinoV3 vendor 目录在正确位置: vfmkd/dinov3")
            raise
        except Exception as e:
            print(f"[ERROR] DinoV3 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """准备输入图像"""
        # DinoV3 通常期望输入是 PIL Image 或 Tensor
        # 这里提供基本的预处理
        from PIL import Image
        import torchvision.transforms as transforms
        
        # 转换为 PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if image.ndim == 3 and image.shape[-1] == 3:
                image = Image.fromarray(image)
            else:
                raise ValueError(f"不支持的图像格式: {image.shape}")
        
        # DinoV3 的标准预处理
        # 通常包括 resize 到 518x518 (或根据模型配置)
        transform = transforms.Compose([
            transforms.Resize((518, 518)),  # DinoV3 的标准输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度
        return image_tensor.to(self.device)
    
    def extract_features(self, images: Union[torch.Tensor, np.ndarray], 
                        image_ids: List[str] = None,
                        save_features: bool = True) -> Dict[str, torch.Tensor]:
        """
        提取 DinoV3 特征
        
        Args:
            images: 输入图像 (B, H, W, 3) 或 (B, 3, H, W)
            image_ids: 图片 ID 列表，用于保存和可视化
            save_features: 是否保存特征到文件
            
        Returns:
            特征字典:
            {
                'cls_token': (B, D)  # CLS token 特征
                'patch_tokens': (B, N, D)  # Patch tokens 特征
            }
        """
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        
        # 处理图片 ID
        if image_ids is None:
            if images.ndim == 3:
                image_ids = ["unknown"]
            else:
                image_ids = [f"batch_{i:03d}" for i in range(images.shape[0])]
        
        # 处理批次
        if images.ndim == 4:
            batch_size = images.shape[0]
            all_features = []
            
            for i in range(batch_size):
                img = images[i]
                current_image_id = image_ids[i] if i < len(image_ids) else f"batch_{i:03d}"
                features = self._extract_single_image_features(img, current_image_id, save_features)
                all_features.append(features)
            
            # 合并批次特征
            batch_features = {}
            for key in all_features[0].keys():
                batch_features[key] = torch.stack([feat[key] for feat in all_features])
            
            return batch_features
        else:
            # 单张图片
            current_image_id = image_ids[0] if image_ids else "unknown"
            return self._extract_single_image_features(images, current_image_id, save_features)
    
    def _extract_single_image_features(self, image: np.ndarray, 
                                      image_id: str = "unknown",
                                      save_features: bool = True) -> Dict[str, torch.Tensor]:
        """提取单张图片的特征"""
        with torch.no_grad():
            # 准备输入
            image_tensor = self._prepare_image(image)
            
            # 前向传播
            outputs = self.model(image_tensor)
            
            # 提取特征
            features = {}
            
            # DinoV3 通常返回字典或特殊格式
            # 需要根据实际 API 调整
            if isinstance(outputs, dict):
                if 'cls_token' in outputs and self.extract_cls_token:
                    features['cls_token'] = outputs['cls_token']
                
                if 'patch_tokens' in outputs or 'x_norm_clstoken' in outputs:
                    patch_feat = outputs.get('patch_tokens') or outputs.get('x_norm_clstoken')
                    if patch_feat is not None and self.extract_patch_tokens:
                        features['patch_tokens'] = patch_feat
                
                # 如果有其他特征也提取
                for key in ['x_norm_clstoken', 'x_norm_patchtokens']:
                    if key in outputs:
                        features[key] = outputs[key]
            else:
                # 如果不是字典，尝试直接使用
                # DinoV3 可能返回 tuple 或其他格式
                if hasattr(outputs, 'last_hidden_state'):
                    # transformers 格式
                    hidden_state = outputs.last_hidden_state
                    if self.extract_cls_token:
                        features['cls_token'] = hidden_state[:, 0]  # CLS token
                    if self.extract_patch_tokens:
                        features['patch_tokens'] = hidden_state[:, 1:]  # Patch tokens
                else:
                    # 假设是 tensor
                    if isinstance(outputs, torch.Tensor):
                        if outputs.dim() == 3:  # (B, N, D)
                            if self.extract_cls_token:
                                features['cls_token'] = outputs[:, 0]
                            if self.extract_patch_tokens:
                                features['patch_tokens'] = outputs[:, 1:]
                        elif outputs.dim() == 2:  # (B, D) - 可能是 CLS token
                            features['cls_token'] = outputs
        
        # 保存特征
        if save_features:
            self._save_features(features, image_id)
        
        return features
    
    def _save_features(self, features: Dict[str, torch.Tensor], image_id: str):
        """保存特征到文件"""
        output_dir = Path(self.feature_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备保存数据
        save_data = {}
        for key, feat in features.items():
            if isinstance(feat, torch.Tensor):
                save_data[key] = feat.detach().cpu().numpy()
            else:
                save_data[key] = feat
        
        # 保存 NPZ 文件
        output_file = output_dir / f"{image_id}_dinov3_features.npz"
        np.savez(output_file, **save_data)
        print(f"[SAVE] 特征已保存: {output_file}")
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """前向传播"""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        # 确保是 numpy 数组
        if isinstance(x, np.ndarray):
            if x.ndim == 4:
                # 批次处理
                batch_size = x.shape[0]
                all_features = []
                
                for i in range(batch_size):
                    img = x[i]
                    features = self._extract_single_image_features(img)
                    all_features.append(features)
                
                # 合并批次特征
                batch_features = {}
                for key in all_features[0].keys():
                    batch_features[key] = torch.stack([feat[key] for feat in all_features])
                
                return batch_features
            else:
                # 单张图片
                return self._extract_single_image_features(x)
        else:
            raise ValueError(f"不支持的输入类型: {type(x)}")
    
    def get_feature_types(self) -> List[str]:
        """返回特征类型"""
        feature_types = []
        if self.extract_cls_token:
            feature_types.append('cls_token')
        if self.extract_patch_tokens:
            feature_types.append('patch_tokens')
        return feature_types
    
    def get_feature_dims(self) -> Dict[str, int]:
        """返回特征维度"""
        # DinoV3 的特征维度取决于模型大小
        dim_map = {
            'vit_small': 384,
            'vit_base': 768,
            'vit_large': 1024,
            'vit_giant2': 1536,
        }
        base_dim = dim_map.get(self.model_type, 768)
        
        dims = {}
        if self.extract_cls_token:
            dims['cls_token'] = base_dim
        if self.extract_patch_tokens:
            dims['patch_tokens'] = base_dim
        
        return dims
    
    def get_feature_strides(self) -> list:
        """返回特征下采样倍数"""
        return [self.patch_size]  # DinoV3 的 patch size 通常是 14
    
    def get_model_info(self) -> Dict[str, any]:
        """获取模型详细信息"""
        info = super().get_model_info()
        info.update({
            'model_type': self.model_type,
            'checkpoint_path': self.checkpoint_path,
            'patch_size': self.patch_size,
        })
        return info

