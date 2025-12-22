"""
SAM2教师模型实现
基于SAM2 vendor的真实SAM2模型加载和特征提取
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional, Union, List
import warnings
import matplotlib.pyplot as plt
import cv2

from .base_teacher import BaseTeacher


class SAM2Teacher(BaseTeacher):
    """真实SAM2教师模型（基于SAM2 vendor实现）"""
    
    def __init__(self, config):
        # 设置模型名称
        self.model_name = 'SAM2.1-Hiera'
        
        # 从配置获取参数
        self.checkpoint_path = config.get('checkpoint_path', 'weights/sam2.1_hiera_base_plus.pt')
        self.device = config.get('device', 'cuda')
        self.model_type = config.get('model_type', 'sam2.1_hiera_b+')
        
        # 可视化配置
        self.enable_visualization = config.get('enable_visualization', True)
        self.vis_output_dir = config.get('vis_output_dir', 'datasets/coco128/SAM_Cache/visualizations')
        
        # 特征存储配置
        self.feature_output_dir = config.get('feature_output_dir', 'datasets/coco128/SAM_Cache')
        
        # 多尺度特征保存开关
        # 默认保存 P4(64x64) 和 P5(32x32)
        self.save_p3 = config.get('save_p3', False)  # P3: 128x128
        self.save_p4 = config.get('save_p4', True)   # P4: 64x64 (默认)
        self.save_p5 = config.get('save_p5', True)   # P5: 32x32 (默认)
        
        # 边缘图提取配置
        self.save_edge = config.get('save_edge', True)  # 是否保存边缘图 (默认True)

        # 诊断/回退配置
        # 若检测到P4_S16的std异常偏大（>0.5），将可选回退为/255简单预处理路径的实时特征
        self.enable_diag_compare = config.get('enable_diag_compare', True)
        self.fallback_if_high_std = config.get('fallback_if_high_std', True)

        super().__init__(config)
        
        # 加载真实SAM2模型
        self.model, self.predictor = self._load_sam2()
        
    def _load_sam2(self):
        """加载真实SAM2模型（参考extract_sam2_new.py的实现）"""
        try:
            # 添加SAM2 vendor路径到sys.path
            sam2_path = Path(__file__).parent.parent / "sam2"
            if str(sam2_path) not in sys.path:
                sys.path.insert(0, str(sam2_path))
            
            # 导入SAM2的构建函数和Hydra
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import hydra
            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            
            print(f"[INFO] 正在加载真实SAM2模型: {self.model_type}")
            print(f"[INFO] 权重路径: {self.checkpoint_path}")
            print(f"[INFO] 设备: {self.device}")
            
            # 检查权重文件是否存在
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.exists():
                print(f"[WARNING] 权重文件不存在: {checkpoint_path}")
                print("[INFO] 将创建未初始化的SAM2模型（仅用于测试）")
                checkpoint_path = None
            else:
                print(f"[INFO] 找到权重文件: {checkpoint_path}")
            
            # 设置设备
            dev = torch.device(self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu")
            
            # GPU内存优化
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.95)
                total_memory = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
                allocated_memory = total_memory * 0.95
                print(f"[GPU] SAM2.1显存分配: 95% (约{allocated_memory:.1f}GB/{total_memory:.1f}GB)")
                
                # 启用TF32加速
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    print(f"[GPU] 启用TF32加速（Ampere+架构）")
                except Exception as e:
                    print(f"[GPU] TF32不可用: {e}")
            
            # 设置Hydra配置搜索路径
            sam2_config_dir = sam2_path / "sam2" / "configs"
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            
            # 使用相对路径作为配置文件名
            config_name = f"sam2.1/sam2.1_hiera_b+.yaml"
            
            # 构建SAM2模型
            with initialize_config_dir(config_dir=str(sam2_config_dir), version_base=None):
                model = build_sam2(
                    config_file=config_name, 
                    ckpt_path=str(checkpoint_path) if checkpoint_path else None, 
                    device=str(dev)
                )
            
            model.eval()  # 推理/特征提取模式
            
            # 确保模型在GPU上
            print(f"[INFO] 模型已加载到设备: {dev}")
            print(f"[INFO] 模型参数设备检查: {next(model.parameters()).device}")
            
            # 创建预测器
            predictor = SAM2ImagePredictor(model)
            
            # 获取模型信息
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[SUCCESS] 真实SAM2模型加载成功")
            print(f"   - 模型类型: {self.model_type}")
            print(f"   - 设备: {dev}")
            print(f"   - 参数数量: {n_params:,}")
            print(f"   - 图像编码器: {type(model.image_encoder).__name__}")
            
            # 验证predictor API
            must_have = ["set_image", "predict", "get_image_embedding"]
            miss = [name for name in must_have if not hasattr(predictor, name)]
            if miss:
                raise RuntimeError(f"[E] predictor 缺少必要API: {miss}")
            print("   - predictor API: OK -> set_image / predict / get_image_embedding")
            
            return model, predictor
            
        except ImportError as e:
            print(f"[ERROR] 无法导入SAM2模块: {e}")
            print("[INFO] 请确保SAM2 vendor目录在正确位置")
            raise
        except Exception as e:
            print(f"[ERROR] SAM2模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _visualize_features(self, features: Dict[str, torch.Tensor], image_id: str, original_image: np.ndarray = None):
        """可视化特征能量图"""
        if not self.enable_visualization:
            return
            
        # 创建可视化输出目录
        vis_dir = Path(self.vis_output_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 为每个特征类型生成能量图
        for feat_name, feat_tensor in features.items():
            if feat_tensor.dim() != 4:  # 确保是4D张量 [B, C, H, W]
                continue
                
            # 取第一个batch的特征
            feat = feat_tensor[0].detach().cpu().numpy()  # [C, H, W]
            
            # 计算能量图：对所有通道求L2范数
            energy_map = np.sqrt(np.sum(feat ** 2, axis=0))  # [H, W]
            
            # 归一化到0-1
            energy_map = (energy_map - energy_map.min()) / (energy_map.max() - energy_map.min() + 1e-8)
            
            # 创建可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始图片
            if original_image is not None:
                axes[0].imshow(original_image)
                axes[0].set_title(f'Original Image\n{image_id}')
                axes[0].axis('off')
            else:
                axes[0].text(0.5, 0.5, f'Image ID: {image_id}', ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
            
            # 能量图
            im1 = axes[1].imshow(energy_map, cmap='hot', interpolation='bilinear')
            axes[1].set_title(f'{feat_name} Energy Map\n{feat.shape[1]}x{feat.shape[2]}')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # 叠加图
            if original_image is not None:
                # 将能量图调整到原图尺寸
                h_orig, w_orig = original_image.shape[:2]
                energy_resized = cv2.resize(energy_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                
                # 创建叠加图
                overlay = original_image.copy()
                heatmap = plt.cm.hot(energy_resized)[:, :, :3] * 255
                overlay = cv2.addWeighted(overlay, 0.7, heatmap.astype(np.uint8), 0.3, 0)
                
                axes[2].imshow(overlay)
                axes[2].set_title(f'Overlay\n{feat_name}')
            else:
                axes[2].imshow(energy_map, cmap='hot')
                axes[2].set_title(f'{feat_name} Energy Map')
            
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # 保存可视化结果
            output_path = vis_dir / f"{image_id}_{feat_name}_energy.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[VIS] 特征可视化已保存: {output_path}")

    def _save_features(self, features: Dict[str, torch.Tensor], image_id: str, metadata: Dict = None):
        """保存特征到NPZ文件，按图片ID命名"""
        import time
        
        # 创建特征输出目录
        output_dir = Path(self.feature_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备保存数据
        save_data = {}
        for key, feat in features.items():
            if isinstance(feat, torch.Tensor):
                save_data[key] = feat.detach().cpu().numpy()
            else:
                save_data[key] = feat
        
        # 添加元数据
        if metadata:
            save_data.update(metadata)
        save_data['image_id'] = image_id
        save_data['extraction_time'] = np.array(time.time())
        
        # 保存NPZ文件，按图片ID命名
        output_file = output_dir / f"{image_id}_sam2_features.npz"
        np.savez(output_file, **save_data)
        
        print(f"[SAVE] 特征已保存: {output_file}")
        return output_file
    
    def extract_features(self, images: Union[torch.Tensor, np.ndarray], image_ids: List[str] = None, save_features: bool = True) -> Dict[str, torch.Tensor]:
        """
        提取SAM2多尺度特征，支持可视化和按图片ID保存
        
        Args:
            images: 输入图像 (B, H, W, 3) 或 (B, 3, H, W)
            image_ids: 图片ID列表，用于保存和可视化
            save_features: 是否保存特征到文件
            
        Returns:
            特征字典:
            {
                'IMAGE_EMB_S16': (B, 256, 64, 64)  # 主要特征
            }
        """
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        
        # 确保是numpy数组且为uint8格式
        if images.dtype != np.uint8:
            images = (images * 255).astype(np.uint8)
        
        # 处理图片ID
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
    
    def _extract_single_image_features(self, image: np.ndarray, image_id: str = "unknown", save_features: bool = True) -> Dict[str, torch.Tensor]:
        """提取单张图片的多尺度特征（修复版本：使用正确的vision_features）"""
        # 确保图像格式正确 (H, W, 3)
        if image.ndim == 3 and image.shape[-1] == 3:
            pass  # 正确格式
        elif image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
        else:
            raise ValueError(f"不支持的图像格式: {image.shape}")
        
        # 使用SAM2官方的transform pipeline（包含resize到1024x1024）
        with torch.no_grad():
            # 使用predictor的transform进行预处理（自动resize + normalize）
            # 这是SAM2官方的标准流程
            from sam2.utils.transforms import SAM2Transforms
            if not hasattr(self, '_sam2_transforms'):
                self._sam2_transforms = SAM2Transforms(
                    resolution=self.model.image_size,  # 1024
                    mask_threshold=0.0,
                    max_hole_area=0.0,
                    max_sprinkle_area=0.0
                )
            
            # 使用官方transform（自动resize到1024x1024 + normalize）
            image_tensor = self._sam2_transforms(image).unsqueeze(0)
            
            # 明确确保tensor在正确的GPU设备上
            image_tensor = image_tensor.to(self.device)
            print(f"[DEBUG] image_tensor设备: {image_tensor.device}, dtype: {image_tensor.dtype}, shape: {image_tensor.shape}")
            
            # 直接使用image_encoder获取原始特征（SAM2Transforms: Resize+ToTensor+ImageNet Normalize）
            backbone_out = self.model.image_encoder(image_tensor)
            print(f"[DEBUG] backbone_out设备: {backbone_out['vision_features'].device}")
            
            # 提取多尺度特征
            # vision_features: 主特征 (1, 256, 64, 64) - 16倍下采样
            vision_features = backbone_out['vision_features']
            
            # backbone_fpn: 多尺度特征金字塔
            # 通常包含 [P2, P3, P4, P5] 或类似的多尺度特征
            fpn_features = backbone_out['backbone_fpn']
            
            # 确保是4D张量 (B, C, H, W)
            if vision_features.dim() == 3:
                vision_features = vision_features.unsqueeze(0)
        
        # 构建特征字典 - 使用backbone_fpn索引方式提取，保证纯净无额外处理
        # FPN层映射关系（scalp=0时）：
        # backbone_fpn[0]: 256x256 (P2, S4)
        # backbone_fpn[1]: 128x128 (P3, S8)
        # backbone_fpn[2]: 64x64   (P4, S16) <- 主特征，历史称为IMAGE_EMB_S16
        # backbone_fpn[3]: 32x32   (P5, S32)
        
        print(f"[INFO] FPN总层数: {len(fpn_features)}")
        print(f"[INFO] 启用的特征尺度: P3={self.save_p3}(128x128), P4={self.save_p4}(64x64), P5={self.save_p5}(32x32)")
        
        features = {}

        # 诊断：并行计算一份“简单/255预处理”的P4_S16用于对比
        rt_feat_p4 = None
        if self.enable_diag_compare:
            try:
                from PIL import Image as _PILImage
                import torchvision.transforms.functional as _TF
                img_pil = _PILImage.fromarray(image)
                img_pil = img_pil.resize((self.model.image_size, self.model.image_size))
                img_tensor_simple = _TF.to_tensor(img_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    rt_out = self.model.image_encoder(img_tensor_simple)
                    rt_feat_p4 = rt_out['backbone_fpn'][2]
            except Exception as _e:
                print(f"[DIAG] 生成/255实时特征失败: {_e}")
        
        # 根据配置提取多尺度特征（使用索引方式）
        # 注意：P4_S16 同时作为主特征 IMAGE_EMB_S16（向后兼容）
        fpn_config = [
            (1, 128, 8, 'P3', self.save_p3),   # backbone_fpn[1]
            (2, 64, 16, 'P4', True),           # backbone_fpn[2] - 始终保存作为主特征
            (3, 32, 32, 'P5', self.save_p5),   # backbone_fpn[3]
        ]
        
        for fpn_idx, expected_h, scale, level, is_enabled in fpn_config:
            if not is_enabled:
                print(f"[INFO] 跳过 {level} (backbone_fpn[{fpn_idx}]): 未启用")
                continue
            
            if fpn_idx >= len(fpn_features):
                print(f"[WARNING] {level} (backbone_fpn[{fpn_idx}]): 索引超出FPN范围")
                continue
            
            fpn_feat = fpn_features[fpn_idx]
            if fpn_feat.dim() == 3:
                fpn_feat = fpn_feat.unsqueeze(0)
            
            h, w = fpn_feat.shape[-2:]
            if h != expected_h:
                print(f"[WARNING] {level} 分辨率不匹配: 期望{expected_h}x{expected_h}, 实际{h}x{w}")
            
            # 保存特征
            feat_key = f'{level}_S{scale}'
            # 若为P4层，做分布诊断与可选回退
            if level == 'P4':
                p4_mean = float(fpn_feat.mean())
                p4_std = float(fpn_feat.std())
                rt_mean = rt_std = None
                if rt_feat_p4 is not None:
                    rt_mean = float(rt_feat_p4.mean())
                    rt_std = float(rt_feat_p4.std())
                # 回退条件：P4 std异常大且存在实时参考
                use_rt = (self.fallback_if_high_std and rt_feat_p4 is not None and p4_std > 0.5 and rt_std is not None and rt_std < 0.5)
                if use_rt:
                    print(f"[WARN] P4_S16分布异常: std={p4_std:.3f}，使用/255实时特征回退 std={rt_std:.3f}")
                    features[feat_key] = rt_feat_p4
                else:
                    features[feat_key] = fpn_feat
            else:
                features[feat_key] = fpn_feat
            
            # P4特征同时保存为IMAGE_EMB_S16（向后兼容，避免冗余存储）
            if level == 'P4':
                # 不重复保存，只是添加别名引用（在打印时说明）
                print(f"[INFO] {feat_key}: 从backbone_fpn[{fpn_idx}]提取, shape={fpn_feat.shape} (主特征，兼容键名: IMAGE_EMB_S16)")
            else:
                print(f"[INFO] {feat_key}: 从backbone_fpn[{fpn_idx}]提取, shape={fpn_feat.shape}")
        
        # 打印特征统计
        print(f"[INFO] 图像ID: {image_id}")
        for feat_name, feat_tensor in features.items():
            print(f"  {feat_name}: shape={feat_tensor.shape}, mean={feat_tensor.mean():.4f}, std={feat_tensor.std():.4f}")
        
        # 可视化特征
        self._visualize_features(features, image_id, image)
        
        # 保存特征
        if save_features:
            metadata = {
                'image_shape': np.array(image.shape),
                'model_type': self.model_type,
                'feature_dims': {k: tuple(v.shape) for k, v in features.items()},
                'extraction_method': 'vision_features_correct_with_diag',
            }
            self._save_features(features, image_id, metadata)
        
        return features
    
    def predict_masks(self, images: Union[torch.Tensor, np.ndarray], 
                     boxes: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        预测掩码
        
        Args:
            images: 输入图像
            boxes: 边界框 (N, 4) 格式为 [x1, y1, x2, y2]
            
        Returns:
            掩码字典
        """
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        
        if images.ndim == 4:
            # 批次处理
            batch_size = images.shape[0]
            all_masks = []
            
            for i in range(batch_size):
                img = images[i]
                if boxes is not None and len(boxes) > i:
                    img_boxes = boxes[i] if boxes.ndim == 3 else boxes
                else:
                    img_boxes = None
                
                masks = self._predict_single_image_masks(img, img_boxes)
                all_masks.append(masks)
            
            return all_masks
        else:
            # 单张图片
            return self._predict_single_image_masks(images, boxes)
    
    def _predict_single_image_masks(self, image: np.ndarray, 
                                   boxes: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """预测单张图片的掩码"""
        # 设置图像
        self.predictor.set_image(image)
        
        if boxes is None or len(boxes) == 0:
            # 没有框，返回空掩码
            return {
                'masks': np.zeros((0, 1024, 1024), dtype=np.float32),
                'scores': np.zeros((0,), dtype=np.float32),
                'logits': np.zeros((0, 1024, 1024), dtype=np.float32)
            }
        
        # 使用真实SAM2预测掩码
        with torch.no_grad():
            masks, scores, logits = self.predictor.predict(
                box=boxes.astype(np.float32),
                multimask_output=True,
                return_logits=True
            )
        
        # 确保输出格式正确
        if scores.ndim == 1:
            scores = scores[np.newaxis, :]
        if logits.ndim == 3:
            logits = logits[np.newaxis, :]
        
        # 选择最佳掩码
        best_idx = scores.argmax(axis=1)
        rows = np.arange(best_idx.shape[0])
        best_masks = masks[rows, best_idx]
        best_scores = scores[rows, best_idx]
        best_logits = logits[rows, best_idx]
        
        return {
            'masks': best_masks.astype(np.float32),
            'scores': best_scores.astype(np.float32), 
            'logits': best_logits.astype(np.float32)
        }
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """前向传播"""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        # 确保是numpy数组且为uint8格式
        if x.dtype != np.uint8:
            x = (x * 255).astype(np.uint8)
        
        # 处理批次
        if x.ndim == 4:
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
    
    def get_feature_types(self) -> List[str]:
        """返回特征类型"""
        return ['IMAGE_EMB_S16']
    
    def get_feature_dims(self) -> Dict[str, int]:
        """返回特征维度"""
        return {'IMAGE_EMB_S16': 256}
    
    def get_feature_strides(self) -> list:
        """返回特征下采样倍数"""
        return [16]  # IMAGE_EMB_S16
    
    def get_model_info(self) -> Dict[str, any]:
        """获取模型详细信息"""
        info = super().get_model_info()
        info.update({
            'model_type': self.model_type,
            'checkpoint_path': self.checkpoint_path,
            'image_encoder_type': type(self.model.image_encoder).__name__,
        })
        return info