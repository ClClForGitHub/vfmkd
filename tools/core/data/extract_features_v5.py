#!/usr/bin/env python3
"""
SA-1B 通用处理流水线 V5 (Universal Edition - Ultimate Quality)

环境: RTX 3090 + 100核 CPU

变更日志:
- [Quality] JPEG 保存质量提升至 100 (最高质量)
- [Quality] 强制关闭色度采样 (subsampling=0)，保持 4:4:4 原色采样，确保训练数据无损级画质
- [Fix] Resize 逻辑维持 PIL.Image.resize(..., Image.BILINEAR) 以对齐训练代码
- [New] 新增三种运行模式：full / pack_only / resize_only
- [Opt] 保持多进程和异步 I/O 架构

功能模式 (--mode):
1. full (默认): [原图] -> PIL Resize/编码(内存) -> SAM2提特征(GPU) -> 校验 -> [Tar Shard]
2. pack_only: [现有NPZ目录] + [现有JPG目录] -> 极速扫描匹配 -> [Tar Shard]
3. resize_only: [原图] -> PIL Resize -> [JPG目录]
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import cv2
from PIL import Image
from pycocotools import mask as mask_utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import concurrent.futures
import tarfile
import io
import time
import multiprocessing as mp
import gc

# 添加项目路径和SAM2路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sam2_path = project_root / "vfmkd" / "sam2"
if str(sam2_path) not in sys.path:
    sys.path.insert(0, str(sam2_path))

try:
    from vfmkd.teachers.sam2_teacher import SAM2Teacher
    from tools.core.bbox.test_bbox_strategies import compute_strategy_geometry_color
    SAM2_AVAILABLE = True
except ImportError:
    SAM2Teacher = None
    compute_strategy_geometry_color = None
    SAM2_AVAILABLE = False


# =========================================================================
# 模块 A: Resize Only Worker (多进程 - PIL版)
# =========================================================================

def _resize_worker(args):
    """单一图片的 Resize 任务 (使用 PIL，与训练时完全一致)"""
    src_path, dst_path, target_size = args
    try:
        if os.path.exists(dst_path):
            return 0  # Skipped
        
        # 使用 PIL 打开和处理，确保与训练一致
        image_pil = Image.open(src_path).convert('RGB')
        
        # 仅当尺寸不匹配时才 Resize
        if image_pil.size != (target_size, target_size):
            # 使用 BILINEAR 插值（与训练时完全一致）
            image_pil = image_pil.resize((target_size, target_size), Image.BILINEAR)
        
        # 创建目录
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # 保存为最高质量 JPG（质量100 + 关闭色度采样，保持4:4:4原色采样）
        # subsampling=0: 关闭色度子采样，确保无损级画质
        image_pil.save(dst_path, format='JPEG', quality=100, subsampling=0)
        
        return 1  # Success
    except Exception as e:
        print(f"❌ Resize Error: {src_path} -> {e}")
        return -1  # Error


def run_resize_only(args):
    """模式：仅 Resize 图片"""
    print(f"\n{'='*60}")
    print("🚀 启动模式: RESIZE ONLY (PIL High-Quality)")
    print(f"{'='*60}")
    
    src_dir = Path(args.data_dir)
    dst_dir = Path(args.output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 输入目录: {src_dir}")
    print(f"📁 输出目录: {dst_dir}")
    print(f"📐 目标尺寸: {args.target_size}x{args.target_size}")
    print("\n正在扫描图片...")
    
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    files = []
    for ext in exts:
        files.extend(list(src_dir.rglob(ext)))
    
    print(f"📊 找到 {len(files)} 张图片，准备处理...")
    
    tasks = []
    for p in files:
        rel_path = p.relative_to(src_dir)
        out_p = (dst_dir / rel_path).with_suffix('.jpg')
        tasks.append((str(p), str(out_p), args.target_size))
    
    success, skip, error = 0, 0, 0
    
    # 多进程处理
    with mp.Pool(args.num_workers) as pool:
        for res in tqdm(pool.imap_unordered(_resize_worker, tasks), total=len(tasks), desc="Resize进度"):
            if res == 1:
                success += 1
            elif res == 0:
                skip += 1
            else:
                error += 1
    
    print(f"\n{'='*60}")
    print("✅ Resize 完成!")
    print(f"   成功: {success}")
    print(f"   跳过: {skip} (已存在)")
    print(f"   失败: {error}")
    print(f"{'='*60}\n")


# =========================================================================
# 模块 B: Pack Only Worker (极速优化版)
# =========================================================================

def _pack_worker(args):
    """打包单个 Shard"""
    shard_idx, shard_data, output_dir, mode = args
    shard_name = os.path.join(output_dir, f"sa1b_shard_{shard_idx:05d}.tar")
    
    count = 0
    try:
        with tarfile.open(shard_name, mode, bufsize=4*1024*1024, format=tarfile.PAX_FORMAT) as tar:
            for npz_path, img_path, image_id in shard_data:
                # 添加 NPZ 文件
                tar.add(npz_path, arcname=os.path.basename(npz_path))
                # 添加 JPG 文件（规范化命名: sa_xxxx.jpg 或保持原名）
                clean_id = image_id.replace("sa_", "") if image_id.startswith("sa_") else image_id
                jpg_arcname = f"sa_{clean_id}.jpg" if not os.path.basename(img_path).startswith("sa_") else os.path.basename(img_path)
                tar.add(img_path, arcname=jpg_arcname)
                count += 1
        return count, None
    except Exception as e:
        return 0, str(e)


def _fast_scan(folder, suffix):
    """os.scandir 极速扫描"""
    print(f"📁 正在扫描 {folder} ...")
    idx = {}
    lst = []
    gc.disable()
    try:
        with os.scandir(folder) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(suffix):
                    if suffix == '.jpg':
                        stem = entry.name[:-4]
                        # 规范化key：sa_xxxx 或 xxxx 都映射到统一格式
                        if stem.startswith('sa_'):
                            key = stem
                        else:
                            key = f"sa_{stem}"
                        idx[key] = entry.path
                    else:
                        # _features.npz len=13
                        key = entry.name[:-13]
                        lst.append(entry.path)
    finally:
        gc.enable()
    lst.sort()
    return idx, lst


def run_pack_only(args):
    """模式：仅打包已有文件"""
    print(f"\n{'='*60}")
    print("🚀 启动模式: PACK ONLY (极速版)")
    print(f"{'='*60}")
    
    npz_dir = args.data_dir
    jpg_dir = args.images_dir
    
    if not jpg_dir:
        print("❌ 错误: Pack模式需要指定 --images-dir (JPG图片目录)")
        return
    
    if not os.path.exists(npz_dir):
        print(f"❌ 错误: NPZ目录不存在: {npz_dir}")
        return
    
    if not os.path.exists(jpg_dir):
        print(f"❌ 错误: JPG目录不存在: {jpg_dir}")
        return
    
    print(f"📁 NPZ目录: {npz_dir}")
    print(f"📁 JPG目录: {jpg_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    
    # 极速扫描
    img_index, _ = _fast_scan(jpg_dir, '.jpg')
    _, npz_files = _fast_scan(npz_dir, '_features.npz')
    
    print(f"\n📊 扫描完成: JPG {len(img_index)} 个, NPZ {len(npz_files)} 个")
    
    # 匹配
    pairs = []
    for npz in tqdm(npz_files, desc="匹配中"):
        fname = os.path.basename(npz)
        key = fname[:-13]  # 去掉 '_features.npz'
        if key in img_index:
            pairs.append((npz, img_index[key], key))
    
    print(f"✅ 匹配成功: {len(pairs)} 对")
    
    if len(pairs) == 0:
        print("⚠️  没有匹配的文件对，退出")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    shard_size = args.shard_size
    num_shards = (len(pairs) + shard_size - 1) // shard_size
    
    tasks = []
    mode = "w"
    for i in range(num_shards):
        sub = pairs[i*shard_size : (i+1)*shard_size]
        tasks.append((i, sub, args.output_dir, mode))
    
    workers = min(args.num_workers, num_shards)
    print(f"\n🚀 启动 {workers} 个进程打包...")
    print(f"   将生成 {num_shards} 个 Shard (每个约 {shard_size} 个样本)\n")
    
    success_count = 0
    error_count = 0
    
    with mp.Pool(workers) as pool:
        for count, err in tqdm(pool.imap_unordered(_pack_worker, tasks), total=len(tasks), desc="打包进度"):
            if err:
                print(f"❌ Shard Error: {err}")
                error_count += 1
            else:
                success_count += count
    
    print(f"\n{'='*60}")
    print("✅ 打包完成!")
    print(f"   成功打包: {success_count} 个样本")
    print(f"   失败: {error_count} 个 Shard")
    print(f"{'='*60}\n")


# =========================================================================
# 模块 C: Full Pipeline (V4 完整逻辑 - PIL版)
# =========================================================================

def verify_data_integrity(npz_dict, args):
    """校验关键数据是否存在"""
    missing = []
    
    # 1. 验证 Edge (edge_256x256 必须保存)
    if args.save_edge:
        if 'edge_256x256' not in npz_dict:
            missing.append('edge_256x256')
        if args.enable_s16 and 'edge_64x64' not in npz_dict:
            missing.append('edge_64x64')
        if args.enable_s32 and 'edge_32x32' not in npz_dict:
            missing.append('edge_32x32')
    
    # 2. 验证 Weights
    if args.save_weights:
        if args.enable_s4 and 'fg_map_256x256' not in npz_dict:
            missing.append('fg_map_256x256')
        if args.enable_s8 and 'fg_map_128x128' not in npz_dict:
            missing.append('fg_map_128x128')
        if args.enable_s16 and 'fg_map_64x64' not in npz_dict:
            missing.append('fg_map_64x64')
        if args.enable_s32 and 'fg_map_32x32' not in npz_dict:
            missing.append('fg_map_32x32')
    
    # 3. 验证 Features (根据启用的层级)
    if args.save_feature:
        if args.enable_s4:
            has_s4 = any('S4' in k or 'P2' in k or '256' in k for k in npz_dict.keys())
            if not has_s4:
                missing.append('feature_S4')
        if args.enable_s8:
            has_s8 = any('S8' in k or 'P3' in k or '128' in k for k in npz_dict.keys())
            if not has_s8:
                missing.append('feature_S8')
        if args.enable_s16:
            has_s16 = any('S16' in k or 'P4' in k or '64' in k for k in npz_dict.keys())
            if not has_s16:
                missing.append('feature_S16')
        if args.enable_s32:
            has_s32 = any('S32' in k or 'P5' in k or '32' in k for k in npz_dict.keys())
            if not has_s32:
                missing.append('feature_S32')
    
    # 4. 验证 BBox (如果开启且标记为有bbox)
    if args.save_bbox:
        if npz_dict.get('has_bbox', False):
            if 'bboxes' not in npz_dict:
                missing.append('bboxes')
    
    return (len(missing) == 0), missing


class CPUWorker_Full:
    """Full模式专用的 Worker (集成 PIL + 最高画质，与训练时完全一致)"""
    
    def __init__(self, config):
        self.kernel_size = config.get('kernel_size', 3)
        self.kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        self.max_instances = config.get('max_instances', 1)
        self.enable_bbox_selection = config.get('enable_bbox_selection', True)
        self.target_size = 1024
        
        self.edge_sizes = [256, 64, 32]
        self.weight_sizes = [256, 128, 64, 32]
    
    def process(self, image_path, json_path):
        try:
            # 1. 加载图像 (使用 PIL，与训练时完全一致)
            # 训练时使用: Image.open().convert('RGB').resize((1024, 1024))
            image_pil = Image.open(str(image_path)).convert('RGB')
            
            # 2. Resize (PIL BILINEAR，与训练时完全一致)
            if image_pil.size != (self.target_size, self.target_size):
                image_pil_resized = image_pil.resize((self.target_size, self.target_size), Image.BILINEAR)
            else:
                image_pil_resized = image_pil
            
            # 3. 编码为 JPG Bytes (用于保存到 Tar)
            # 使用PIL保存，质量100 + 关闭色度采样（保持4:4:4原色采样，确保无损级画质）
            # subsampling=0: 关闭色度子采样，避免颜色信息损失
            jpg_io = io.BytesIO()
            image_pil_resized.save(jpg_io, format='JPEG', quality=100, subsampling=0)
            jpg_bytes = jpg_io.getvalue()
            
            # 4. 转换数据供计算使用
            # SAM2 Teacher 和 边缘计算需要 Numpy 数组
            # 注意: PIL 是 RGB，OpenCV 需要 BGR
            image_rgb_np = np.array(image_pil)
            image_bgr_np = cv2.cvtColor(image_rgb_np, cv2.COLOR_RGB2BGR)
            
            # 5. 加载 JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 6. 计算 Edge/Weights (使用 numpy)
            edge_maps, weight_maps = self._compute_edges_and_weights(json_data)
            
            result = {
                'success': True,
                'image_id': Path(image_path).stem,
                'original_shape': np.array(image_rgb_np.shape),
                'edge_maps': edge_maps,
                'weight_maps': weight_maps,
                'jpg_bytes': jpg_bytes,
            }
            
            # 7. 选框策略 (传入 BGR numpy)
            if self.enable_bbox_selection and compute_strategy_geometry_color:
                bbox_data = self._compute_bbox(json_data, image_bgr_np)
                result.update(bbox_data)
            else:
                result.update({
                    'has_bbox': False,
                    'bboxes': np.empty((0, 4), dtype=np.float32),
                    'num_bboxes': 0,
                    'geometry_color_flag': 0
                })
            
            # 返回 RGB numpy 给 SAM2 (SAM2 通常处理 RGB)
            return image_rgb_np, result
            
        except Exception as e:
            return None, {
                'success': False,
                'image_id': Path(image_path).stem,
                'error': str(e)
            }
    
    def _compute_edges_and_weights(self, json_data):
        """完全保留原有的核心算法逻辑：Method B边缘提取 + 权重图生成"""
        image_info = json_data.get('image', {})
        height = int(image_info.get('height', image_info.get('h', 0)))
        width = int(image_info.get('width', image_info.get('w', 0)))
        annotations = json_data.get('annotations', [])
        
        # Method B：每个实例单独提取边缘后合并
        combined_edge_map = np.zeros((height, width), dtype=np.uint8)
        union_mask = np.zeros((height, width), dtype=np.uint8)
        
        if len(annotations) > 0:
            for ann in annotations:
                rle = ann.get('segmentation')
                if rle is None:
                    continue
                
                mask = mask_utils.decode(rle)
                
                # 合并掩码（用于权重图）
                union_mask = np.maximum(union_mask, mask)
                
                # 对每个实例单独提取边缘
                edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, self.kernel)
                edge = (edge > 0).astype(np.uint8)
                
                # 使用bitwise_or替代logical_or（直接在uint8上操作）
                combined_edge_map = np.bitwise_or(combined_edge_map, edge)
        
        # 生成多尺度边缘图
        edge_maps = {}
        edge_float = combined_edge_map.astype(np.float32)
        for size in self.edge_sizes:
            edge_small = cv2.resize(edge_float, (size, size), interpolation=cv2.INTER_AREA)
            edge_maps[size] = (edge_small > 0).astype(np.uint8)
        
        # 生成多尺度权重图（使用Torch CPU进行池化）
        weight_maps = {}
        union_tensor = torch.from_numpy(union_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        
        for size in self.weight_sizes:
            fg_prob = F.adaptive_avg_pool2d(union_tensor, (size, size)).squeeze().numpy()
            
            # 前景权重：area归一化
            fg_binary = (fg_prob > 0.5).astype(np.float32)
            num_fg = np.clip(fg_binary.sum(), a_min=1, a_max=None)
            fg_map = fg_binary / num_fg
            
            # 背景权重：归一化
            bg_map = 1.0 - fg_binary
            bg_sum = bg_map.sum()
            if bg_sum > 0:
                bg_map = bg_map / bg_sum
            
            weight_maps[size] = {
                'fg_map': fg_map.astype(np.float32),
                'bg_map': bg_map.astype(np.float32)
            }
        
        return edge_maps, weight_maps
    
    def _compute_bbox(self, json_data, image_bgr):
        """执行选框策略，完全保留原有逻辑"""
        try:
            H, W = image_bgr.shape[:2]
            
            # 构造符合接口的数据结构
            data = {
                'image': {
                    'height': H,
                    'width': W,
                    'h': H,
                    'w': W,
                }
            }
            annotations = json_data.get('annotations', [])
            
            # 注意：compute_strategy_geometry_color 内部会转换BGR到RGB
            selected_components = compute_strategy_geometry_color(
                data=data,
                annotations=annotations,
                image_rgb=image_bgr,  # 传入BGR，函数内部会转换
                clip_data=None,
                max_instances=self.max_instances,
                max_display=0,
                debug_trace=None,
            )
            
            if selected_components and len(selected_components) > 0:
                bboxes = []
                masks = []
                for comp in selected_components:
                    bboxes.append(comp['box'])
                    masks.append(comp['mask'])
                
                return {
                    'has_bbox': True,
                    'bboxes': np.array(bboxes, dtype=np.float32),
                    'num_bboxes': len(bboxes),
                    'masks': masks,
                    'geometry_color_flag': 1
                }
            else:
                return {
                    'has_bbox': False,
                    'bboxes': np.empty((0, 4), dtype=np.float32),
                    'num_bboxes': 0,
                    'masks': [],
                    'geometry_color_flag': 1
                }
        except Exception as e:
            return {
                'has_bbox': False,
                'bboxes': np.empty((0, 4), dtype=np.float32),
                'num_bboxes': 0,
                'masks': [],
                'geometry_color_flag': 0
            }


class SA1BDataset_Full(Dataset):
    """Full模式的Dataset"""
    
    def __init__(self, data_dir, output_dir, worker_config, max_images=None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.worker = CPUWorker_Full(worker_config)
        
        # 扫描文件
        print("📁 正在扫描数据集...")
        all_images = list(self.data_dir.glob("*.jpg"))
        if max_images:
            all_images = all_images[:max_images]
        
        # 去重逻辑（检查已有Shard）
        self.task_list = []
        existing_shards = set()
        for tar_file in self.output_dir.glob("sa1b_shard_*.tar"):
            existing_shards.add(tar_file.stem)
        
        for img_path in all_images:
            json_path = self.data_dir / f"{img_path.stem}.json"
            if json_path.exists():
                self.task_list.append((img_path, json_path))
        
        print(f"📊 任务统计: 总计 {len(all_images)} | 有效任务 {len(self.task_list)}")
        if existing_shards:
            print(f"⚠️  检测到已有 Shard 文件 {len(existing_shards)} 个，请确认不会覆盖")
    
    def __len__(self):
        return len(self.task_list)
    
    def __getitem__(self, idx):
        image_path, json_path = self.task_list[idx]
        return self.worker.process(image_path, json_path)


def collate_fn_full(batch):
    """自定义 batch 处理，过滤失败的样本"""
    images = []
    results = []
    for img, res in batch:
        if img is not None and res.get('success', False):
            images.append(img)
            results.append(res)
    
    if not images:
        return None, None
    
    return images, results


def assemble_save_dict(cpu_data, gpu_features, args, teacher_model_type):
    """组装保存字典，完全遵循 V4 的保存逻辑规则"""
    save_dict = {}
    
    # 1. 特征 (Features) - 仅当 Master 开关开启时
    if args.save_feature and gpu_features:
        feat_configs = [
            (args.enable_s4, 'P2', 256, 'S4'),
            (args.enable_s8, 'P3', 128, 'S8'),
            (args.enable_s16, 'P4', 64, 'S16'),
            (args.enable_s32, 'P5', 32, 'S32'),
        ]
        
        for enabled, key_prefix, size, scale_name in feat_configs:
            if not enabled:
                continue
            
            found_key = None
            for feat_key in gpu_features.keys():
                if key_prefix in feat_key and str(size) in feat_key:
                    found_key = feat_key
                    break
                if key_prefix in feat_key and scale_name in feat_key:
                    found_key = feat_key
                    break
            
            if found_key and found_key in gpu_features:
                feat_tensor = gpu_features[found_key]
                if feat_tensor.dim() == 4:
                    feat_tensor = feat_tensor.squeeze(0)
                if isinstance(feat_tensor, torch.Tensor):
                    feat_tensor = feat_tensor.detach().cpu().numpy()
                save_dict[found_key] = feat_tensor
            elif enabled:
                # 插值补全
                base_key = None
                base_feat = None
                for k in ['P4_S16', 'P5_S32', 'P3_S8']:
                    if k in gpu_features:
                        base_key = k
                        base_feat = gpu_features[k]
                        break
                
                if base_feat is not None:
                    if isinstance(base_feat, torch.Tensor):
                        base_feat = base_feat.detach().cpu()
                    if base_feat.dim() == 4:
                        base_feat = base_feat.squeeze(0)
                    base_feat = base_feat.unsqueeze(0)
                    resized = F.interpolate(
                        base_feat.unsqueeze(0),
                        size=(size, size),
                        mode='bilinear',
                        align_corners=False
                    )
                    feat_key = f'{key_prefix}_{scale_name}'
                    save_dict[feat_key] = resized.squeeze(0).squeeze(0).cpu().numpy()
    
    # 2. 边缘 (Edges)
    if args.save_edge and 'edge_maps' in cpu_data:
        edge_maps = cpu_data['edge_maps']
        if 256 in edge_maps:
            save_dict['edge_256x256'] = edge_maps[256]
        if args.enable_s16 and 64 in edge_maps:
            save_dict['edge_64x64'] = edge_maps[64]
        if args.enable_s32 and 32 in edge_maps:
            save_dict['edge_32x32'] = edge_maps[32]
    
    # 3. 权重 (Weights)
    if args.save_weights and 'weight_maps' in cpu_data:
        weight_maps = cpu_data['weight_maps']
        w_configs = [
            (args.enable_s4, 256),
            (args.enable_s8, 128),
            (args.enable_s16, 64),
            (args.enable_s32, 32)
        ]
        for enabled, size in w_configs:
            if enabled and size in weight_maps:
                save_dict[f'fg_map_{size}x{size}'] = weight_maps[size]['fg_map']
                save_dict[f'bg_map_{size}x{size}'] = weight_maps[size]['bg_map']
    
    # 4. 选框 (BBox)
    if args.save_bbox and 'bboxes' in cpu_data:
        if cpu_data.get('has_bbox', False):
            save_dict['has_bbox'] = np.array(True, dtype=bool)
            save_dict['num_bboxes'] = np.array(cpu_data['num_bboxes'], dtype=np.int32)
            save_dict['bboxes'] = cpu_data['bboxes']
            save_dict['geometry_color_flag'] = np.array(cpu_data.get('geometry_color_flag', 1), dtype=np.uint8)
            if 'masks' in cpu_data and len(cpu_data['masks']) > 0:
                save_dict['masks'] = np.array(cpu_data['masks'], dtype=object)
        else:
            save_dict['has_bbox'] = np.array(False, dtype=bool)
            save_dict['num_bboxes'] = np.array(0, dtype=np.int32)
            save_dict['bboxes'] = np.empty((0, 4), dtype=np.float32)
            save_dict['geometry_color_flag'] = np.array(cpu_data.get('geometry_color_flag', 0), dtype=np.uint8)
    
    # 5. 元数据 (总是保存)
    save_dict['image_id'] = cpu_data['image_id']
    save_dict['image_shape'] = cpu_data['original_shape']
    save_dict['model_type'] = teacher_model_type
    
    return save_dict


def _full_write_task(shard_idx, buffer, output_dir):
    """Full模式的后台写入线程"""
    name = output_dir / f"sa1b_shard_{shard_idx:05d}.tar"
    try:
        with tarfile.open(name, "w") as tar:
            for item in buffer:
                image_id = item['image_id']
                jpg_bytes = item['jpg_bytes']
                npz_data = item['npz_data']
                
                # 写入 JPG
                clean_id = image_id.replace("sa_", "") if image_id.startswith("sa_") else image_id
                jpg_name = f"sa_{clean_id}.jpg"
                
                jpg_io = io.BytesIO(jpg_bytes)
                ti = tarfile.TarInfo(name=jpg_name)
                ti.size = len(jpg_bytes)
                tar.addfile(ti, jpg_io)
                
                # 写入 NPZ
                npz_io = io.BytesIO()
                np.savez_compressed(npz_io, **npz_data)
                nb = npz_io.getvalue()
                
                npz_name = f"{image_id}_features.npz"
                ti = tarfile.TarInfo(name=npz_name)
                ti.size = len(nb)
                npz_io.seek(0)
                tar.addfile(ti, npz_io)
        
        print(f"📦 [Shard {shard_idx:05d}] 写入完成 ({len(buffer)} 个样本) -> {name.name}")
        return True
    except Exception as e:
        print(f"❌ [Shard {shard_idx:05d}] 写入失败: {e}")
        return False


def run_full_pipeline(args):
    """模式：完整流水线"""
    print(f"\n{'='*60}")
    print("🚀 启动模式: FULL PIPELINE (One-Pass PIL Edition)")
    print(f"{'='*60}")
    
    if not SAM2_AVAILABLE:
        print("❌ 错误: 无法导入 SAM2Teacher，Full模式需要SAM2支持")
        return
    
    # 构建配置
    _ckpt = args.checkpoint
    if _ckpt is None:
        if args.teacher_model.endswith('hiera_b+'):
            _ckpt = 'weights/sam2.1_hiera_base_plus.pt'
        else:
            _ckpt = 'weights/sam2.1_hiera_base_plus.pt'
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化 Dataset 和 DataLoader
    print(f"🔥 初始化 CPU Worker (Num Workers: {args.num_workers})...")
    
    worker_config = {
        'kernel_size': args.kernel_size,
        'max_instances': args.max_instances,
        'enable_bbox_selection': args.save_bbox
    }
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset = SA1BDataset_Full(args.data_dir, args.output_dir, worker_config, args.max_images)
    
    if len(dataset) == 0:
        print("✅ 所有任务已完成，无需处理")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_full,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    # 初始化 GPU 模型
    print(f"🔥 初始化 GPU Teacher ({args.teacher_model})...")
    teacher_config = {
        'model_type': args.teacher_model,
        'checkpoint_path': _ckpt,
        'device': device,
        'enable_visualization': False,
        'feature_output_dir': args.output_dir,
        'enable_diag_compare': bool(args.diag_compare),
        'fallback_if_high_std': bool(args.diag_fallback),
    }
    
    teacher = SAM2Teacher(teacher_config)
    print(f"✅ GPU模型初始化完成: {teacher.model_name}")
    
    # 打印配置信息
    print(f"\n📋 配置信息:")
    print(f"   数据目录: {args.data_dir}")
    print(f"   输出目录: {args.output_dir}")
    print(f"   教师模型: {args.teacher_model}")
    print(f"   权重文件: {teacher_config['checkpoint_path']}")
    print(f"   计算设备: {device}")
    print(f"   最大图像数: {args.max_images or '全部'}")
    print(f"   CPU进程数: {args.num_workers}")
    print(f"   GPU批大小: {args.batch_size}")
    print(f"   Shard大小: {args.shard_size}")
    print(f"\n💾 保存设置:")
    print(f"   Feature: {args.save_feature} | Edge: {args.save_edge} | Weight: {args.save_weights} | BBox: {args.save_bbox}")
    print(f"   层级设置 -> S4:{args.enable_s4} | S8:{args.enable_s8} | S16:{args.enable_s16} | S32:{args.enable_s32}")
    print(f"\n🚀 开始流水线处理...\n")
    
    # Buffer State
    shard_buffer = []
    shard_counter = 0
    total_processed = 0
    success_count = 0
    error_count = 0
    
    timing_stats = {
        'gpu_inference': [],
        'assemble': [],
    }
    
    # 后台写入线程池
    io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    try:
        for batch_idx, batch_result in enumerate(tqdm(dataloader, desc="提取特征")):
            if batch_result is None or batch_result[0] is None:
                continue
            
            images_list, cpu_results_list = batch_result
            
            # 逐个过模型 (SAM2Teacher通常不支持batch，所以循环处理)
            for i, img_rgb in enumerate(images_list):
                res = cpu_results_list[i]
                img_id = res['image_id']
                
                try:
                    # 1. GPU 推理
                    gpu_feats = {}
                    if args.save_feature:
                        gpu_start = time.time()
                        with torch.no_grad():
                            gpu_feats = teacher.extract_features(
                                img_rgb,
                                image_ids=[img_id],
                                save_features=False
                            )
                        gpu_time = time.time() - gpu_start
                        timing_stats['gpu_inference'].append(gpu_time)
                    
                    # 2. 组装数据字典
                    assemble_start = time.time()
                    save_dict = assemble_save_dict(res, gpu_feats, args, args.teacher_model)
                    assemble_time = time.time() - assemble_start
                    timing_stats['assemble'].append(assemble_time)
                    
                    # 3. 验证数据完整性
                    is_valid, missing = verify_data_integrity(save_dict, args)
                    if not is_valid:
                        error_count += 1
                        print(f"⚠️ [Skip] {img_id} 缺失关键数据: {missing}")
                        continue
                    
                    # 4. 加入 Buffer
                    shard_buffer.append({
                        'image_id': img_id,
                        'jpg_bytes': res['jpg_bytes'],
                        'npz_data': save_dict
                    })
                    
                    total_processed += 1
                    success_count += 1
                    
                    # 5. 触发写盘
                    if len(shard_buffer) >= args.shard_size:
                        buffer_to_write = shard_buffer[:]
                        shard_buffer = []
                        io_executor.submit(_full_write_task, shard_counter, buffer_to_write, output_path)
                        shard_counter += 1
                        print(f"📦 [进度] 已处理 {total_processed} 张 | 已生成 {shard_counter} 个 Shard | Buffer: {len(shard_buffer)}")
                    
                    # 每100张打印一次详细计时
                    if success_count % 100 == 0:
                        gpu_str = f"GPU推理{gpu_time:.3f}s" if args.save_feature else "跳过GPU"
                        print(f"✅ {img_id}: {gpu_str} | 组装{assemble_time:.3f}s | 累计{success_count}张")
                
                except Exception as e:
                    error_count += 1
                    print(f"\n❌ {img_id}: 处理失败 - {e}")
                    import traceback
                    traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    finally:
        # 处理剩余 Buffer
        if len(shard_buffer) > 0:
            print(f"\n🧹 清理剩余 Buffer ({len(shard_buffer)} 个)...")
            io_executor.submit(_full_write_task, shard_counter, shard_buffer, output_path)
            shard_counter += 1
        
        print("⏳ 等待后台写入任务完成...")
        io_executor.shutdown(wait=True)
        
        # 打印统计信息
        print(f"\n{'='*60}")
        print("🎉 特征提取完成!")
        print(f"{'='*60}")
        print(f"✅ 本次成功: {success_count} 个")
        print(f"❌ 失败: {error_count} 个")
        print(f"📦 生成 Shard: {shard_counter} 个")
        
        # 打印平均耗时统计
        if success_count > 0:
            print(f"\n⏱️  平均耗时 (每张):")
            if timing_stats['gpu_inference']:
                avg_gpu = np.mean(timing_stats['gpu_inference'])
                print(f"  GPU推理: {avg_gpu:.3f}s")
            if timing_stats['assemble']:
                avg_assemble = np.mean(timing_stats['assemble'])
                print(f"  数据组装: {avg_assemble:.3f}s")
            total_avg = sum(timing_stats['gpu_inference']) + sum(timing_stats['assemble'])
            if total_avg > 0:
                print(f"  总计: {total_avg/success_count:.3f}s")
        print(f"{'='*60}\n")


# =========================================================================
# 主入口
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SA-1B 通用处理流水线 V5 (Universal Edition - Ultimate Quality)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  1. Full模式 (完整流水线):
     python extract_features_v5.py --mode full \\
        --data-dir /path/to/sa1b \\
        --output-dir /path/to/output \\
        --num-workers 32 --batch-size 4 --shard-size 1024

  2. Resize模式 (仅处理图片):
     python extract_features_v5.py --mode resize_only \\
        --data-dir /path/to/images \\
        --output-dir /path/to/resized \\
        --num-workers 32 --target-size 1024

  3. Pack模式 (仅打包已有文件):
     python extract_features_v5.py --mode pack_only \\
        --data-dir /path/to/npz \\
        --images-dir /path/to/jpg \\
        --output-dir /path/to/output \\
        --num-workers 8 --shard-size 1024
        """
    )
    
    # 核心模式选择
    parser.add_argument("--mode", type=str, default="full",
                       choices=["full", "pack_only", "resize_only"],
                       help="运行模式: full(全流程), pack_only(仅打包现成文件), resize_only(仅处理图片)")
    
    # 通用路径参数
    parser.add_argument("--data-dir", type=str, required=True,
                       help="输入目录 (Full/Resize:原图目录, Pack:NPZ目录)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--images-dir", type=str, default=None,
                       help="[Pack模式专用] JPG图片目录")
    
    # 性能参数
    parser.add_argument("--num-workers", type=int, default=32,
                       help="CPU并行进程数（默认32）")
    parser.add_argument("--shard-size", type=int, default=1024,
                       help="每个Tar包包含的样本数（默认1024）")
    
    # Full模式专用参数
    parser.add_argument("--batch-size", type=int, default=4,
                       help="[Full模式] GPU批大小（默认4）")
    parser.add_argument("--teacher-model", type=str, default="sam2.1_hiera_b+",
                       choices=["sam2.1_hiera_t", "sam2.1_hiera_s", "sam2.1_hiera_b+", "sam2.1_hiera_l"],
                       help="[Full模式] 教师模型类型")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="[Full模式] 权重文件路径")
    parser.add_argument("--device", type=str, default="cuda:4",
                       help="[Full模式] 指定GPU设备 (默认cuda:4)")
    parser.add_argument("--max-images", type=int, default=None,
                       help="[Full模式] 最大处理图像数量")
    parser.add_argument("--diag-compare", action='store_true',
                       help="[Full模式] 启用诊断：保存前对比分布，打印mean/std")
    parser.add_argument("--diag-fallback", action='store_true',
                       help="[Full模式] 启用回退：若std异常则回退为/255实时特征")
    
    # Full模式开关
    parser.add_argument("--save-feature", action="store_true", default=True,
                       help="[Full模式] 总开关: 保存特征（默认True）")
    parser.add_argument("--no-save-feature", action="store_false", dest="save_feature",
                       help="[Full模式] 关闭特征保存")
    parser.add_argument("--save-weights", action="store_true", default=True,
                       help="[Full模式] 总开关: 保存权重图（默认True）")
    parser.add_argument("--no-save-weights", action="store_false", dest="save_weights",
                       help="[Full模式] 关闭权重图保存")
    parser.add_argument("--save-edge", action="store_true", default=True,
                       help="[Full模式] 总开关: 保存边缘图（默认True）")
    parser.add_argument("--no-save-edge", action="store_false", dest="save_edge",
                       help="[Full模式] 关闭边缘图保存")
    parser.add_argument("--save-bbox", action="store_true", default=True,
                       help="[Full模式] 总开关: 保存BBox（默认True）")
    parser.add_argument("--no-save-bbox", action="store_false", dest="save_bbox",
                       help="[Full模式] 关闭BBox保存")
    
    # 层级开关 (Full模式)
    parser.add_argument("--enable-s4", action="store_true", default=False,
                       help="[Full模式] 开启 S4 (256x256) 层级")
    parser.add_argument("--enable-s8", action="store_true", default=False,
                       help="[Full模式] 开启 S8 (128x128) 层级")
    parser.add_argument("--enable-s16", action="store_true", default=True,
                       help="[Full模式] 开启 S16 (64x64) 层级（默认True）")
    parser.add_argument("--enable-s32", action="store_true", default=True,
                       help="[Full模式] 开启 S32 (32x32) 层级（默认True）")
    
    # Resize模式专用参数
    parser.add_argument("--target-size", type=int, default=1024,
                       help="[Resize模式] 目标尺寸（默认1024）")
    
    # 辅助参数 (Full模式)
    parser.add_argument("--kernel-size", type=int, default=3,
                       help="[Full模式] 边缘提取核大小")
    parser.add_argument("--max-instances", type=int, default=1,
                       help="[Full模式] 选框策略最多选择的实例数（默认1）")
    
    args = parser.parse_args()
    
    # 根据模式执行
    if args.mode == "full":
        run_full_pipeline(args)
    elif args.mode == "pack_only":
        run_pack_only(args)
    elif args.mode == "resize_only":
        run_resize_only(args)


if __name__ == "__main__":
    # 必须设置 spawn 启动方式以兼容 PyTorch/OpenCV 多进程
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过了
    
    main()

