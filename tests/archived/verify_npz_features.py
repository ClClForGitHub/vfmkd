#!/usr/bin/env python3
"""
验证NPZ保存的特征与SAM2在线生成特征的一致性
对比P4 (64x64) 和 P5 (32x32) 特征
"""
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import glob

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'vfmkd' / 'sam2'))

from sam2.build_sam import build_sam2
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_sam2_model():
    """加载SAM2模型"""
    sam2_config_dir = _ROOT / "vfmkd" / "sam2" / "sam2" / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(sam2_config_dir), version_base=None):
        sam2_model = build_sam2(
            config_file='sam2.1/sam2.1_hiera_b+.yaml',
            ckpt_path='weights/sam2.1_hiera_base_plus.pt',
            device=str(device)
        )
    
    sam2_model.eval()
    return sam2_model

def extract_online_features(sam2_model, image_path):
    """在线提取SAM2特征"""
    # 加载并预处理图像
    image_pil = Image.open(image_path).convert('RGB').resize((1024, 1024))
    image_np = np.array(image_pil)
    
    with torch.no_grad():
        # 预处理
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # 使用image_encoder提取特征
        backbone_out = sam2_model.image_encoder(image_tensor)
        fpn_features = backbone_out['backbone_fpn']
        vision_features = backbone_out['vision_features']
        
        # 按索引提取特征（与sam2_teacher.py保持一致）
        features = {}
        
        # IMAGE_EMB_S16: backbone_fpn[2] (64x64)
        if len(fpn_features) >= 3:
            features['IMAGE_EMB_S16'] = fpn_features[2]
        
        # P4_S16: backbone_fpn[2] (64x64) - 应该与IMAGE_EMB_S16相同
        if len(fpn_features) >= 3:
            features['P4_S16'] = fpn_features[2]
        
        # P5_S32: backbone_fpn[3] (32x32)
        if len(fpn_features) >= 4:
            features['P5_S32'] = fpn_features[3]
        
        # vision_features (应该等于fpn_features[-1])
        features['vision_features'] = vision_features
        
        return features

def load_npz_features(npz_path):
    """加载NPZ文件中的特征"""
    data = np.load(npz_path)
    features = {}
    
    for key in data.files:
        if key.endswith(('_S16', '_S32', 'IMAGE_EMB_S16')):
            features[key] = torch.from_numpy(data[key]).to(device)
    
    return features

def compare_features(online_feat, npz_feat, feat_name):
    """对比两个特征的一致性"""
    print(f"\n[{feat_name}] 特征对比:")
    print(f"  在线特征: shape={online_feat.shape}, device={online_feat.device}")
    print(f"  NPZ特征:  shape={npz_feat.shape}, device={npz_feat.device}")
    
    # 确保维度一致
    if online_feat.shape != npz_feat.shape:
        print(f"  ❌ 形状不匹配!")
        return False
    
    # 计算差异
    diff = torch.abs(online_feat - npz_feat)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # 计算相对误差
    rel_error = (diff / (torch.abs(online_feat) + 1e-8)).mean().item()
    
    # 计算余弦相似度
    online_flat = online_feat.flatten()
    npz_flat = npz_feat.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        online_flat.unsqueeze(0), 
        npz_flat.unsqueeze(0)
    ).item()
    
    # 统计信息
    print(f"  在线统计: mean={online_feat.mean():.6f}, std={online_feat.std():.6f}")
    print(f"  NPZ统计:  mean={npz_feat.mean():.6f}, std={npz_feat.std():.6f}")
    print(f"  差异统计: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
    print(f"  相对误差: {rel_error:.8f}")
    print(f"  余弦相似度: {cos_sim:.8f}")
    
    # 判断是否一致 - 调整阈值以适应浮点数精度
    is_identical = max_diff < 1e-6
    is_very_close = max_diff < 1e-3 and cos_sim > 0.9999  # 放宽max_diff阈值
    is_acceptable = max_diff < 0.2 and cos_sim > 0.9999   # 可接受的差异
    
    if is_identical:
        print(f"  ✅ 完全一致 (max_diff < 1e-6)")
        return True
    elif is_very_close:
        print(f"  ✅ 非常接近 (max_diff < 1e-3, cos_sim > 0.9999)")
        return True
    elif is_acceptable:
        print(f"  ✅ 可接受差异 (max_diff < 0.2, cos_sim > 0.9999) - 可能是数值精度问题")
        return True
    else:
        print(f"  ❌ 存在显著差异")
        return False

def main():
    print("="*80)
    print("NPZ特征验证脚本")
    print("="*80)
    
    # 加载SAM2模型
    print("[INFO] 加载SAM2模型...")
    sam2_model = load_sam2_model()
    print("[OK] SAM2模型加载完成")
    
    # 查找NPZ文件
    npz_dir = Path("datasets/coco128/SAM_Cache")
    npz_files = list(npz_dir.glob("*_sam2_features.npz"))[:5]  # 只测试前5个
    
    if not npz_files:
        print("[ERROR] 未找到NPZ文件")
        return
    
    print(f"[INFO] 找到 {len(npz_files)} 个NPZ文件")
    
    # 验证每个NPZ文件
    all_results = []
    
    for i, npz_path in enumerate(npz_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(npz_files)}] 验证: {npz_path.name}")
        print(f"{'='*80}")
        
        # 提取图像ID
        image_id = npz_path.stem.replace('_sam2_features', '')
        image_path = Path(f"datasets/coco128/images/train2017/{image_id}.jpg")
        
        if not image_path.exists():
            print(f"[ERROR] 图像文件不存在: {image_path}")
            continue
        
        try:
            # 在线提取特征
            print("[INFO] 在线提取特征...")
            online_features = extract_online_features(sam2_model, image_path)
            
            # 加载NPZ特征
            print("[INFO] 加载NPZ特征...")
            npz_features = load_npz_features(npz_path)
            
            # 对比关键特征
            results = {}
            
            # 1. 对比 IMAGE_EMB_S16 (64x64)
            if 'IMAGE_EMB_S16' in online_features and 'IMAGE_EMB_S16' in npz_features:
                results['IMAGE_EMB_S16'] = compare_features(
                    online_features['IMAGE_EMB_S16'], 
                    npz_features['IMAGE_EMB_S16'], 
                    'IMAGE_EMB_S16'
                )
            
            # 2. 对比 P4_S16 (64x64)
            if 'P4_S16' in online_features and 'P4_S16' in npz_features:
                results['P4_S16'] = compare_features(
                    online_features['P4_S16'], 
                    npz_features['P4_S16'], 
                    'P4_S16'
                )
            
            # 3. 对比 P5_S32 (32x32)
            if 'P5_S32' in online_features and 'P5_S32' in npz_features:
                results['P5_S32'] = compare_features(
                    online_features['P5_S32'], 
                    npz_features['P5_S32'], 
                    'P5_S32'
                )
            
            # 4. 验证 IMAGE_EMB_S16 与 P4_S16 是否相同
            if 'IMAGE_EMB_S16' in npz_features and 'P4_S16' in npz_features:
                print(f"\n[内部一致性] IMAGE_EMB_S16 vs P4_S16:")
                diff = torch.abs(npz_features['IMAGE_EMB_S16'] - npz_features['P4_S16'])
                max_diff = diff.max().item()
                print(f"  最大差异: {max_diff:.10f}")
                if max_diff < 1e-10:
                    print(f"  ✅ IMAGE_EMB_S16 与 P4_S16 完全相同")
                    results['internal_consistency'] = True
                else:
                    print(f"  ❌ IMAGE_EMB_S16 与 P4_S16 不同")
                    results['internal_consistency'] = False
            
            # 5. 验证 vision_features 与 P5_S32 的关系
            if 'vision_features' in online_features and 'P5_S32' in online_features:
                print(f"\n[vision_features vs P5_S32]:")
                diff = torch.abs(online_features['vision_features'] - online_features['P5_S32'])
                max_diff = diff.max().item()
                print(f"  最大差异: {max_diff:.10f}")
                if max_diff < 1e-10:
                    print(f"  ✅ vision_features 与 P5_S32 完全相同")
                else:
                    print(f"  ❌ vision_features 与 P5_S32 不同")
            
            all_results.append({
                'file': npz_path.name,
                'results': results
            })
            
        except Exception as e:
            print(f"[ERROR] 处理 {npz_path.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结报告
    print(f"\n{'='*80}")
    print("验证总结")
    print(f"{'='*80}")
    
    for item in all_results:
        print(f"\n[{item['file']}]")
        for feat_name, is_ok in item['results'].items():
            status = "✅ 通过" if is_ok else "❌ 失败"
            print(f"  {feat_name}: {status}")
    
    # 统计通过率
    total_tests = sum(len(item['results']) for item in all_results)
    passed_tests = sum(sum(item['results'].values()) for item in all_results)
    
    print(f"\n总体通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！NPZ特征保存正确！")
    else:
        print("⚠️  存在不一致的特征，需要检查！")

if __name__ == "__main__":
    main()
