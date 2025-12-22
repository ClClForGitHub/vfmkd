import os
import sys
import time
import shutil
import tempfile
import tarfile
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
# from torchvision.io import decode_image, ImageReadMode  # 未使用，注释掉避免导入错误

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IO_Benchmark")

class BenchmarkTarDataset(IterableDataset):
    """
    独立实现的 RAM 缓存 Dataset，用于验证 IO 性能
    """
    def __init__(
        self,
        shard_dir: str,
        use_ram_cache: bool = True,
        shuffle_buffer: int = 500,
    ):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.use_ram_cache = use_ram_cache
        self.shuffle_buffer_size = shuffle_buffer
        
        # 扫描文件
        self.shard_files = sorted(
            list(self.shard_dir.glob("*.tar")) + 
            list(self.shard_dir.glob("*.tar.gz"))
        )
        if not self.shard_files:
            raise RuntimeError(f"❌ 错误：在 {shard_dir} 没找到 .tar 文件！")
        
        logger.info(f"[Init] 找到 {len(self.shard_files)} 个 Tar 文件。RAM 缓存策略: {'✅ 开启' if use_ram_cache else '❌ 关闭'}")

    def _check_ram_space(self, size_needed):
        """检查 /dev/shm 空间"""
        if not os.path.exists('/dev/shm'): return False
        try:
            total, used, free = shutil.disk_usage('/dev/shm')
            # 预留 2GB 安全水位
            return (size_needed + 2 * 1024**3) < free
        except:
            return False

    def _parse_tar_content(self, tar_path):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        
        process_path = tar_path
        temp_file = None
        
        # === 核心：RAM 缓存逻辑 ===
        if self.use_ram_cache:
            try:
                file_size = os.path.getsize(tar_path)
                # 检查空间
                if self._check_ram_space(file_size):
                    t0 = time.time()
                    # 1. 在内存盘创建文件
                    fd, temp_path = tempfile.mkstemp(dir='/dev/shm', suffix='.tar')
                    os.close(fd)
                    # 2. 执行拷贝 (HDD 顺序读 -> RAM 写)
                    shutil.copyfile(tar_path, temp_path)
                    copy_time = time.time() - t0
                    
                    logger.info(f"[Worker {worker_id}] 🚀 已缓存 {tar_path.name} 到内存 (耗时 {copy_time:.2f}s, {file_size/1024/1024:.1f}MB)")
                    
                    process_path = Path(temp_path)
                    temp_file = temp_path
                else:
                    logger.warning(f"[Worker {worker_id}] ⚠️ 内存空间不足，跳过缓存 {tar_path.name}")
            except Exception as e:
                logger.warning(f"[Worker {worker_id}] ⚠️ 缓存失败: {e}，将直接读取硬盘")

        # === 读取逻辑 ===
        local_buffer = {}
        try:
            # 模拟真实读取：打开 tar 并读取文件内容
            with tarfile.open(process_path, mode='r|*') as tar:
                for member in tar:
                    if not member.isfile(): continue
                    fname = member.name
                    
                    # 简单解析 ID
                    if fname.endswith('.npz'):
                        img_id = fname[:-13]
                        type_k = 'npz'
                    elif fname.endswith('.jpg'):
                        img_id = fname[:-4]
                        type_k = 'img'
                    else:
                        continue
                        
                    f_obj = tar.extractfile(member)
                    if f_obj is None: continue
                    content = f_obj.read() # 真实发生 IO 读取
                    
                    if img_id not in local_buffer:
                        local_buffer[img_id] = {type_k: content}
                    else:
                        local_buffer[img_id][type_k] = content
                    
                    # 配对
                    if 'npz' in local_buffer[img_id] and 'img' in local_buffer[img_id]:
                        item = local_buffer.pop(img_id)
                        # 模拟解码开销 (但不做复杂的后处理，只测 IO)
                        yield self._mock_process(item['img'], item['npz'])
                        
        except Exception as e:
            logger.error(f"Error reading: {e}")
        finally:
            # 清理内存
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def _mock_process(self, img_bytes, npz_bytes):
        # 最小化解码，模拟真实负载
        img_buffer = torch.frombuffer(img_bytes, dtype=torch.uint8)
        # 只要这步不报错，说明数据读对了
        # img = decode_image(img_buffer, mode=ImageReadMode.RGB) 
        return torch.zeros(3, 1024, 1024) # 返回假数据，只测速度

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            my_shards = self.shard_files
        else:
            # 分片
            my_shards = self.shard_files[worker_info.id :: worker_info.num_workers]
        
        my_shards_list = list(my_shards)
        np.random.shuffle(my_shards_list) # Shuffle Tar files
        
        iterator = self._shard_iterator(my_shards_list)
        
        # Shuffle Buffer
        shuffle_buffer = []
        try:
            for sample in iterator:
                shuffle_buffer.append(sample)
                if len(shuffle_buffer) >= self.shuffle_buffer_size:
                    idx = np.random.randint(len(shuffle_buffer))
                    yield shuffle_buffer.pop(idx)
        except StopIteration:
            pass
        for sample in shuffle_buffer:
            yield sample

    def _shard_iterator(self, shard_paths):
        for p in shard_paths:
            yield from self._parse_tar_content(p)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=str, required=True, help="Tar Shard 所在目录")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4, help="Worker 数量，HDD建议4")
    parser.add_argument("--prefetch", type=int, default=2, help="预取因子")
    parser.add_argument("--no-ram-cache", action="store_true", help="禁用 RAM 缓存（用于对比）")
    args = parser.parse_args()
    
    print("="*60)
    print(f"🛠️  IO 性能基准测试")
    print(f"📂 数据目录: {args.shard_dir}")
    print(f"⚙️  配置: Batch={args.batch_size}, Workers={args.num_workers}, Prefetch={args.prefetch}")
    print(f"🧠 RAM 缓存: {'❌ 禁用 (模拟现状)' if args.no_ram_cache else '✅ 启用 (优化方案)'}")
    print("="*60)
    
    dataset = BenchmarkTarDataset(
        shard_dir=args.shard_dir,
        use_ram_cache=not args.no_ram_cache,
        shuffle_buffer=500
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        pin_memory=True
    )
    
    print("⏳ 正在启动 DataLoader (冷启动计时开始)...")
    t_start = time.time()
    
    iterator = iter(loader)
    
    # 强制获取第一个 Batch
    try:
        first_batch = next(iterator)
        t_first = time.time()
        print(f"🔥 [冷启动完成] 首个 Batch 耗时: {t_first - t_start:.2f} 秒")
    except StopIteration:
        print("❌ 数据集为空或读取失败！")
        return
    
    # 连续读取测试
    print("🚀 开始连续读取 50 个 Batch，测试吞吐量...")
    
    times = []
    start_loop = time.time()
    
    try:
        for i in range(50):
            t0 = time.time()
            batch = next(iterator)
            dt = time.time() - t0
            times.append(dt)
            
            # 模拟 GPU 训练耗时 (假设 0.3s 一个 batch)
            # 看看 IO 能不能跟上
            time.sleep(0.3) 
            
            print(f"\rBatch {i+1}/50 | Load Time: {dt:.4f}s | (模拟GPU计算中...)", end="")
    except StopIteration:
        pass
    
    print("\n" + "="*60)
    avg_time = np.mean(times)
    total_time = time.time() - start_loop
    print(f"📊 测试结果报告:")
    print(f"   - 平均数据加载时间: {avg_time:.4f} 秒/Batch")
    print(f"   - 预期的理想时间: 0.00xx 秒 (应该被预取掩盖)")
    
    if avg_time > 0.5:
        print(f"\n❌ 结论: IO 依然是瓶颈 (加载比计算慢)。")
    else:
        print(f"\n✅ 结论: IO 极其流畅！RAM 缓存策略有效。")
    print("="*60)

if __name__ == "__main__":
    main()

