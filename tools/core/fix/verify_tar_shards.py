#!/usr/bin/env python3
"""
验证 tar shard 的完整性与覆盖度：
1) 逐个 shard 校验：NPZ 与 JPG 是否一一对应；读取异常标记为无效
2) 汇总统计：有效/无效 shard 数、总样本数（按 image_id 去重）
3) 覆盖度校验（可选）：与原始目录对比，确保每个 *_features.npz 与 *.jpg 都出现在 shards 中

用法示例：
  仅校验 shard 内部一致性
    python tools/core/fix/verify_tar_shards.py \
      --shard-dir /home/.../sa1b_tar_shards

  同时对比原始目录覆盖度（推荐）
    python tools/core/fix/verify_tar_shards.py \
      --shard-dir /home/.../sa1b_tar_shards \
      --features-dir /home/.../sa1b/extracted \
      --images-dir  /home/.../sa1b_resized_1024 \
      --report /home/.../sa1b_tar_shards/verification_report.txt
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tqdm import tqdm


def _safe_open_tar(path: Path) -> tarfile.TarFile | None:
    try:
        # 自动识别是否压缩（.tar / .tar.gz / .tar.xz / .tar.bz2）
        return tarfile.open(path, mode="r:*")
    except Exception:
        return None


def _collect_ids_from_tar(tar: tarfile.TarFile) -> Tuple[Set[str], Set[str]]:
    """
    从 tar 中提取 image_id 集合：
    - npz_image_ids: 由 *_features.npz 提取（去掉后缀与 _features）
    - jpg_image_ids: 由 *.jpg 提取（basename，确保以 sa_ 开头）
    """
    npz_ids: Set[str] = set()
    jpg_ids: Set[str] = set()

    for m in tar:
        name = m.name
        if not name:
            continue
        if name.endswith("_features.npz"):
            image_id = Path(name).stem.replace("_features", "")
            if image_id:
                npz_ids.add(image_id)
        elif name.lower().endswith(".jpg"):
            base = Path(name).stem
            # 规范：jpg 的 image_id 形如 "sa_xxx"
            if base.startswith("sa_"):
                jpg_ids.add(base)
    return npz_ids, jpg_ids


def verify_single_shard(shard_path: Path) -> Dict:
    """
    校验单个 shard：
    - 可读性（能否完整遍历）
    - NPZ/JPG 是否匹配
    """
    result = {
        "shard": shard_path.name,
        "valid": False,
        "npz_count": 0,
        "jpg_count": 0,
        "missing_jpg": [],   # 存在 NPZ、缺 JPG 的 image_id
        "missing_npz": [],   # 存在 JPG、缺 NPZ 的 image_id
        "error": "",
    }

    tar = _safe_open_tar(shard_path)
    if tar is None:
        result["error"] = "failed to open (corrupted or under writing)"
        return result

    try:
        npz_ids, jpg_ids = _collect_ids_from_tar(tar)
        result["npz_count"] = len(npz_ids)
        result["jpg_count"] = len(jpg_ids)

        miss_jpg = sorted(npz_ids - jpg_ids)
        miss_npz = sorted(jpg_ids - npz_ids)
        result["missing_jpg"] = miss_jpg
        result["missing_npz"] = miss_npz
        result["valid"] = (len(miss_jpg) == 0 and len(miss_npz) == 0)
        return result
    except Exception as e:
        result["error"] = f"read error: {e}"
        return result
    finally:
        try:
            tar.close()
        except Exception:
            pass


def verify_all_shards(shard_dir: Path) -> Dict:
    shard_dir = Path(shard_dir)
    shard_files = sorted(list(shard_dir.glob("sa1b_shard_*.tar*")))
    if not shard_files:
        raise FileNotFoundError(f"在 {shard_dir} 下没有找到 shard（sa1b_shard_*.tar*）")

    all_npz_ids: Set[str] = set()
    all_jpg_ids: Set[str] = set()
    results: List[Dict] = []

    for shard_path in tqdm(shard_files, desc="验证 shards", unit="tar"):
        r = verify_single_shard(shard_path)
        results.append(r)
        # 若 shard 有缺失，不汇总其 IDs，以避免统计被污染
        if r["valid"]:
            # 再次打开提取 IDs（避免在 verify_single_shard 里保留大集合）
            tar = _safe_open_tar(shard_path)
            if tar is not None:
                try:
                    npz_ids, jpg_ids = _collect_ids_from_tar(tar)
                    all_npz_ids.update(npz_ids)
                    all_jpg_ids.update(jpg_ids)
                finally:
                    try:
                        tar.close()
                    except Exception:
                        pass

    valid_count = sum(1 for r in results if r["valid"])
    invalid_count = len(results) - valid_count

    return {
        "total_shards": len(results),
        "valid_shards": valid_count,
        "invalid_shards": invalid_count,
        "results": results,
        "all_npz_ids": all_npz_ids,
        "all_jpg_ids": all_jpg_ids,
    }


def scan_source_ids(features_dir: Path | None, images_dir: Path | None) -> Tuple[Set[str], Set[str]]:
    """
    扫描原始目录：
    - NPZ: *_features.npz → image_id
    - JPG: *.jpg → sa_<normalized>
    """
    src_npz_ids: Set[str] = set()
    src_jpg_ids: Set[str] = set()

    if features_dir:
        for p in tqdm(sorted(Path(features_dir).glob("*_features.npz")), desc="扫描 NPZ", unit="npz", leave=False):
            src_npz_ids.add(p.stem.replace("_features", ""))
    if images_dir:
        for p in tqdm(sorted(Path(images_dir).glob("*.jpg")), desc="扫描 JPG", unit="jpg", leave=False):
            stem = p.stem.strip()
            if not stem:
                continue
            if stem.startswith("sa_"):
                normalized = stem[3:]
            else:
                normalized = stem
            if not normalized:
                continue
            src_jpg_ids.add(f"sa_{normalized}")

    return src_npz_ids, src_jpg_ids


def write_report(report_path: Path, shard_summary: Dict, cov: Dict | None) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Tar Shard 验证报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"总 Shard 数: {shard_summary['total_shards']}\n")
        f.write(f"有效 Shard 数: {shard_summary['valid_shards']}\n")
        f.write(f"无效 Shard 数: {shard_summary['invalid_shards']}\n\n")

        if cov is not None:
            f.write("覆盖度对比\n")
            f.write("-" * 60 + "\n")
            f.write(f"源 NPZ 总数: {cov['source_npz_total']}\n")
            f.write(f"源 JPG 总数: {cov['source_jpg_total']}\n")
            f.write(f"Shards 中 NPZ 总数: {cov['shard_npz_total']}\n")
            f.write(f"Shards 中 JPG 总数: {cov['shard_jpg_total']}\n")
            f.write(f"缺失（源有、shard 无）NPZ 数: {len(cov['missing_npz_in_shards'])}\n")
            f.write(f"缺失（源有、shard 无）JPG 数: {len(cov['missing_jpg_in_shards'])}\n\n")

            if cov["missing_npz_in_shards"]:
                f.write("缺失的 NPZ image_id（部分）:\n")
                for x in list(sorted(cov["missing_npz_in_shards"]))[:200]:
                    f.write(f"  {x}\n")
                f.write("\n")
            if cov["missing_jpg_in_shards"]:
                f.write("缺失的 JPG image_id（部分）:\n")
                for x in list(sorted(cov["missing_jpg_in_shards"]))[:200]:
                    f.write(f"  {x}\n")
                f.write("\n")

        # 列出无效 shard 及原因
        bad = [r for r in shard_summary["results"] if not r["valid"]]
        if bad:
            f.write("无效 Shard 详情\n")
            f.write("-" * 60 + "\n")
            for r in bad[:200]:
                f.write(f"{r['shard']}: npz={r['npz_count']} jpg={r['jpg_count']} err={r.get('error','')}\n")
                if r["missing_jpg"][:10]:
                    f.write(f"  缺失 JPG 示例: {r['missing_jpg'][:10]}\n")
                if r["missing_npz"][:10]:
                    f.write(f"  缺失 NPZ 示例: {r['missing_npz'][:10]}\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="验证 tar shards 完整性与覆盖度")
    parser.add_argument("--shard-dir", type=Path, required=True, help="shard 目录（含 sa1b_shard_*.tar*）")
    parser.add_argument("--features-dir", type=Path, default=None, help="原始 *_features.npz 目录（可选）")
    parser.add_argument("--images-dir", type=Path, default=None, help="原始 *.jpg 目录（可选）")
    parser.add_argument("--report", type=Path, default=None, help="输出报告路径（可选）")
    args = parser.parse_args()

    # 1) 校验 shards
    shard_summary = verify_all_shards(args.shard_dir)
    print(f"\nShards 概览：")
    print(f"  总数: {shard_summary['total_shards']}, 有效: {shard_summary['valid_shards']}, 无效: {shard_summary['invalid_shards']}")
    if shard_summary["invalid_shards"] > 0:
        print("  注意：存在无效/正在写入中的 shard（读取失败或 NPZ/JPG 不匹配）")

    # 2) 覆盖度对比（可选）
    cov = None
    if args.features_dir or args.images_dir:
        src_npz_ids, src_jpg_ids = scan_source_ids(args.features_dir, args.images_dir)
        shard_npz_ids = shard_summary["all_npz_ids"]
        shard_jpg_ids = shard_summary["all_jpg_ids"]

        missing_npz = sorted(src_npz_ids - shard_npz_ids)
        missing_jpg = sorted(src_jpg_ids - shard_jpg_ids)

        cov = {
            "source_npz_total": len(src_npz_ids),
            "source_jpg_total": len(src_jpg_ids),
            "shard_npz_total": len(shard_npz_ids),
            "shard_jpg_total": len(shard_jpg_ids),
            "missing_npz_in_shards": missing_npz,
            "missing_jpg_in_shards": missing_jpg,
        }

        print("\n覆盖度对比：")
        print(f"  源 NPZ: {cov['source_npz_total']}, Shards NPZ: {cov['shard_npz_total']}, 缺失: {len(missing_npz)}")
        print(f"  源 JPG: {cov['source_jpg_total']}, Shards JPG: {cov['shard_jpg_total']}, 缺失: {len(missing_jpg)}")
        if missing_npz:
            print(f"  缺失 NPZ 示例: {missing_npz[:10]}")
        if missing_jpg:
            print(f"  缺失 JPG 示例: {missing_jpg[:10]}")

    # 3) 报告
    if args.report:
        write_report(args.report, shard_summary, cov)
        print(f"\n报告已写入: {args.report}")


if __name__ == "__main__":
    main()


