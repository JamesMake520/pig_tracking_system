"""
Atlas 200I DK A2 批量视频处理

NPU 模型只加载一次，对目录下所有 mp4 逐一运行推理+计数，输出汇总 CSV。

使用方式:
    python deploy/atlas_batch.py \
        --video_dir /root/pig_tracking/videos \
        --om /root/pig_tracking/inference_model.om \
        --output /root/pig_tracking/output

可选参数:
    --conf 0.5          检测置信度阈值
    --out_ratio 0.45    OUT 区占比
    --wait_ratio 0.25   WAIT 区占比
    --no_video          不保存标注视频（节省磁盘）
    --pattern "*.mp4"   视频文件匹配模式（默认 *.mp4）
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "OC_SORT"))

from deploy.atlas_infer import AscendInfer
from deploy.atlas_pipeline import run_pipeline
from deploy.resource_monitor import ResourceMonitor


def main():
    parser = argparse.ArgumentParser(description="Atlas 200I DK A2 批量视频处理")
    parser.add_argument("--video_dir", required=True, help="视频文件目录")
    parser.add_argument("--om", default="/root/pig_tracking/inference_model.om", help="OM 模型路径")
    parser.add_argument("--output", default="/root/pig_tracking/output", help="结果根目录")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--out_ratio", type=float, default=0.45, help="OUT 区占比")
    parser.add_argument("--wait_ratio", type=float, default=0.25, help="WAIT 区占比")
    parser.add_argument("--no_video", action="store_true", help="不保存标注视频")
    parser.add_argument("--pattern", default="*.mp4", help="文件匹配模式（默认 *.mp4）")
    parser.add_argument("--no_monitor", action="store_true", help="关闭资源监控")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    video_files = sorted(video_dir.glob(args.pattern))
    if not video_files:
        print(f"错误: {video_dir} 中未找到匹配 '{args.pattern}' 的视频文件")
        return

    print(f"找到 {len(video_files)} 个视频文件")
    print(f"NPU 模型: {args.om}")
    print("=" * 70)

    # NPU 模型只加载一次
    print("加载 NPU 模型...")
    detector = AscendInfer(args.om, device_id=0)

    # 整批监控（跨所有视频）
    batch_monitor = ResourceMonitor(interval=1.0) if not args.no_monitor else None
    if batch_monitor:
        batch_monitor.start()

    summary = []
    total_start = time.time()

    try:
        for idx, video_path in enumerate(video_files, 1):
            print(f"\n[{idx}/{len(video_files)}] 处理: {video_path.name}")
            print("-" * 70)

            # 每个视频单独输出目录
            video_output_dir = output_root / video_path.stem
            t0 = time.time()
            try:
                valid_count, total_ids = run_pipeline(
                    rtsp_url=str(video_path),
                    output_dir=str(video_output_dir),
                    conf_thresh=args.conf,
                    out_ratio=args.out_ratio,
                    wait_ratio=args.wait_ratio,
                    save_video=not args.no_video,
                    detector=detector,      # 复用已加载的模型
                    enable_monitor=False,   # 单视频监控由整批 monitor 代替
                )
                elapsed = time.time() - t0
                summary.append({
                    "video": video_path.name,
                    "valid_count": valid_count,
                    "total_ids": total_ids,
                    "elapsed_s": f"{elapsed:.1f}",
                    "status": "OK",
                })
                print(f"  有效计数: {valid_count}  总ID: {total_ids}  耗时: {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [失败] {e}")
                summary.append({
                    "video": video_path.name,
                    "valid_count": -1,
                    "total_ids": -1,
                    "elapsed_s": f"{elapsed:.1f}",
                    "status": f"ERROR: {e}",
                })

    finally:
        detector.release()
        if batch_monitor:
            batch_stats = batch_monitor.stop()
            ResourceMonitor.print_summary(batch_stats)

    # 汇总报告
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("批处理完成")
    print("=" * 70)
    print(f"{'视频文件':<30} {'有效计数':>8} {'总ID':>6} {'耗时(s)':>8} {'状态'}")
    print("-" * 70)
    for r in summary:
        print(f"{r['video']:<30} {r['valid_count']:>8} {r['total_ids']:>6} {r['elapsed_s']:>8}  {r['status']}")
    print("=" * 70)
    print(f"总耗时: {total_elapsed/60:.1f} 分钟  均耗时: {total_elapsed/len(video_files):.1f} 秒/视频")

    # 保存汇总 CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_root / f"batch_summary_{ts}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "valid_count", "total_ids", "elapsed_s", "status"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n汇总 CSV: {csv_path}")


if __name__ == "__main__":
    main()
