#!/usr/bin/env python3
"""
批量处理视频文件
自动处理 input_videos 目录下的所有 .mp4 视频文件
使用 ByteTrack 进行追踪和计数
"""
import subprocess
import sys
from pathlib import Path
import time

# 配置 - 请根据实际情况修改这些路径
CURRENT_DIR = Path(__file__).parent.parent  # 项目根目录
VIDEO_DIR = CURRENT_DIR / "input_videos"
OUTPUT_BASE_DIR = CURRENT_DIR / "output"
MODEL_PATH = CURRENT_DIR / "models" / "checkpoint_best_ema.pth"
SCRIPT_PATH = CURRENT_DIR / "scripts" / "compare_trackers.py"

def process_video(video_path):
    """处理单个视频"""
    video_name = video_path.stem  # 获取不带扩展名的文件名
    output_dir = OUTPUT_BASE_DIR / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"处理视频: {video_path.name}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")

    # 构建命令
    cmd = [
        sys.executable,  # Python解释器
        str(SCRIPT_PATH),
        "--video_path", str(video_path),
        "--model_path", str(MODEL_PATH),
        "--output_dir", str(output_dir),
        "--no_timestamp"
    ]

    # 运行命令
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        elapsed = time.time() - start_time
        print(f"\n[成功] 完成: {video_path.name} (耗时: {elapsed:.1f}秒)")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n[失败] 失败: {video_path.name} (耗时: {elapsed:.1f}秒)")
        print(f"错误信息: {e}")
        return False, elapsed

def main():
    # 获取所有mp4视频
    video_files = sorted(VIDEO_DIR.glob("*.mp4"))

    if not video_files:
        print(f"错误: 在 {VIDEO_DIR} 中没有找到视频文件")
        return

    print(f"\n找到 {len(video_files)} 个视频文件")
    print("="*80)

    # 处理统计
    success_count = 0
    fail_count = 0
    total_time = 0
    results = []

    # 逐个处理视频
    for i, video_path in enumerate(video_files, 1):
        print(f"\n进度: [{i}/{len(video_files)}]")
        success, elapsed = process_video(video_path)

        total_time += elapsed
        results.append({
            'name': video_path.name,
            'success': success,
            'time': elapsed
        })

        if success:
            success_count += 1
        else:
            fail_count += 1

    # 打印总结
    print("\n" + "="*80)
    print("批处理完成!")
    print("="*80)
    print(f"总视频数: {len(video_files)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"平均耗时: {total_time/len(video_files):.1f} 秒/视频")

    # 打印详细结果
    print("\n详细结果:")
    print("-"*80)
    for res in results:
        status = "[成功]" if res['success'] else "[失败]"
        print(f"{status} {res['name']:<30} {res['time']:.1f}秒")

    print(f"\n所有结果已保存到: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()
