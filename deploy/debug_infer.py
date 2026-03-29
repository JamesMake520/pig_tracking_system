"""
调试推理输出：打印模型原始输出的 shape、数值范围和 sigmoid 后的置信度分布
用法: python deploy/debug_infer.py
"""
import sys
import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deploy.atlas_infer import AscendInfer

OM_PATH   = "/root/pig_tracking/inference_model_310B.om"
VIDEO_PATH = "/root/pig_tracking/片段/1-46头.mp4"

def main():
    infer = AscendInfer(OM_PATH, device_id=0)

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    assert ret, "读取视频帧失败"

    print(f"输入帧尺寸: {frame.shape}")

    # 前处理
    inp = infer._preprocess(frame)
    print(f"输入张量: shape={inp.shape}  dtype={inp.dtype}  min={inp.min():.3f}  max={inp.max():.3f}")

    # 推理（拿原始字节输出）
    raw = infer._infer_raw(inp)

    print(f"\n输出数量: {len(raw)}")
    for i, out in enumerate(raw):
        print(f"\n--- 输出[{i}] ---")
        print(f"  字节数: {len(out)}")
        print(f"  float32 元素数: {len(out) // 4}")

        arr = out.view(np.float32)
        print(f"  数值范围: min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")

        # 按 300 个 query 推测 shape
        n_elem = len(arr)
        per_query = n_elem / 300
        print(f"  推测 shape: [300, {per_query:.1f}]")

        if i == 1:  # labels 输出
            # 尝试 reshape 并计算 sigmoid 置信度
            try:
                labels = arr.reshape(300, -1)
                scores = 1.0 / (1.0 + np.exp(-np.clip(labels, -88, 88)))
                conf = scores.max(axis=1)
                print(f"\n  sigmoid 置信度统计:")
                print(f"    max={conf.max():.4f}  min={conf.min():.4f}  mean={conf.mean():.4f}")
                thresholds = [0.1, 0.2, 0.3, 0.5]
                for t in thresholds:
                    print(f"    > {t}: {(conf > t).sum()} 个")
            except Exception as e:
                print(f"  reshape 失败: {e}")

    infer.release()

if __name__ == "__main__":
    main()
