"""
Atlas 200I DK A2 实时猪只计数 Pipeline

流程: RTSP 摄像头 → NPU 检测 (RF-DETR OM) → ByteTrack (ARM CPU) → ZoneAnalyzer → 计数

使用方式:
    python atlas_pipeline.py \
        --rtsp "rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0" \
        --om  /root/pig_tracking/inference_model.om \
        --output /root/pig_tracking/output

依赖（Atlas 上安装）:
    pip install opencv-python-headless numpy scipy
    # ByteTrack 代码在项目 trackers/ 目录中（纯 Python/numpy/scipy）
"""

import argparse
import sys
import csv
import time
import cv2
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
from collections import defaultdict

# ---- 将项目根目录和 OC_SORT 加入 sys.path ----
# 假设本文件位于 <project_root>/deploy/atlas_pipeline.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "OC_SORT"))

from deploy.atlas_infer import AscendInfer
from deploy.resource_monitor import ResourceMonitor


# ===========================================================================
# 区域分析（从 scripts/compare_trackers.py 提取，去除 rfdetr 依赖）
# ===========================================================================

class PigTrajectory:
    """跟踪单只猪的轨迹和状态变化"""

    def __init__(self, track_id, first_frame, first_zone):
        self.track_id = track_id
        self.zone_history = [first_zone]
        self.state_changes = []
        self.first_frame = first_frame
        self.last_frame = first_frame
        self.status = "ACTIVE"
        self.positions = []
        self.box_sizes = []
        self.confidences = []
        self.lost_frames = []
        self.recovered_count = 0
        self.last_cx = None

    def add_point(self, zone, frame_idx, fps, cx=None, cy=None, w=None, h=None, conf=None):
        if frame_idx - self.last_frame > 1:
            for lost_f in range(self.last_frame + 1, frame_idx):
                self.lost_frames.append(lost_f)
            self.recovered_count += 1
        self.last_frame = frame_idx
        if cx is not None:
            self.positions.append((cx, cy, frame_idx))
            self.last_cx = cx
        if w is not None:
            self.box_sizes.append((w, h, frame_idx))
        if conf is not None:
            self.confidences.append((conf, frame_idx))
        if self.zone_history[-1] != zone:
            timestamp = frame_idx / fps
            self.state_changes.append({
                'frame': frame_idx,
                'from_zone': self.zone_history[-1],
                'to_zone': zone,
                'timestamp': f"{timestamp:.2f}s"
            })
            self.zone_history.append(zone)

    def get_travel_distance(self):
        if len(self.positions) < 2:
            return 0
        total = 0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i - 1][0]
            dy = self.positions[i][1] - self.positions[i - 1][1]
            total += (dx ** 2 + dy ** 2) ** 0.5
        return total

    def get_avg_speed(self, fps):
        dist = self.get_travel_distance()
        duration = (self.last_frame - self.first_frame) / fps
        return dist / duration if duration > 0 else 0

    def get_avg_confidence(self):
        if not self.confidences:
            return 0
        return sum(c[0] for c in self.confidences) / len(self.confidences)

    def analyze(self):
        history = self.zone_history
        if not history:
            return False, "Empty"
        if history[0] != 'ENTRY':
            return False, f"Ghost (Started in {history[0]})"
        if 'OUT' not in history:
            last_zone = history[-1]
            if last_zone == 'ENTRY':
                return False, "Turned Back"
            elif last_zone == 'WAIT':
                return False, "Stuck in Wait"
            return False, "Incomplete"
        try:
            last_entry_idx = len(history) - 1 - history[::-1].index('ENTRY')
            final_attempt = history[last_entry_idx:]
            if 'WAIT' in final_attempt:
                first_wait = final_attempt.index('WAIT')
                if 'OUT' in final_attempt[first_wait:]:
                    if last_entry_idx > 0:
                        return True, "Valid (Retried)"
                    first_out = final_attempt.index('OUT', first_wait)
                    if 'WAIT' in final_attempt[first_out:]:
                        return True, "Valid (Hesitated)"
                    return True, "Valid Sequence"
            return False, "Jumped Over"
        except ValueError:
            return False, "Logic Error"


class ZoneAnalyzer:
    """区域分析器（三区域：ENTRY / WAIT / OUT）"""

    def __init__(self, width, fps, out_ratio=0.45, wait_ratio=0.25):
        self.width = width
        self.fps = fps
        self.out_ratio = out_ratio
        self.wait_ratio = wait_ratio
        self.split_0 = width * (out_ratio / 2)
        self.split_1 = width * out_ratio
        self.split_2 = width * (out_ratio + wait_ratio)

        self.trajectories = {}
        self.valid_count = 0
        self.id_events = []
        self.line_counters = {'line0': 0, 'line1': 0, 'line2': 0}
        self.prev_positions = {}

    def get_zone(self, cx):
        if cx < self.split_1:
            return 'OUT'
        elif cx < self.split_2:
            return 'WAIT'
        else:
            return 'ENTRY'

    def update(self, track_id, cx, frame_idx, cy=None, w=None, h=None, conf=None):
        zone = self.get_zone(cx)

        if track_id in self.prev_positions:
            prev_cx = self.prev_positions[track_id]
            if prev_cx > self.split_0 >= cx:
                self.line_counters['line0'] += 1
            elif prev_cx < self.split_0 <= cx:
                self.line_counters['line0'] -= 1
            if prev_cx > self.split_1 >= cx:
                self.line_counters['line1'] += 1
            elif prev_cx < self.split_1 <= cx:
                self.line_counters['line1'] -= 1
            if prev_cx > self.split_2 >= cx:
                self.line_counters['line2'] += 1
            elif prev_cx < self.split_2 <= cx:
                self.line_counters['line2'] -= 1

        self.prev_positions[track_id] = cx

        if track_id not in self.trajectories:
            self.trajectories[track_id] = PigTrajectory(track_id, frame_idx, zone)
            self.id_events.append({
                'frame': frame_idx,
                'timestamp': f"{frame_idx / self.fps:.2f}s",
                'event': 'NEW_ID',
                'track_id': track_id,
                'zone': zone,
                'details': f"ID {track_id} appeared in {zone}"
            })
        else:
            old_zone = self.trajectories[track_id].zone_history[-1]
            self.trajectories[track_id].add_point(zone, frame_idx, self.fps, cx, cy, w, h, conf)
            if old_zone != zone:
                self.id_events.append({
                    'frame': frame_idx,
                    'timestamp': f"{frame_idx / self.fps:.2f}s",
                    'event': 'ZONE_CHANGE',
                    'track_id': track_id,
                    'zone': zone,
                    'details': f"ID {track_id}: {old_zone} -> {zone}"
                })

    def finalize(self, output_dir: Path, tracker_name: str) -> int:
        output_dir = Path(output_dir)

        # 三条线均值四舍五入 → 最终计数
        line_avg = (self.line_counters['line0'] +
                    self.line_counters['line1'] +
                    self.line_counters['line2']) / 3.0
        final_count = int(line_avg)

        # ID 事件 CSV
        events_path = output_dir / f"{tracker_name}_id_events.csv"
        with open(events_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'timestamp', 'event', 'track_id', 'zone', 'details'])
            writer.writeheader()
            writer.writerows(self.id_events)

        # 状态变化 TXT
        txt_path = output_dir / f"{tracker_name}_state_changes.txt"
        traj_valid_count = 0
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 70}\nTracker: {tracker_name}\n{'=' * 70}\n\n")
            for tid in sorted(self.trajectories.keys()):
                traj = self.trajectories[tid]
                is_valid, reason = traj.analyze()
                status = "VALID" if is_valid else "INVALID"
                f.write(f"ID {tid}: {status} ({reason})\n")
                f.write(f"  首次出现: Frame {traj.first_frame} ({traj.first_frame / self.fps:.2f}s)\n")
                f.write(f"  最后出现: Frame {traj.last_frame} ({traj.last_frame / self.fps:.2f}s)\n")
                f.write(f"  区域路径: {' -> '.join(traj.zone_history)}\n")
                if traj.state_changes:
                    f.write("  状态变化:\n")
                    for change in traj.state_changes:
                        f.write(f"    [{change['timestamp']}] Frame {change['frame']}: {change['from_zone']} -> {change['to_zone']}\n")
                f.write(f"  移动距离: {traj.get_travel_distance():.1f} px\n")
                f.write(f"  平均速度: {traj.get_avg_speed(self.fps):.1f} px/s\n")
                f.write(f"  平均置信度: {traj.get_avg_confidence():.2f}\n")
                f.write("\n")
                if is_valid:
                    traj_valid_count += 1

            f.write(f"{'=' * 70}\n")
            f.write(f"Line0: {self.line_counters['line0']}  "
                    f"Line1: {self.line_counters['line1']}  "
                    f"Line2: {self.line_counters['line2']}  "
                    f"均值: {line_avg:.2f}\n")
            f.write(f"FINAL COUNT (三线均值向下取整): {final_count}\n")
            f.write(f"TOTAL IDs: {len(self.trajectories)}\n")
            f.write(f"{'=' * 70}\n")

        # 轨迹报告 CSV
        report_path = output_dir / f"{tracker_name}_trajectory_report.csv"
        with open(report_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['TrackID', 'IsValid', 'Reason', 'FirstFrame', 'FirstTime(s)',
                             'LastFrame', 'LastTime(s)', 'Duration(s)', 'ZoneHistory'])
            for tid in sorted(self.trajectories.keys()):
                traj = self.trajectories[tid]
                is_valid, reason = traj.analyze()
                duration = (traj.last_frame - traj.first_frame) / self.fps
                writer.writerow([
                    tid, is_valid, reason,
                    traj.first_frame, f"{traj.first_frame / self.fps:.2f}",
                    traj.last_frame, f"{traj.last_frame / self.fps:.2f}",
                    f"{duration:.2f}", "->".join(traj.zone_history)
                ])
            # 汇总行
            writer.writerow([])
            writer.writerow(['FINAL_COUNT', final_count,
                             f'Line0={self.line_counters["line0"]} Line1={self.line_counters["line1"]} Line2={self.line_counters["line2"]} avg={line_avg:.2f}',
                             '', '', '', '', '', ''])

        self.valid_count = final_count
        print(f"  报告已保存至: {output_dir}")
        return final_count


def is_blue_object(frame, bbox, blue_threshold=1.3):
    """检测物体是否为蓝色（过滤蓝色干扰物）"""
    x1, y1, x2, y2 = map(int, bbox[:4])
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return False
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    b_mean = np.mean(roi[:, :, 0])
    g_mean = np.mean(roi[:, :, 1])
    r_mean = np.mean(roi[:, :, 2])
    rg_mean = (r_mean + g_mean) / 2.0
    return b_mean > rg_mean * blue_threshold and b_mean > 80


# ===========================================================================
# ByteTrack 导入（纯 Python/numpy/scipy，兼容 ARM）
# ===========================================================================

def _load_bytetrack():
    try:
        from trackers.byte_tracker.byte_tracker import BYTETracker
        return BYTETracker
    except ImportError as e:
        raise ImportError(
            f"ByteTrack 导入失败: {e}\n"
            "请确认 trackers/byte_tracker/ 目录已复制到 Atlas，\n"
            "并已安装依赖: pip install scipy"
        )


# ===========================================================================
# RTSP 打开
# ===========================================================================

def open_rtsp(rtsp_url: str, retry: int = 5) -> cv2.VideoCapture:
    # 本地文件路径直接打开，不走重试逻辑
    is_local = not rtsp_url.startswith("rtsp://") and not rtsp_url.startswith("rtsp:/")
    if is_local:
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            print(f"[Pipeline] 本地文件打开成功: {rtsp_url}")
            return cap
        raise RuntimeError(f"无法打开本地文件: {rtsp_url}")

    for attempt in range(retry):
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # 减少缓冲延迟
            print(f"[Pipeline] RTSP 连接成功: {rtsp_url}")
            return cap
        print(f"[Pipeline] RTSP 连接失败（{attempt + 1}/{retry}），重试...")
        time.sleep(2)
    raise RuntimeError(f"无法打开 RTSP 流: {rtsp_url}")


# ===========================================================================
# 主 Pipeline
# ===========================================================================

def run_pipeline(
    rtsp_url: str,
    om_path: str = "",
    output_dir: str = "/root/pig_tracking/output",
    conf_thresh: float = 0.5,
    out_ratio: float = 0.45,
    wait_ratio: float = 0.25,
    save_video: bool = True,
    max_frames: int = 0,
    detector=None,          # 传入已有 AscendInfer 实例可复用，不重复加载模型
    enable_monitor: bool = True,  # 是否启用资源监控
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 初始化 NPU 检测器（未传入时自行创建）
    _own_detector = detector is None
    if _own_detector:
        print("[Pipeline] 加载 NPU 模型...")
        detector = AscendInfer(om_path, device_id=0)

    # 2. 打开 RTSP 流
    cap = open_rtsp(rtsp_url)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0
    print(f"[Pipeline] 流分辨率: {width}x{height}  FPS: {fps:.1f}")

    # 3. 初始化 ByteTrack
    BYTETracker = _load_bytetrack()
    byte_args = SimpleNamespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False)
    tracker = BYTETracker(byte_args, frame_rate=int(fps))

    # 4. 初始化区域分析器
    analyzer = ZoneAnalyzer(width, fps, out_ratio, wait_ratio)

    # 5. 视频写入（可选）
    writer = None
    if save_video:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(output_dir / f"atlas_output_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[Pipeline] 输出视频: {out_path}")

    is_local = not rtsp_url.startswith("rtsp://")
    frame_idx = 0
    print("[Pipeline] 开始运行...（Ctrl+C 停止）")

    # 资源监控
    monitor = ResourceMonitor(interval=1.0) if enable_monitor else None
    if monitor:
        monitor.start()
    _last_stats_time = time.time()
    _STATS_INTERVAL = 10.0  # 每 10 秒打印一次实时资源

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_local:
                    print("[Pipeline] 本地文件播放完毕。")
                    break
                print("[Pipeline] RTSP 流断开，尝试重连...")
                cap.release()
                try:
                    cap = open_rtsp(rtsp_url, retry=3)
                    continue
                except RuntimeError:
                    print("[Pipeline] 重连失败，退出。")
                    break

            # ---- NPU 检测 ----
            results = detector.predict(frame, conf_thresh=conf_thresh)
            detections = []
            if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                for box, score in zip(results.xyxy, results.confidence):
                    x1, y1, x2, y2 = map(float, box)
                    bbox = [x1, y1, x2, y2, float(score)]
                    if is_blue_object(frame, bbox):
                        continue
                    detections.append(bbox)

            # ---- ByteTrack（ARM CPU）----
            dets_np = np.array(detections, dtype=np.float64) if detections else np.empty((0, 5))
            tracks = tracker.update(dets_np, (height, width), (height, width))
            active_tracks = [(int(t.track_id), t.tlbr) for t in tracks if t.is_activated]

            # ---- 区域分析 ----
            for tid, bbox in active_tracks:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                w  = bbox[2] - bbox[0]
                h  = bbox[3] - bbox[1]
                analyzer.update(tid, cx, frame_idx, cy, w, h)

            # ---- 绘制标注 ----
            annotated = frame.copy()

            # 三条分区线
            cv2.line(annotated, (int(analyzer.split_0), 0), (int(analyzer.split_0), height), (255, 128, 0), 2)
            cv2.line(annotated, (int(analyzer.split_1), 0), (int(analyzer.split_1), height), (0, 255, 255), 2)
            cv2.line(annotated, (int(analyzer.split_2), 0), (int(analyzer.split_2), height), (0, 255, 255), 2)

            # 总计数（三线均值取整）
            avg = (analyzer.line_counters['line0'] +
                   analyzer.line_counters['line1'] +
                   analyzer.line_counters['line2']) / 3.0
            total_count = int(avg)
            cv2.rectangle(annotated, (width - 200, 0), (width, 60), (0, 0, 0), -1)
            cv2.putText(annotated, f"TOTAL: {total_count}", (width - 190, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # 帧号
            cv2.putText(annotated, f"Frame:{frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 线穿越计数
            lys = height - 120
            cv2.rectangle(annotated, (5, lys - 5), (220, height - 40), (0, 0, 0), -1)
            cv2.putText(annotated, "Line Crossings:", (10, lys + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, f"Line0: {analyzer.line_counters['line0']}",
                        (10, lys + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
            cv2.putText(annotated, f"Line1: {analyzer.line_counters['line1']}",
                        (10, lys + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(annotated, f"Line2: {analyzer.line_counters['line2']}",
                        (10, lys + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 跟踪框
            for tid, bbox in active_tracks:
                x1, y1, x2, y2 = map(int, bbox)
                np.random.seed(tid)
                color = tuple(map(int, np.random.randint(50, 255, 3)))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"ID:{tid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if writer is not None:
                writer.write(annotated)

            frame_idx += 1

            # 定时打印资源占用
            if monitor:
                _now = time.time()
                if _now - _last_stats_time >= _STATS_INTERVAL:
                    cur = monitor.current()
                    print(f"[Monitor] Frame {frame_idx:>6} | {ResourceMonitor.format_current(cur)}")
                    _last_stats_time = _now

            if max_frames > 0 and frame_idx >= max_frames:
                print(f"[Pipeline] 达到最大帧数 {max_frames}，停止。")
                break

    except KeyboardInterrupt:
        print("\n[Pipeline] 用户中断。")

    finally:
        cap.release()
        if writer is not None:
            writer.release()

        valid_count = analyzer.finalize(output_dir, "ByteTrack_Atlas")
        total_ids = len(analyzer.trajectories)
        print(f"\n[Pipeline] 最终有效计数: {valid_count}")
        print(f"[Pipeline] 总轨迹 ID 数: {total_ids}")

        if monitor:
            stats = monitor.stop()
            ResourceMonitor.print_summary(stats)

        if _own_detector:
            detector.release()
        print("[Pipeline] 结束。")

    return valid_count, total_ids


# ===========================================================================
# 入口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Atlas 200I DK A2 实时猪只计数")
    parser.add_argument(
        "--rtsp", required=True,
        help="大华摄像头 RTSP 地址，例如: rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
    )
    parser.add_argument("--om", default="/root/pig_tracking/inference_model.om",
                        help="OM 模型路径（默认: /root/pig_tracking/inference_model.om）")
    parser.add_argument("--output", default="/root/pig_tracking/output",
                        help="结果输出目录")
    parser.add_argument("--conf", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--out_ratio", type=float, default=0.45, help="OUT 区占比（默认 45%%）")
    parser.add_argument("--wait_ratio", type=float, default=0.25, help="WAIT 区占比（默认 25%%）")
    parser.add_argument("--no_video", action="store_true", help="不保存输出视频")
    parser.add_argument("--max_frames", type=int, default=0, help="最大处理帧数（0=不限制）")
    parser.add_argument("--no_monitor", action="store_true", help="关闭资源监控")
    args = parser.parse_args()

    run_pipeline(
        rtsp_url=args.rtsp,
        om_path=args.om,
        output_dir=args.output,
        conf_thresh=args.conf,
        out_ratio=args.out_ratio,
        wait_ratio=args.wait_ratio,
        save_video=not args.no_video,
        max_frames=args.max_frames,
        enable_monitor=not args.no_monitor,
    )


if __name__ == "__main__":
    main()
