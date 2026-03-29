#!/usr/bin/env python3
"""
Comprehensive Tracker Comparison Script
四种跟踪器对比分析：DeepSORT+ReID, DeepSORT纯IOU, ByteTrack, SimpleIOU

输出:
1. 每个跟踪器的结果视频 (包含实时 Total 计数)
2. 详细日志 CSV (ID变化, 状态变化, 时间戳)
3. 对比报告
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from types import SimpleNamespace
import csv
from datetime import datetime
from supervision import box_iou_batch

LINE_NAMES = ('line0', 'line1', 'line2')

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "OC_SORT"))

from rfdetr import RFDETRBase


class PigTrajectory:
    """跟踪单只猪的轨迹和状态变化"""

    def __init__(self, track_id, first_frame, first_zone):
        self.track_id = track_id
        self.zone_history = [first_zone]
        self.state_changes = []
        self.first_frame = first_frame
        self.last_frame = first_frame
        self.status = "ACTIVE"

        # 扩展调试信息
        self.positions = []  # [(cx, cy, frame)]
        self.box_sizes = []  # [(w, h, frame)]
        self.confidences = []  # [(score, frame)]
        self.lost_frames = []  # 丢失帧
        self.recovered_count = 0
        self.last_cx = None
        self.line_crossings = {line_name: [] for line_name in LINE_NAMES}

    def add_point(self, zone, frame_idx, fps, cx=None, cy=None, w=None, h=None, conf=None):
        # 检测丢失帧
        if frame_idx - self.last_frame > 1:
            for lost_f in range(self.last_frame + 1, frame_idx):
                self.lost_frames.append(lost_f)
            self.recovered_count += 1

        self.last_frame = frame_idx

        # 记录位置
        if cx is not None:
            self.positions.append((cx, cy, frame_idx))
            self.last_cx = cx
        if w is not None:
            self.box_sizes.append((w, h, frame_idx))
        if conf is not None:
            self.confidences.append((conf, frame_idx))

        # 区域变化
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

    def record_line_crossing(self, event):
        self.line_crossings[event['line_name']].append(event.copy())

    def get_line_crossing_stats(self, line_name):
        events = self.line_crossings[line_name]
        positive = sum(1 for event in events if event['counter_change'] > 0)
        reverse = sum(1 for event in events if event['counter_change'] < 0)
        return {
            'positive': positive,
            'reverse': reverse,
            'net': positive - reverse
        }

    def analyze(self):
        """分析轨迹是否有效"""
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

        # Last attempt analysis
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
    """区域分析器"""

    def __init__(self, width, fps, out_ratio=0.45, wait_ratio=0.25):
        self.width = width
        self.fps = fps
        self.out_ratio = out_ratio
        self.wait_ratio = wait_ratio
        self.split_0 = width * (out_ratio / 2)  # OUT1 | OUT2 (OUT区中间)
        self.split_1 = width * out_ratio  # OUT2 | WAIT
        self.split_2 = width * (out_ratio + wait_ratio)  # WAIT | ENTRY

        self.trajectories = {}
        self.valid_count = 0
        self.id_events = []  # 所有ID事件日志

        # 线穿越计数器（3条线）
        self.line_counters = {
            'line0': 0,  # OUT区内部 (split_0)
            'line1': 0,  # OUT-WAIT边界 (split_1)
            'line2': 0   # WAIT-ENTRY边界 (split_2)
        }
        self.line_positions = {
            'line0': self.split_0,
            'line1': self.split_1,
            'line2': self.split_2
        }
        self.prev_positions = {}  # 记录每个ID的上一帧x坐标

    def get_zone(self, cx):
        if cx < self.split_1:
            return 'OUT'
        elif cx < self.split_2:
            return 'WAIT'
        else:
            return 'ENTRY'

    def _record_line_crossing(self, track_id, line_name, frame_idx, prev_cx, cx, prev_zone, curr_zone,
                              positive_direction):
        counter_change = 1 if positive_direction else -1
        direction = 'POSITIVE' if positive_direction else 'REVERSE'
        counter_before = self.line_counters[line_name]
        self.line_counters[line_name] += counter_change
        counter_after = self.line_counters[line_name]

        event = {
            'frame': frame_idx,
            'timestamp': f"{frame_idx / self.fps:.2f}s",
            'track_id': track_id,
            'line_name': line_name,
            'line_position_x': round(self.line_positions[line_name], 2),
            'direction': direction,
            'counter_change': counter_change,
            'line_count_before': counter_before,
            'line_count_after': counter_after,
            'prev_cx': round(prev_cx, 2),
            'curr_cx': round(cx, 2),
            'prev_zone': prev_zone,
            'curr_zone': curr_zone,
            'details': (
                f"ID {track_id} crossed {line_name} in {direction} direction, "
                f"counter {counter_before}->{counter_after}, "
                f"cx {prev_cx:.2f}->{cx:.2f}, zone {prev_zone}->{curr_zone}"
            )
        }

        if track_id in self.trajectories:
            self.trajectories[track_id].record_line_crossing(event)

    def update(self, track_id, cx, frame_idx, cy=None, w=None, h=None, conf=None):
        zone = self.get_zone(cx)

        # 检测线穿越（方向：右到左为正方向+1，左到右为反方向-1）
        if track_id in self.prev_positions:
            prev_cx = self.prev_positions[track_id]
            prev_zone = self.get_zone(prev_cx)

            # 检测穿越Line0 (split_0: OUT区内部)
            if prev_cx > self.split_0 >= cx:  # 右到左穿越（正方向）
                self._record_line_crossing(track_id, 'line0', frame_idx, prev_cx, cx, prev_zone, zone, True)
            elif prev_cx < self.split_0 <= cx:  # 左到右穿越（反方向）
                self._record_line_crossing(track_id, 'line0', frame_idx, prev_cx, cx, prev_zone, zone, False)

            # 检测穿越Line1 (split_1: OUT-WAIT边界)
            if prev_cx > self.split_1 >= cx:  # 右到左穿越（正方向）
                self._record_line_crossing(track_id, 'line1', frame_idx, prev_cx, cx, prev_zone, zone, True)
            elif prev_cx < self.split_1 <= cx:  # 左到右穿越（反方向）
                self._record_line_crossing(track_id, 'line1', frame_idx, prev_cx, cx, prev_zone, zone, False)

            # 检测穿越Line2 (split_2: WAIT-ENTRY边界)
            if prev_cx > self.split_2 >= cx:  # 右到左穿越（正方向）
                self._record_line_crossing(track_id, 'line2', frame_idx, prev_cx, cx, prev_zone, zone, True)
            elif prev_cx < self.split_2 <= cx:  # 左到右穿越（反方向）
                self._record_line_crossing(track_id, 'line2', frame_idx, prev_cx, cx, prev_zone, zone, False)

        # 更新位置记录
        self.prev_positions[track_id] = cx

        if track_id not in self.trajectories:
            self.trajectories[track_id] = PigTrajectory(track_id, frame_idx, zone)
            self.id_events.append({
                'frame': frame_idx,
                'timestamp': f"{frame_idx / self.fps:.2f}s",
                'event': 'NEW_ID',
                'track_id': track_id,
                'zone': zone,
                'details': f"ID {track_id} appeared in {zone} (conf={conf:.2f})" if conf else f"ID {track_id} appeared in {zone}"
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

    def get_current_valid_count(self):
        """计算当前有效计数"""
        count = 0
        for tid, traj in self.trajectories.items():
            is_valid, _ = traj.analyze()
            if is_valid:
                count += 1
        return count

    def finalize(self, output_dir, tracker_name):
        """保存详细报告 - 精确到秒"""
        # 三条线均值四舍五入 → 最终计数
        line_avg = (self.line_counters['line0'] +
                    self.line_counters['line1'] +
                    self.line_counters['line2']) / 3.0
        final_count = round(line_avg)

        # 1. ID事件日志 (CSV)
        events_path = output_dir / f"{tracker_name}_id_events.csv"
        with open(events_path, 'w', encoding='utf-8',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'timestamp', 'event', 'track_id', 'zone', 'details'])
            writer.writeheader()
            writer.writerows(self.id_events)

        # 2. 按ID汇总三条线穿越情况 (CSV)
        line_crossing_path = output_dir / f"{tracker_name}_id_line_crossing_summary.csv"
        with open(line_crossing_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = [
                'TrackID', 'IsValid', 'Reason',
                'FirstFrame', 'FirstTime(s)', 'LastFrame', 'LastTime(s)',
                'ZoneHistory', 'LineName', 'LinePositionX',
                'PositiveCrossings', 'ReverseCrossings', 'NetCounterChange',
                'CrossingEventCount', 'CrossingEvents'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for tid in sorted(self.trajectories.keys()):
                traj = self.trajectories[tid]
                is_valid, reason = traj.analyze()
                zone_history = "->".join(traj.zone_history)

                for line_name in LINE_NAMES:
                    stats = traj.get_line_crossing_stats(line_name)
                    events = traj.line_crossings[line_name]
                    event_details = "; ".join([
                        f"F{event['frame']}@{event['timestamp']} {event['direction']} "
                        f"{event['counter_change']:+d} count {event['line_count_before']}->{event['line_count_after']} "
                        f"cx {event['prev_cx']:.2f}->{event['curr_cx']:.2f} "
                        f"zone {event['prev_zone']}->{event['curr_zone']}"
                        for event in events
                    ]) if events else "NO_CROSSING"

                    writer.writerow({
                        'TrackID': tid,
                        'IsValid': is_valid,
                        'Reason': reason,
                        'FirstFrame': traj.first_frame,
                        'FirstTime(s)': f"{traj.first_frame / self.fps:.2f}",
                        'LastFrame': traj.last_frame,
                        'LastTime(s)': f"{traj.last_frame / self.fps:.2f}",
                        'ZoneHistory': zone_history,
                        'LineName': line_name,
                        'LinePositionX': f"{self.line_positions[line_name]:.2f}",
                        'PositiveCrossings': stats['positive'],
                        'ReverseCrossings': stats['reverse'],
                        'NetCounterChange': stats['net'],
                        'CrossingEventCount': len(events),
                        'CrossingEvents': event_details
                    })

        # 3. 详细状态变化日志 (TXT) - 精确到秒
        txt_path = output_dir / f"{tracker_name}_state_changes.txt"
        traj_valid_count = 0
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 70}\n")
            f.write(f"Tracker: {tracker_name}\n")
            f.write(f"{'=' * 70}\n\n")

            for tid in sorted(self.trajectories.keys()):
                traj = self.trajectories[tid]
                is_valid, reason = traj.analyze()
                status = "✅ VALID" if is_valid else "❌ INVALID"

                f.write(f"ID {tid}: {status} ({reason})\n")
                f.write(f"  首次出现: Frame {traj.first_frame} ({traj.first_frame / self.fps:.2f}s)\n")
                f.write(f"  最后出现: Frame {traj.last_frame} ({traj.last_frame / self.fps:.2f}s)\n")
                f.write(f"  持续时间: {(traj.last_frame - traj.first_frame) / self.fps:.2f}s\n")
                f.write(f"  区域路径: {' -> '.join(traj.zone_history)}\n")

                if traj.state_changes:
                    f.write(f"  状态变化:\n")
                    for change in traj.state_changes:
                        f.write(
                            f"    [{change['timestamp']}] Frame {change['frame']}: {change['from_zone']} -> {change['to_zone']}\n")

                f.write(f"  移动距离: {traj.get_travel_distance():.1f} px\n")
                f.write(f"  平均速度: {traj.get_avg_speed(self.fps):.1f} px/s\n")
                f.write(f"  平均置信度: {traj.get_avg_confidence():.2f}\n")
                f.write(f"  丢失帧数: {len(traj.lost_frames)}\n")
                f.write(f"  恢复次数: {traj.recovered_count}\n")
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

        # 4. 轨迹分析报告 (CSV)
        report_path = output_dir / f"{tracker_name}_trajectory_report.csv"
        with open(report_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['TrackID', 'IsValid', 'Reason', 'FirstFrame', 'FirstTime(s)', 'LastFrame', 'LastTime(s)',
                             'Duration(s)', 'ZoneHistory', 'StateChanges'])

            for tid in sorted(self.trajectories.keys()):
                traj = self.trajectories[tid]
                is_valid, reason = traj.analyze()
                duration = (traj.last_frame - traj.first_frame) / self.fps
                zone_str = "->".join(traj.zone_history)
                changes_str = "; ".join(
                    [f"{c['timestamp']}: {c['from_zone']}->{c['to_zone']}" for c in traj.state_changes])

                writer.writerow([
                    tid, is_valid, reason,
                    traj.first_frame, f"{traj.first_frame / self.fps:.2f}",
                    traj.last_frame, f"{traj.last_frame / self.fps:.2f}",
                    f"{duration:.2f}", zone_str, changes_str
                ])

            # 汇总行
            writer.writerow([])
            writer.writerow(['FINAL_COUNT', final_count,
                             f'Line0={self.line_counters["line0"]} Line1={self.line_counters["line1"]} Line2={self.line_counters["line2"]} avg={line_avg:.2f}',
                             '', '', '', '', '', '', ''])

        self.valid_count = final_count
        print(f"  {tracker_name}: Events CSV    -> {events_path}")
        print(f"  {tracker_name}: Line Cross CSV -> {line_crossing_path}")
        print(f"  {tracker_name}: State TXT     -> {txt_path}")
        print(f"  {tracker_name}: Trajectory CSV -> {report_path}")
        return final_count


def is_blue_object(frame, bbox, blue_threshold=1.3):
    """
    检测物体是否为蓝色
    Args:
        frame: 原始图像帧
        bbox: 检测框 [x1, y1, x2, y2, score]
        blue_threshold: 蓝色判定阈值，蓝色通道要大于红色和绿色的平均值的倍数
    Returns:
        True if 物体主要为蓝色
    """
    x1, y1, x2, y2 = map(int, bbox[:4])

    # 确保边界在图像范围内
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return False

    # 提取检测框区域
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return False

    # 计算BGR通道的平均值
    b_mean = np.mean(roi[:, :, 0])  # 蓝色通道
    g_mean = np.mean(roi[:, :, 1])  # 绿色通道
    r_mean = np.mean(roi[:, :, 2])  # 红色通道

    # 判断是否为蓝色：蓝色通道明显大于红色和绿色
    rg_mean = (r_mean + g_mean) / 2.0

    if b_mean > rg_mean * blue_threshold and b_mean > 80:  # 蓝色要足够明显且有一定亮度
        return True

    return False


# ===== Simple IOU Tracker =====
class SimpleIOUTrack:
    _id_counter = 0

    def __init__(self, bbox):
        SimpleIOUTrack._id_counter += 1
        self.track_id = SimpleIOUTrack._id_counter
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        self.age += 1

    def predict(self):
        self.age += 1
        self.time_since_update += 1


class SimpleIOUTracker:
    """最简单的IOU跟踪器"""

    def __init__(self, iou_threshold=0.3, max_age=5, min_hits=3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        SimpleIOUTrack._id_counter = 0

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def update(self, detections):
        """detections: list of [x1, y1, x2, y2, score]"""
        # Predict existing tracks
        for track in self.tracks:
            track.predict()

        if len(detections) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.hits >= self.min_hits and t.time_since_update == 0]

        # Match detections to tracks
        det_boxes = [d[:4] for d in detections]

        matched_tracks = set()
        matched_dets = set()

        # Greedy matching
        for i, det_box in enumerate(det_boxes):
            best_iou = self.iou_threshold
            best_track_idx = -1

            for j, track in enumerate(self.tracks):
                if j in matched_tracks:
                    continue
                iou = self._iou(det_box, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = j

            if best_track_idx >= 0:
                self.tracks[best_track_idx].update(det_box)
                matched_tracks.add(best_track_idx)
                matched_dets.add(i)

        # Create new tracks for unmatched detections
        for i, det_box in enumerate(det_boxes):
            if i not in matched_dets:
                self.tracks.append(SimpleIOUTrack(det_box))

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed tracks
        return [t for t in self.tracks if t.hits >= self.min_hits and t.time_since_update == 0]


def run_tracker(tracker_type, detector, video_path, output_dir, fps, width, height, limit=0, out_ratio=0.45,
                wait_ratio=0.25):
    """运行单个跟踪器"""
    print(f"\n{'=' * 50}")
    print(f"Running: {tracker_type}")
    print(f"{'=' * 50}")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if limit > 0:
        total_frames = min(total_frames, limit)

    output_video = output_dir / f"{tracker_type}_result.mp4"
    out = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    analyzer = ZoneAnalyzer(width, fps, out_ratio, wait_ratio)

    # Initialize tracker based on type
    use_temporal = "_Temporal" in tracker_type
    base_type = tracker_type.replace("_Temporal", "")

    # Initialize temporal enhancer if needed
    temporal_enhancer = None
    if use_temporal:
        from tools.tracking.temporal_enhancer import TemporalEnhancer
        temporal_enhancer = TemporalEnhancer(motion_weight=0.3, velocity_threshold=50)

    if base_type == "DeepSORT_ReID":
        from deep_sort_realtime.deepsort_tracker import DeepSort
        tracker = DeepSort(max_age=30, n_init=3, embedder="mobilenet", half=True, bgr=True, embedder_gpu=True)
        use_reid = True
    elif base_type == "DeepSORT_NoReID":
        from deep_sort_realtime.deepsort_tracker import DeepSort
        tracker = DeepSort(max_age=15, n_init=5, embedder=None)
        use_reid = False
    elif base_type == "ByteTrack":
        from trackers.byte_tracker.byte_tracker import BYTETracker
        byte_args = SimpleNamespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False)
        tracker = BYTETracker(byte_args, frame_rate=int(fps))
        use_reid = None
    elif base_type == "SimpleIOU":
        tracker = SimpleIOUTracker(iou_threshold=0.3, max_age=10, min_hits=3)
        use_reid = None
    elif tracker_type == "Temporal":
        from tools.tracking.rfdetr_temporal import TemporalTracker
        tracker = TemporalTracker(iou_threshold=0.3, motion_weight=0.4, max_age=30, min_hits=3)
        use_reid = None

    pbar = tqdm(total=total_frames, desc=tracker_type)
    frame_idx = 0

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect
        results = detector.predict(frame)
        detections = []

        if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
            for box, score in zip(results.xyxy, results.confidence):
                if score < 0.5:
                    continue
                x1, y1, x2, y2 = map(float, box)
                bbox = [x1, y1, x2, y2, float(score)]

                # 过滤蓝色物体
                if is_blue_object(frame, bbox):
                    continue  # 跳过蓝色物体

                detections.append(bbox)

        # 2. Track
        active_tracks = []  # Initialize to avoid undefined variable error
        if base_type == "DeepSORT_ReID" or base_type == "DeepSORT_NoReID":
            ds_inputs = [[[int(d[0]), int(d[1]), int(d[2] - d[0]), int(d[3] - d[1])], d[4], 0] for d in detections]
            if base_type == "DeepSORT_ReID":
                tracks = tracker.update_tracks(ds_inputs, frame=frame)
            else:
                random_embeds = [np.random.randn(128) / 10 + 0.5 for _ in ds_inputs]
                random_embeds = [e / (np.linalg.norm(e) + 1e-8) for e in random_embeds]
                tracks = tracker.update_tracks(ds_inputs, frame=frame, embeds=random_embeds)
            active_tracks = [(int(t.track_id), t.to_ltrb()) for t in tracks if t.is_confirmed()]

        elif base_type == "ByteTrack":
            dets = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
            tracks = tracker.update(dets, (height, width), (height, width))
            active_tracks = [(int(t.track_id), t.tlbr) for t in tracks if t.is_activated]

        elif base_type == "SimpleIOU":
            tracks = tracker.update(detections)
            active_tracks = [(t.track_id, t.bbox) for t in tracks]

        elif tracker_type == "Temporal":
            tracks = tracker.update(detections)
            active_tracks = [(t.track_id, t.bbox[:4]) for t in tracks]

        # Apply temporal enhancement if enabled
        if temporal_enhancer is not None:
            active_tracks = temporal_enhancer.update(active_tracks, detections)

        # 3. Analyze
        for tid, bbox in active_tracks:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            analyzer.update(tid, cx, frame_idx, cy, w, h)

        # 4. Draw
        annotated = frame.copy()

        # Draw zones (3 lines)
        cv2.line(annotated, (int(analyzer.split_0), 0), (int(analyzer.split_0), height), (255, 128, 0), 2)  # Line 0
        cv2.line(annotated, (int(analyzer.split_1), 0), (int(analyzer.split_1), height), (0, 255, 255), 2)  # Line 1
        cv2.line(annotated, (int(analyzer.split_2), 0), (int(analyzer.split_2), height), (0, 255, 255), 2)  # Line 2

        # Draw total count (average of 3 lines, rounded)
        avg_count = (analyzer.line_counters['line0'] + analyzer.line_counters['line1'] +
                     analyzer.line_counters['line2']) / 3.0
        total_count = round(avg_count)
        cv2.rectangle(annotated, (width - 200, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(annotated, f"TOTAL: {total_count}", (width - 190, 40), 0, 1.0, (0, 255, 0), 2)

        # Draw tracker name
        cv2.putText(annotated, tracker_type, (10, height - 20), 0, 0.6, (255, 255, 255), 2)

        # Draw line counters in bottom-left corner
        line_y_start = height - 120
        cv2.rectangle(annotated, (5, line_y_start - 5), (220, height - 40), (0, 0, 0), -1)
        cv2.putText(annotated, "Line Crossings:", (10, line_y_start + 15), 0, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Line 0: {analyzer.line_counters['line0']}",
                    (10, line_y_start + 35), 0, 0.5, (255, 128, 0), 1)
        cv2.putText(annotated, f"Line 1: {analyzer.line_counters['line1']}",
                    (10, line_y_start + 55), 0, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated, f"Line 2: {analyzer.line_counters['line2']}",
                    (10, line_y_start + 75), 0, 0.5, (0, 255, 255), 1)

        # Draw tracks
        for tid, bbox in active_tracks:
            x1, y1, x2, y2 = map(int, bbox)
            np.random.seed(tid)
            color = tuple(map(int, np.random.randint(50, 255, 3)))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"ID:{tid}", (x1, y1 - 5), 0, 0.4, color, 1)

        out.write(annotated)
        pbar.update(1)
        frame_idx += 1

    cap.release()
    out.release()
    pbar.close()

    # Finalize
    valid_count = analyzer.finalize(output_dir, tracker_type)
    print(f"  Video: {output_video}")
    print(f"  Valid Count: {valid_count}")

    return valid_count, len(analyzer.trajectories)


def main():
    parser = argparse.ArgumentParser(description='Compare Trackers with Zone Configurations')
    parser.add_argument('--video_path', type=str,default=r"I:\rf-detr\片段\2-30头.mp4", required=False)
    parser.add_argument('--model_path', type=str, default=r"G:\Jin的U盘资料\YOLO\rf-detr-base.pth",required=False)
    parser.add_argument('--output_dir', type=str, default=r"G:\Jin的U盘资料\YOLO\output")
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out_ratio', type=float, default=0.45, help='OUT区占比 (默认45%)')
    parser.add_argument('--wait_ratio', type=float, default=0.25, help='WAIT区占比 (默认25%)')
    parser.add_argument('--zone_name', type=str, default='', help='区域配置名称 (用于目录)')
    parser.add_argument('--tracker', type=str, default='',
                        help='单独运行某个追踪器 (ByteTrack)')
    parser.add_argument('--no_timestamp', action='store_true', help='不使用时间戳子目录')

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.no_timestamp:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    entry_ratio = 1.0 - args.out_ratio - args.wait_ratio
    print(f"Output Directory: {output_dir}")
    print(
        f"Zone Config: OUT={args.out_ratio * 100:.0f}% | WAIT={args.wait_ratio * 100:.0f}% | ENTRY={entry_ratio * 100:.0f}%")

    # Load detector once
    print("\nLoading detector...")
    detector = RFDETRBase(pretrain_weights=args.model_path, conf_thresh=0.5)

    # Get video info
    cap = cv2.VideoCapture(args.video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Select trackers (8 total: 4 original + 4 temporal-enhanced)
    if args.tracker:
        trackers = [args.tracker]
    else:
        trackers= ["DeepSORT_ReID", "DeepSORT_NoReID", "ByteTrack", "SimpleIOU"]
    results = {}

    for tracker_type in trackers:
        try:
            valid, total_ids = run_tracker(
                tracker_type, detector, args.video_path, output_dir, fps, width, height,
                args.limit, args.out_ratio, args.wait_ratio
            )
            results[tracker_type] = {'valid': valid, 'total_ids': total_ids}
        except Exception as e:
            print(f"  Error: {e}")
            results[tracker_type] = {'valid': -1, 'total_ids': -1, 'error': str(e)}

    # Summary report
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Tracker':<20} {'Valid Count':<15} {'Total IDs':<15}")
    print("-" * 60)
    for tracker, res in results.items():
        if res['valid'] >= 0:
            print(f"{tracker:<20} {res['valid']:<15} {res['total_ids']:<15}")
        else:
            print(f"{tracker:<20} ERROR: {res.get('error', 'Unknown')}")
    print("=" * 60)

    # Save summary
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Tracker Comparison Summary\n")
        f.write(f"Video: {args.video_path}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        for tracker, res in results.items():
            f.write(f"{tracker}: Valid={res['valid']}, TotalIDs={res['total_ids']}\n")

    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
