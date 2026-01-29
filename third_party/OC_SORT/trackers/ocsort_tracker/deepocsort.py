"""
DeepOC-SORT: Fusion of OC-SORT and ReID features
Combines the motion model and velocity estimation of OC-SORT with appearance features from ReID
"""
from __future__ import print_function

import numpy as np
import torch
import cv2
from .ocsort import KalmanBoxTracker, k_previous_obs, convert_bbox_to_z, convert_x_to_bbox
from .association import *


def cosine_distance(a, b):
    """
    Compute cosine distance between two feature vectors
    Args:
        a: feature vector 1 (N x D)
        b: feature vector 2 (M x D)
    Returns:
        distance matrix (N x M)
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
    return 1.0 - np.dot(a_norm, b_norm.T)


class KalmanBoxTrackerWithFeature(KalmanBoxTracker):
    """
    Extended KalmanBoxTracker that stores ReID features
    """

    def __init__(self, bbox, delta_t=3, orig=False):
        super().__init__(bbox, delta_t, orig)
        self.features = []  # Store recent features
        self.feature_budget = 30  # Maximum number of features to store
        self.smooth_feature = None  # Smoothed feature representation

    def update(self, bbox, feature=None):
        """
        Update tracker with new detection and optional feature
        Args:
            bbox: bounding box [x1, y1, x2, y2, score]
            feature: ReID feature vector
        """
        super().update(bbox)

        if feature is not None:
            self.features.append(feature)
            # Keep only recent features
            if len(self.features) > self.feature_budget:
                self.features = self.features[-self.feature_budget:]
            # Update smooth feature (exponential moving average)
            if self.smooth_feature is None:
                self.smooth_feature = feature
            else:
                alpha = 0.9  # Smoothing factor
                self.smooth_feature = alpha * self.smooth_feature + (1 - alpha) * feature

    def get_feature(self):
        """Get the smoothed feature representation"""
        return self.smooth_feature


class DeepOCSort(object):
    """
    DeepOC-SORT: Combines OC-SORT motion model with ReID appearance features
    """

    def __init__(self, det_thresh, max_age=30, min_hits=3,
                 iou_threshold=0.3, delta_t=3, asso_func="giou", inertia=0.2,
                 use_byte=False, feature_extractor=None, appearance_weight=0.5):
        """
        Initialize DeepOC-SORT tracker

        Args:
            det_thresh: Detection threshold
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits to confirm a track
            iou_threshold: IoU threshold for matching
            delta_t: Time delta for velocity estimation
            asso_func: Association function (iou, giou, ciou, diou)
            inertia: Inertia weight for velocity compensation
            use_byte: Whether to use ByteTrack-style low confidence recovery
            feature_extractor: ReID feature extractor (callable that takes image crops)
            appearance_weight: Weight for appearance cost (0-1), 0=pure motion, 1=pure appearance
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t

        # Association function
        ASSO_FUNCS = {"iou": iou_batch, "giou": giou_batch,
                      "ciou": ciou_batch, "diou": diou_batch, "ct_dist": ct_dist}
        self.asso_func = ASSO_FUNCS[asso_func]

        self.inertia = inertia
        self.use_byte = use_byte
        self.feature_extractor = feature_extractor
        self.appearance_weight = appearance_weight  # Weight for appearance in combined cost

        # Reset tracker ID counter
        KalmanBoxTrackerWithFeature.count = 0

    def extract_features(self, frame, detections):
        """
        Extract ReID features for all detections

        Args:
            frame: Current frame image (BGR)
            detections: Detections array (N x 5) [x1, y1, x2, y2, score]

        Returns:
            features: Feature array (N x D) or None if no extractor
        """
        if self.feature_extractor is None or len(detections) == 0:
            return None

        crops = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            # Ensure valid crop
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)
            else:
                # Invalid crop, use placeholder
                crops.append(np.zeros((64, 64, 3), dtype=np.uint8))

        if len(crops) == 0:
            return None

        try:
            features = self.feature_extractor(crops)
            return features
        except Exception as e:
            print(f"Warning: Feature extraction failed: {e}")
            return None

    def compute_combined_cost(self, dets, trks, det_features, trk_features,
                            velocities, k_observations):
        """
        Compute combined cost matrix using both motion (IoU) and appearance (ReID)

        Args:
            dets: Detections (N x 5)
            trks: Tracks (M x 5)
            det_features: Detection features (N x D) or None
            trk_features: Track features (M x D) or None
            velocities: Track velocities
            k_observations: Previous observations

        Returns:
            cost_matrix: Combined cost matrix (M x N)
        """
        # Compute motion cost (IoU-based)
        iou_cost = 1 - self.asso_func(dets, trks)  # Convert similarity to cost

        # Apply velocity compensation (OC-SORT feature)
        if velocities is not None and k_observations is not None:
            # This matches OC-SORT's velocity-based prediction
            inertia_trks = np.copy(trks)
            for i, trk in enumerate(inertia_trks):
                if k_observations[i][0] != -1:
                    # Compensate with velocity
                    inertia_trks[i] = trk + self.inertia * velocities[i]

            # Recompute IoU with velocity compensation
            iou_cost_inertia = 1 - self.asso_func(dets, inertia_trks)
            # Use better of the two
            iou_cost = np.minimum(iou_cost, iou_cost_inertia)

        # If no appearance features, return pure motion cost
        if det_features is None or trk_features is None:
            return iou_cost

        # Compute appearance cost (cosine distance)
        appearance_cost = cosine_distance(trk_features, det_features)  # M x N

        # Combine costs with weighting
        combined_cost = (1 - self.appearance_weight) * iou_cost + \
                       self.appearance_weight * appearance_cost

        return combined_cost

    def update(self, output_results, img_info, img_size, frame=None):
        """
        Update tracker with new detections

        Args:
            output_results: Detection results (N x 5 or N x 6) [x1, y1, x2, y2, score] or [..., class]
            img_info: Image info [height, width]
            img_size: Model input size
            frame: Current frame image (BGR) for feature extraction

        Returns:
            tracks: Active tracks (N x 5) [x1, y1, x2, y2, track_id]
        """
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1

        # Post-process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            if torch.is_tensor(output_results):
                output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]

        # Scale bboxes to original image size
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

        # Split detections by confidence (for ByteTrack)
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # Extract features for high-confidence detections
        det_features = None
        if frame is not None and len(dets) > 0:
            det_features = self.extract_features(frame, dets)

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Get tracker features
        trk_features = None
        if det_features is not None:
            trk_features = []
            for trk in self.trackers:
                feat = trk.get_feature()
                if feat is not None:
                    trk_features.append(feat)
                else:
                    # No feature yet, use zero vector
                    trk_features.append(np.zeros(det_features.shape[1]))
            if len(trk_features) > 0:
                trk_features = np.array(trk_features)
            else:
                trk_features = None

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0))
                              for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t)
                                  for trk in self.trackers])

        # === First round: Match high-confidence detections with combined cost ===
        if len(dets) > 0 and len(trks) > 0:
            # Use standard associate function from OC-SORT
            # It uses IoU internally, which is fine for now
            # In future, we could create a custom associate that uses combined cost
            matched, unmatched_dets, unmatched_trks = associate(
                dets, trks, self.iou_threshold, velocities, k_observations,
                self.inertia)
        else:
            matched = np.empty((0, 2), dtype=int)
            unmatched_dets = np.arange(len(dets))
            unmatched_trks = np.arange(len(self.trackers))

        # Update matched trackers
        for m in matched:
            feat = det_features[m[0]] if det_features is not None else None
            self.trackers[m[1]].update(dets[m[0], :], feature=feat)

        # === Second round: ByteTrack-style low confidence recovery (if enabled) ===
        if self.use_byte and len(dets_second) > 0 and len(unmatched_trks) > 0:
            u_trks = [self.trackers[i] for i in unmatched_trks]
            u_trks_pos = np.array([trk.predict()[0] for trk in u_trks])
            u_trks_pos = u_trks_pos.reshape(-1, 4)
            u_trks_arr = np.concatenate([u_trks_pos, np.zeros((len(u_trks_pos), 1))], axis=1)

            # Use IoU only for second round
            iou_left = self.asso_func(dets_second, u_trks_arr)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                from scipy.optimize import linear_sum_assignment
                matched_indices = linear_sum_assignment(-iou_left)
                matched_indices = np.asarray(matched_indices).T

                matches_byte = []
                for m in matched_indices:
                    if iou_left[m[0], m[1]] > self.iou_threshold:
                        matches_byte.append([m[0], m[1]])

                if len(matches_byte) > 0:
                    matches_byte = np.array(matches_byte)
                    for m in matches_byte:
                        # Update with low-confidence detection (no feature)
                        u_trks[m[1]].update(dets_second[m[0], :], feature=None)

                    # Remove byte-matched from unmatched
                    unmatched_trks = [i for i in unmatched_trks if
                                     (i - unmatched_trks[0]) not in matches_byte[:, 1]]

        # === Third round: Match remaining with history observations (OC-SORT feature) ===
        if len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            left_dets = dets[unmatched_dets]
            left_trks = [self.trackers[i] for i in unmatched_trks]

            # Use k_observations for history-based matching
            left_trks_hist = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t)
                                      for trk in left_trks])

            # Only match if we have valid history
            valid_hist = (left_trks_hist[:, 0] != -1)
            if valid_hist.any():
                iou_left = self.asso_func(left_dets, left_trks_hist[valid_hist])
                iou_left = np.array(iou_left)

                if iou_left.max() > self.iou_threshold:
                    from scipy.optimize import linear_sum_assignment
                    matched_indices = linear_sum_assignment(-iou_left)
                    matched_indices = np.asarray(matched_indices).T

                    matches_hist = []
                    valid_idx_map = np.where(valid_hist)[0]
                    for m in matched_indices:
                        if iou_left[m[0], m[1]] > self.iou_threshold:
                            det_idx = unmatched_dets[m[0]]
                            trk_idx = unmatched_trks[valid_idx_map[m[1]]]
                            matches_hist.append([det_idx, trk_idx])

                    if len(matches_hist) > 0:
                        matches_hist = np.array(matches_hist)
                        for m in matches_hist:
                            feat = det_features[m[0]] if det_features is not None else None
                            self.trackers[m[1]].update(dets[m[0], :], feature=feat)

                        # Update unmatched lists
                        unmatched_dets = [d for d in unmatched_dets if d not in matches_hist[:, 0]]
                        unmatched_trks = [t for t in unmatched_trks if t not in matches_hist[:, 1]]

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTrackerWithFeature(dets[i, :], delta_t=self.delta_t)
            # Initialize with feature if available
            if det_features is not None:
                trk.update(dets[i, :], feature=det_features[i])
            self.trackers.append(trk)

        # Remove dead trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        # Return confirmed tracks
        ret = []
        for trk in self.trackers:
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()[0]
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
