
"""stTAM automatic discovery, motion classification and mask-guided tracking.

The demo uses the standard frozen EdgeTAM backbone. It does not enable local
experimental CAM/ORM extensions or contain benchmark evaluation machinery.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
import statistics
import tempfile
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Keep --help available before installing model dependencies.
if __name__ == "__main__" and any(x in __import__("sys").argv for x in ("-h", "--help")):
    print(
        "python sttam.py --config configs/endovis2017.yaml --input VIDEO_OR_FRAME_DIR --checkpoint EDGETAM_PT [--output DIR] [--device auto|cpu|cuda] [--max-frames N] [--box-source mask|yolo11] [--detection-cache JSON] [--show]"
    )
    raise SystemExit(0)

import cv2
import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from skimage.morphology import dilation, disk
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import load_video_frames

Box = Tuple[int, int, int, int]
DISCOVERY_FILTER_SETTINGS_BY_VIDEO = {}
MOTION_CLASSIFICATION_SETTINGS_BY_VIDEO = {}
BENCHMARK_BOX_SETTINGS_BY_VIDEO = {}
MASK_PROPAGATION_SETTINGS_BY_VIDEO = {}
MAX_LOST_FRAMES_BY_VIDEO = {}
INITIAL_POINT_PROMPTS_BY_VIDEO = {}
USE_MASK_GATE = True
USE_RESAMPLING = True
HISTORY_LENGTH = 20
MIN_VALID_POINTS = 5
TRACKING_POINTS = 10
RANDOM_SEED = 0


class StreamingPredictor(SAM2VideoPredictor):
    """Frame-at-a-time adapter; no local sam2 modifications are required."""

    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        is_realtime=False,
        max_cache_frames=5,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state for both pre-recorded and realtime video."""
        compute_device = self.device
        if is_realtime:
            images = None
            if os.path.isdir(video_path):
                first_frame_path = os.path.join(video_path, "000000.jpg")
                if os.path.exists(first_frame_path):
                    first_frame = cv2.imread(first_frame_path)
                    video_height, video_width = first_frame.shape[:2]
                else:
                    video_height, video_width = (480, 640)
            else:
                video_height, video_width = (480, 640)
            num_frames = 1000000
        else:
            images, video_height, video_width = load_video_frames(
                video_path=video_path,
                image_size=self.image_size,
                offload_video_to_cpu=offload_video_to_cpu,
                async_loading_frames=async_loading_frames,
                compute_device=compute_device,
            )
            num_frames = len(images)
        inference_state = {
            "images": images,
            "num_frames": num_frames,
            "is_realtime": is_realtime,
            "max_cache_frames": max_cache_frames,
            "current_frame_idx": 0,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "video_height": video_height,
            "video_width": video_width,
            "device": compute_device,
            "storage_device": torch.device("cpu") if offload_state_to_cpu else compute_device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {
                "cond_frame_outputs": set(),
                "non_cond_frame_outputs": set(),
            },
            "tracking_has_started": False,
            "frames_already_tracked": {},
            "ma_sam2": {
                "cam_scores": None,
                "object_sizes": {},
                "orm_frames": [],
                "last_orm_frame": -1,
            },
        }
        if is_realtime:
            inference_state["video_dir"] = video_path
        if not is_realtime or (is_realtime and images is not None):
            self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        is_realtime = inference_state.get("is_realtime", False)
        image, backbone_out = inference_state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is None:
            device = inference_state["device"]
            if is_realtime:
                video_dir = inference_state.get("video_dir", "./sam2_frames")
                frame_path = os.path.join(video_dir, f"{frame_idx:06d}.jpg")
                if not os.path.exists(frame_path):
                    raise FileNotFoundError(f"Frame {frame_idx} not found in {video_dir}")
                image_np = cv2.imread(frame_path)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                h, w = image_np.shape[:2]
                if h != self.image_size or w != self.image_size:
                    image_np = cv2.resize(
                        image_np, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
                    )
                image = torch.from_numpy(image_np).permute(2, 0, 1).to(device).float() / 255.0
                image = image.unsqueeze(0)
            else:
                image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos
        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features


def sample_prompt_points(
    mask: np.ndarray, positive_count: int, negative_count: int, boundary_radius: int = 6
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    
    mask_bool = mask.astype(bool)
    y_coordinates, x_coordinates = np.where(mask_bool)
    if len(x_coordinates) < positive_count:
        return (None, None)
    positive_indices = np.random.choice(len(x_coordinates), positive_count, replace=False)
    positive_points = np.column_stack(
        (x_coordinates[positive_indices], y_coordinates[positive_indices])
    ).astype(np.float32)
    negative_points: Optional[np.ndarray] = None
    if negative_count > 0:
        dilated_mask = dilation(mask_bool, disk(boundary_radius))
        boundary_mask = np.logical_xor(dilated_mask, mask_bool)
        negative_coordinates = np.argwhere(boundary_mask)
        if len(negative_coordinates) > 0:
            actual_count = min(negative_count, len(negative_coordinates))
            selected_indices = np.linspace(
                0, len(negative_coordinates) - 1, actual_count, dtype=int
            )
            negative_points = negative_coordinates[selected_indices][:, [1, 0]].astype(np.float32)
    return (positive_points, negative_points)


def build_prompt_arrays(
    positive_points: np.ndarray, negative_points: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    points = positive_points.reshape(-1, 2).astype(np.float32)
    labels = np.ones(len(points), dtype=np.int32)
    if negative_points is not None and len(negative_points) > 0:
        negative_points = negative_points.reshape(-1, 2).astype(np.float32)
        points = np.vstack([points, negative_points])
        labels = np.concatenate([labels, np.zeros(len(negative_points), dtype=np.int32)])
    return (points, labels)


def update_anchor_points(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_points: Optional[np.ndarray],
    previous_histories: List[List[List[float]]],
    current_tracking_mask: Optional[np.ndarray],
    lk_parameters: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], List[List[List[float]]], bool]:
    
    valid_new_points: List[np.ndarray] = []
    kept_old_indices: List[int] = []
    if previous_points is not None and len(previous_points) > 0:
        tracked_points, states, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, previous_points, None, **lk_parameters
        )
        if tracked_points is not None and states is not None:
            image_height, image_width = current_gray.shape[:2]
            for point_index, (point, state) in enumerate(zip(tracked_points, states)):
                if int(state[0]) != 1:
                    continue
                x_coordinate = int(round(float(point[0][0])))
                y_coordinate = int(round(float(point[0][1])))
                if not (0 <= x_coordinate < image_width and 0 <= y_coordinate < image_height):
                    continue
                if USE_MASK_GATE:
                    if (
                        current_tracking_mask is None
                        or not current_tracking_mask[y_coordinate, x_coordinate]
                    ):
                        continue
                valid_new_points.append(point[0].astype(np.float32))
                kept_old_indices.append(point_index)
    updated_histories: List[List[List[float]]] = []
    for new_index, old_index in enumerate(kept_old_indices):
        if old_index < len(previous_histories):
            history = list(previous_histories[old_index])
        else:
            history = []
        point_x = float(valid_new_points[new_index][0])
        point_y = float(valid_new_points[new_index][1])
        history.append([point_x, point_y])
        if len(history) > HISTORY_LENGTH:
            history = history[-HISTORY_LENGTH:]
        updated_histories.append(history)
    if valid_new_points:
        updated_points: Optional[np.ndarray] = np.asarray(
            valid_new_points, dtype=np.float32
        ).reshape(-1, 1, 2)
    else:
        updated_points = None
    resampled = False
    if (
        USE_RESAMPLING
        and current_tracking_mask is not None
        and (updated_points is None or len(updated_points) < MIN_VALID_POINTS)
    ):
        new_positive_points, _ = sample_prompt_points(
            current_tracking_mask, positive_count=TRACKING_POINTS, negative_count=0
        )
        if new_positive_points is not None:
            updated_points = new_positive_points.reshape(-1, 1, 2).astype(np.float32)
            updated_histories = [
                [[float(point[0]), float(point[1])]] for point in new_positive_points
            ]
            resampled = True
    return (updated_points, updated_histories, resampled)


def extract_motion_features(
    all_histories: List[List[List[List[float]]]],
) -> Tuple[np.ndarray, List[int], Dict[int, Dict[str, float]]]:
    
    feature_list: List[List[float]] = []
    valid_object_indices: List[int] = []
    feature_details: Dict[int, Dict[str, float]] = {}
    for object_index, object_histories in enumerate(all_histories):
        object_point_variances: List[float] = []
        object_point_speeds: List[float] = []
        for trajectory in object_histories:
            if len(trajectory) < 5:
                continue
            trajectory_array = np.asarray(trajectory, dtype=np.float32)
            slopes: List[float] = []
            for time_index in range(1, len(trajectory_array)):
                dx = float(trajectory_array[time_index, 0] - trajectory_array[time_index - 1, 0])
                dy = float(trajectory_array[time_index, 1] - trajectory_array[time_index - 1, 1])
                slopes.append(dy / dx if abs(dx) > 1.0 else 0.0)
            slope_variance = float(statistics.variance(slopes)) if len(slopes) > 5 else 0.0
            average_speed = float(
                np.mean(np.linalg.norm(np.diff(trajectory_array, axis=0), axis=1))
            )
            object_point_variances.append(slope_variance)
            object_point_speeds.append(average_speed)
        if object_point_variances:
            median_variance = float(np.median(object_point_variances))
            median_speed = float(np.median(object_point_speeds))
            feature_list.append([median_variance, median_speed])
            valid_object_indices.append(object_index)
            feature_details[object_index] = {
                "median_slope_variance": median_variance,
                "median_speed": median_speed,
                "valid_trajectories": len(object_point_variances),
            }
    return (np.asarray(feature_list, dtype=np.float64), valid_object_indices, feature_details)


def classify_instrument_candidates(
    all_histories: List[List[List[List[float]]]], number_of_objects: int
) -> Tuple[List[int], Dict[str, Any]]:
    
    final_labels = [1] * number_of_objects
    feature_array, valid_object_indices, feature_details = extract_motion_features(all_histories)
    classification_log: Dict[str, Any] = {
        "valid_object_indices": valid_object_indices,
        "features": feature_array.tolist(),
        "feature_details": feature_details,
        "fallback": None,
    }
    if number_of_objects == 0:
        classification_log["fallback"] = "no_candidate"
        return (final_labels, classification_log)
    if len(valid_object_indices) == 0:
        final_labels = [0] * number_of_objects
        classification_log["fallback"] = "no_valid_feature_keep_all"
        return (final_labels, classification_log)
    if len(valid_object_indices) == 1:
        final_labels[valid_object_indices[0]] = 0
        classification_log["fallback"] = "single_valid_candidate"
        return (final_labels, classification_log)
    kmeans = KMeans(n_clusters=2, n_init=20, random_state=RANDOM_SEED).fit(feature_array)
    cluster_scores: List[float] = []
    for cluster_id in range(2):
        cluster_data = feature_array[kmeans.labels_ == cluster_id]
        cluster_score = (
            float(np.mean(cluster_data[:, 0]) + np.mean(cluster_data[:, 1]))
            if len(cluster_data)
            else -float("inf")
        )
        cluster_scores.append(cluster_score)
    instrument_cluster_id = int(np.argmax(cluster_scores))
    for feature_index, cluster_label in enumerate(kmeans.labels_):
        real_object_index = valid_object_indices[feature_index]
        final_labels[real_object_index] = 0 if int(cluster_label) == instrument_cluster_id else 1
    classification_log.update(
        {
            "cluster_labels": kmeans.labels_.tolist(),
            "cluster_scores": cluster_scores,
            "instrument_cluster_id": instrument_cluster_id,
        }
    )
    return (final_labels, classification_log)


@dataclass
class Track:
    track_id: int
    birth_frame: int
    last_mask: np.ndarray
    status: str = "pending"
    last_box: Optional[Box] = None
    output_box: Optional[Box] = None
    box_mode: str = "full"
    last_score: float = 0.0
    lost_frames: int = 0
    age: int = 0
    active_points: Optional[np.ndarray] = None
    trajectories: List[deque] = field(default_factory=list)
    motion_samples: deque = field(default_factory=lambda: deque(maxlen=150))
    initial_elongation: float = 1.0
    initial_brightness: float = 0.0
    initial_area_ratio: float = 0.0
    initial_box: Optional[Box] = None
    tip_extent_ratio: Optional[float] = None


def mask_to_box(mask: np.ndarray, min_pixels: int = 20) -> Optional[Box]:
    ys, xs = np.where(mask)
    if len(xs) < min_pixels:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = (max(a[0], b[0]), max(a[1], b[1]))
    ix2, iy2 = (min(a[2], b[2]), min(a[3], b[3]))
    intersection = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    area_a = max(0, a[2] - a[0] + 1) * max(0, a[3] - a[1] + 1)
    area_b = max(0, b[2] - b[0] + 1) * max(0, b[3] - b[1] + 1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def box_containment(a: Box, b: Box) -> float:
    """Intersection area divided by the smaller box area."""
    ix1, iy1 = (max(a[0], b[0]), max(a[1], b[1]))
    ix2, iy2 = (min(a[2], b[2]), min(a[3], b[3]))
    intersection = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    area_a = max(1, (a[2] - a[0] + 1) * (a[3] - a[1] + 1))
    area_b = max(1, (b[2] - b[0] + 1) * (b[3] - b[1] + 1))
    return float(intersection / min(area_a, area_b))


def box_gap(a: Box, b: Box) -> float:
    """Euclidean separation between boxes; zero when they touch or overlap."""
    horizontal = max(a[0] - b[2] - 1, b[0] - a[2] - 1, 0)
    vertical = max(a[1] - b[3] - 1, b[1] - a[3] - 1, 0)
    return float(math.hypot(horizontal, vertical))


def union_box(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def preserve_full_box_short_side(
    box: Box, initial_box: Optional[Box], minimum_ratio: float, image_shape: Tuple[int, int]
) -> Box:
    """Prevent a full-mask box collapsing to one thin propagated fragment."""
    if initial_box is None or minimum_ratio <= 0.0:
        return box
    image_height, image_width = image_shape
    initial_width = initial_box[2] - initial_box[0] + 1
    initial_height = initial_box[3] - initial_box[1] + 1
    width = box[2] - box[0] + 1
    height = box[3] - box[1] + 1
    minimum_short_side = minimum_ratio * min(initial_width, initial_height)
    x1, y1, x2, y2 = box
    if width <= height and width < minimum_short_side:
        center = (x1 + x2) / 2.0
        half = (minimum_short_side - 1.0) / 2.0
        x1, x2 = (int(np.floor(center - half)), int(np.ceil(center + half)))
    elif height < width and height < minimum_short_side:
        center = (y1 + y2) / 2.0
        half = (minimum_short_side - 1.0) / 2.0
        y1, y2 = (int(np.floor(center - half)), int(np.ceil(center + half)))
    return (max(0, x1), max(0, y1), min(image_width - 1, x2), min(image_height - 1, y2))


def box_elongation(box: Box) -> float:
    width = box[2] - box[0] + 1
    height = box[3] - box[1] + 1
    return max(width, height) / max(1, min(width, height))


def mask_overlap(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    intersection = int(np.logical_and(a, b).sum())
    if intersection == 0:
        return (0.0, 0.0)
    union = int(np.logical_or(a, b).sum())
    smaller = min(int(a.sum()), int(b.sum()))
    return (intersection / max(1, union), intersection / max(1, smaller))


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def confidence_from_logits(logits: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits[mask], -30.0, 30.0)))
    return float(probability.mean())


def initial_box_mode(mask: np.ndarray, base_box: Box, enabled: bool) -> str:
    """Choose the benchmark-box convention once from the initial predicted mask."""
    if not enabled or int(mask.sum()) < 100:
        return "full"
    image_height, image_width = mask.shape
    box_width = base_box[2] - base_box[0] + 1
    box_height = base_box[3] - base_box[1] + 1
    elongation = max(box_width, box_height) / max(1, min(box_width, box_height))
    base_area = box_width * box_height
    return (
        "tip"
        if 1.3 <= elongation <= 3.0 and base_area >= 0.04 * image_height * image_width
        else "full"
    )


def benchmark_box_from_mask(
    mask: np.ndarray,
    base_box: Box,
    box_mode: str,
    previous_box: Optional[Box] = None,
    endpoint_quantile: float = 0.1,
    prefer_distal_each_frame: bool = False,
    frame_bgr: Optional[np.ndarray] = None,
    appearance_settings: Optional[Dict[str, float]] = None,
    previous_extent_ratio: Optional[float] = None,
) -> Tuple[Box, Optional[float]]:
    """Derive a benchmark box while retaining the full mask for propagation."""
    if box_mode != "tip" or int(mask.sum()) < 100:
        return (base_box, None)
    image_height, image_width = mask.shape
    box_width = base_box[2] - base_box[0] + 1
    box_height = base_box[3] - base_box[1] + 1
    base_area = box_width * box_height
    ys, xs = np.where(mask)
    coordinates = np.column_stack((xs, ys)).astype(np.float32)
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    principal_axis = axes[0]
    projection = centered @ principal_axis
    low_point = coordinates[int(np.argmin(projection))]
    high_point = coordinates[int(np.argmax(projection))]

    def border_distance(point: np.ndarray) -> float:
        x, y = (float(point[0]), float(point[1]))
        return min(x, y, image_width - 1 - x, image_height - 1 - y)

    def endpoint_box(selector: np.ndarray) -> Optional[Box]:
        if int(selector.sum()) < 20:
            return None
        endpoint_coordinates = coordinates[selector]
        x1, y1 = np.floor(endpoint_coordinates.min(axis=0)).astype(int)
        x2, y2 = np.ceil(endpoint_coordinates.max(axis=0)).astype(int)
        pad = 5
        refined = (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(image_width - 1, x2 + pad),
            min(image_height - 1, y2 + pad),
        )
        refined_area = (refined[2] - refined[0] + 1) * (refined[3] - refined[1] + 1)
        return refined if refined_area >= 100 and refined_area <= 1.25 * base_area else None

    def appearance_extent(oriented_projection: np.ndarray) -> Optional[float]:
        """Find the working-end/shaft transition from color inside the mask."""
        settings = appearance_settings or {}
        if not settings.get("appearance_adaptive") or frame_bgr is None:
            return None
        span = float(np.ptp(oriented_projection))
        if span < 8.0:
            return None
        normalized = (oriented_projection - float(oriented_projection.min())) / span
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        values = hsv[..., 2][mask]
        saturations = hsv[..., 1][mask]
        bright = (values >= float(settings.get("appearance_min_value", 140))) & (
            saturations <= float(settings.get("appearance_max_saturation", 115))
        )
        bin_count = 30
        bin_ids = np.minimum((normalized * bin_count).astype(np.int32), bin_count - 1)
        totals = np.bincount(bin_ids, minlength=bin_count)
        bright_totals = np.bincount(bin_ids[bright], minlength=bin_count)
        fractions = np.divide(
            bright_totals, totals, out=np.zeros(bin_count, dtype=np.float64), where=totals > 0
        )
        active = (
            (totals >= 8)
            & (bright_totals >= 4)
            & (fractions >= float(settings.get("appearance_min_bin_fraction", 0.3)))
        ).astype(np.uint8)
        active = cv2.morphologyEx(
            active.reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8)
        )[0]
        runs: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for index, enabled in enumerate(np.r_[active, 0]):
            if enabled and start is None:
                start = index
            elif not enabled and start is not None:
                if index - start >= 2:
                    runs.append((start, index - 1))
                start = None
        if not runs:
            return None
        start_bin, end_bin = max(
            runs,
            key=lambda run: (sum(bright_totals[run[0] : run[1] + 1]), run[1] - run[0], -run[0]),
        )
        if start_bin > int(0.55 * bin_count):
            return None
        padding = float(settings.get("appearance_extent_padding", 0.05))
        extent = float(np.clip((end_bin + 1) / bin_count + padding, 0.12, 1.0))
        if previous_extent_ratio is not None:
            current_weight = float(settings.get("appearance_extent_smoothing", 0.65))
            current_weight = float(np.clip(current_weight, 0.0, 1.0))
            extent = current_weight * extent + (1.0 - current_weight) * previous_extent_ratio
        return extent

    endpoint_quantile = float(np.clip(endpoint_quantile, 0.05, 0.45))
    low_quantile, high_quantile = np.quantile(
        projection, (endpoint_quantile, 1.0 - endpoint_quantile)
    )
    endpoint_boxes = [
        endpoint_box(projection <= low_quantile),
        endpoint_box(projection >= high_quantile),
    ]
    if (
        previous_box is not None
        and (not prefer_distal_each_frame)
        and all((box is not None for box in endpoint_boxes))
    ):
        previous_center = np.asarray(
            [(previous_box[0] + previous_box[2]) / 2, (previous_box[1] + previous_box[3]) / 2]
        )
        centers = [
            np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) for box in endpoint_boxes
        ]
        selected = endpoint_boxes[
            int(np.argmin([np.linalg.norm(c - previous_center) for c in centers]))
        ]
        return (selected or base_box, previous_extent_ratio)
    preferred = 0 if border_distance(low_point) > border_distance(high_point) else 1
    oriented_projection = projection if preferred == 0 else -projection
    extent_ratio = appearance_extent(oriented_projection)
    if extent_ratio is not None:
        minimum = float(oriented_projection.min())
        maximum = float(oriented_projection.max())
        adaptive_selector = oriented_projection <= minimum + extent_ratio * (maximum - minimum)
        adaptive_box = endpoint_box(adaptive_selector)
        if adaptive_box is not None:
            return (adaptive_box, extent_ratio)
    return (
        endpoint_boxes[preferred] or endpoint_boxes[1 - preferred] or base_box,
        previous_extent_ratio,
    )


def sample_points(mask: np.ndarray, count: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) < count:
        return None
    chosen = rng.choice(len(xs), size=count, replace=False)
    return np.column_stack((xs[chosen], ys[chosen])).astype(np.float32).reshape(-1, 1, 2)


def camera_motion(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    corners = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=160, qualityLevel=0.01, minDistance=12, blockSize=7
    )
    if corners is None or len(corners) < 8:
        return np.zeros(2, dtype=np.float32)
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        corners,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
    )
    if moved is None or status is None:
        return np.zeros(2, dtype=np.float32)
    good = status.ravel() == 1
    if good.sum() < 8:
        return np.zeros(2, dtype=np.float32)
    return np.median(moved[good, 0] - corners[good, 0], axis=0).astype(np.float32)


def update_motion(
    track: Track,
    prev_gray: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    global_shift: np.ndarray,
    rng: np.random.Generator,
) -> None:
    old_points = track.active_points
    new_points: List[np.ndarray] = []
    new_trajectories: List[deque] = []
    residual_speeds: List[float] = []
    if old_points is not None and len(old_points):
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            old_points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        )
        if moved is not None and status is not None:
            for index, (point, valid) in enumerate(zip(moved[:, 0], status.ravel())):
                x, y = (int(round(point[0])), int(round(point[1])))
                if valid and 0 <= y < mask.shape[0] and (0 <= x < mask.shape[1]) and mask[y, x]:
                    residual = point - old_points[index, 0] - global_shift
                    residual_speeds.append(float(np.linalg.norm(residual)))
                    history = (
                        track.trajectories[index]
                        if index < len(track.trajectories)
                        else deque(maxlen=20)
                    )
                    history.append(point.copy())
                    new_points.append(point)
                    new_trajectories.append(history)
    if residual_speeds:
        track.motion_samples.append(float(np.median(residual_speeds)))
    if len(new_points) < 8:
        sampled = sample_points(mask, 16, rng)
        if sampled is not None:
            track.active_points = sampled
            track.trajectories = [deque([point[0].copy()], maxlen=20) for point in sampled]
            return
    track.active_points = (
        np.asarray(new_points, dtype=np.float32).reshape(-1, 1, 2) if new_points else None
    )
    track.trajectories = new_trajectories


def motion_score(track: Track) -> Optional[float]:
    if len(track.motion_samples) < 8:
        return None
    values = np.asarray(track.motion_samples, dtype=np.float32)
    return float(np.median(values) + 0.25 * np.percentile(values, 90) + 0.1 * values.std())


def flow_recovery_mask(
    track: Track, prev_gray: Optional[np.ndarray], gray: np.ndarray, global_shift: np.ndarray
) -> Optional[np.ndarray]:
    """Recover a briefly lost confirmed mask using its own optical-flow points."""
    if prev_gray is None:
        return None
    recovery_points = track.active_points
    if recovery_points is None or len(recovery_points) < 4:
        recovery_points = sample_points(track.last_mask, 16, np.random.default_rng(2026))
    if recovery_points is None or len(recovery_points) < 4:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        recovery_points,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
    )
    if moved is None or status is None:
        return None
    valid = status.ravel() == 1
    if int(valid.sum()) < 4:
        return None
    displacement = np.median(moved[valid, 0] - recovery_points[valid, 0], axis=0).astype(np.float32)
    if float(np.linalg.norm(displacement)) > 80.0:
        return None
    height, width = gray.shape
    matrix = np.asarray(
        [[1.0, 0.0, float(displacement[0])], [0.0, 1.0, float(displacement[1])]], dtype=np.float32
    )
    recovered = cv2.warpAffine(
        track.last_mask.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    recovered = largest_component(recovered)
    return recovered if mask_to_box(recovered, min_pixels=20) is not None else None


class STTAMRunner:

    def __init__(self, args: argparse.Namespace, predictor, mask_generator) -> None:
        self.args = args
        self.predictor = predictor
        self.mask_generator = mask_generator
        self.rng = np.random.default_rng(args.seed)
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.state = None
        self.initial_classification_done = False
        self.motion_threshold: Optional[float] = None
        self.temp_dir: Optional[Path] = None
        self.yolo_detections: Optional[List[np.ndarray]] = None
        self.current_video_name: Optional[str] = None

    def discover(self, frame_bgr: np.ndarray, video_name: str) -> List[np.ndarray]:
        height, width = frame_bgr.shape[:2]
        image_area = height * width
        filter_settings = DISCOVERY_FILTER_SETTINGS_BY_VIDEO.get(video_name, {})
        min_area_ratio = max(
            self.args.min_area_ratio,
            float(filter_settings.get("min_area_ratio", self.args.min_area_ratio)),
        )
        min_thin_area_ratio = max(
            self.args.min_thin_area_ratio,
            float(filter_settings.get("min_thin_area_ratio", self.args.min_thin_area_ratio)),
        )
        reject_bottom_margin = int(filter_settings.get("reject_bottom_margin", 0))
        duplicate_containment = float(filter_settings.get("duplicate_containment", 0.88))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(frame_rgb)
        prompts = INITIAL_POINT_PROMPTS_BY_VIDEO.get(video_name, [])
        if prompts:
            self.mask_generator.predictor.set_image(frame_rgb)
            for prompt in prompts:
                guided_masks, scores, _ = self.mask_generator.predictor.predict(
                    point_coords=np.asarray(prompt["points"], dtype=np.float32),
                    point_labels=np.asarray(prompt["labels"], dtype=np.int32),
                    box=np.asarray(prompt["box"], dtype=np.float32) if "box" in prompt else None,
                    multimask_output=True,
                )
                best = int(np.argmax(scores))
                masks.append(
                    {
                        "segmentation": guided_masks[best].astype(bool),
                        "predicted_iou": float(scores[best]) + 1.0,
                    }
                )
            self.mask_generator.predictor.reset_predictor()
        eligible = []
        for item in masks:
            mask = largest_component(np.asarray(item["segmentation"], dtype=bool))
            area_ratio = float(mask.sum() / image_area)
            box = mask_to_box(mask)
            if box is None or area_ratio > self.args.max_area_ratio:
                continue
            if reject_bottom_margin and box[3] >= height - reject_bottom_margin:
                continue
            box_width, box_height = (box[2] - box[0] + 1, box[3] - box[1] + 1)
            elongation = max(box_width, box_height) / max(1, min(box_width, box_height))
            is_thin = elongation >= self.args.thin_elongation
            minimum_area = min_thin_area_ratio if is_thin else min_area_ratio
            if area_ratio < minimum_area:
                continue
            if elongation < 1.15 and area_ratio > 0.03:
                continue
            touches_edge = (
                box[0] <= 8 or box[1] <= 8 or box[2] >= width - 9 or (box[3] >= height - 9)
            )
            mean_brightness = float(gray[mask].mean()) if np.any(mask) else 0.0
            if touches_edge and area_ratio > 0.02 and (mean_brightness < 40.0):
                continue
            priority = (
                float(item["predicted_iou"])
                + (0.1 if is_thin else 0.0)
                + (0.08 if is_thin and touches_edge else 0.0)
            )
            eligible.append((priority, mask, box, area_ratio, elongation, mean_brightness))
        selected = []
        for _, mask, box, area_ratio, elongation, brightness in sorted(
            eligible, reverse=True, key=lambda x: x[0]
        ):
            duplicate_index = None
            for index, kept in enumerate(selected):
                mask_iou, mask_contained = mask_overlap(mask, kept[0])
                if (
                    mask_iou >= 0.45
                    or mask_contained >= 0.75
                    or box_containment(box, kept[1]) >= duplicate_containment
                ):
                    duplicate_index = index
                    break
            if duplicate_index is not None:
                if int(mask.sum()) > int(selected[duplicate_index][0].sum()):
                    selected[duplicate_index] = (mask, box, area_ratio, elongation, brightness)
                continue
            selected.append((mask, box, area_ratio, elongation, brightness))
            if len(self.tracks) + len(selected) >= self.args.max_tracks:
                break
        candidates = [item[0] for item in selected]
        for index, (mask, box, area_ratio, elongation, brightness) in enumerate(selected, 1):
            print(
                f"  candidate {index}: box={box}, area={int(mask.sum())} ({100.0 * area_ratio:.3f}%), elongation={elongation:.2f}, brightness={brightness:.1f}"
            )
        return candidates

    def add_candidates(
        self, masks: Iterable[np.ndarray], frame_index: int, frame_bgr: np.ndarray
    ) -> int:
        added = 0
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        image_height = gray.shape[0]
        for mask in masks:
            box = mask_to_box(mask)
            if box is None:
                continue
            box_width = box[2] - box[0] + 1
            box_height = box[3] - box[1] + 1
            box_mode = initial_box_mode(mask, box, self.args.refine_tool_tip_box)
            box_settings = BENCHMARK_BOX_SETTINGS_BY_VIDEO.get(self.current_video_name or "", {})
            force_tip_min_elongation = box_settings.get("force_tip_min_elongation")
            if box_settings.get("force_full"):
                box_mode = "full"
            elif force_tip_min_elongation is not None:
                initial_elongation = max(box_width, box_height) / max(1, min(box_width, box_height))
                initial_brightness = float(gray[mask].mean()) if np.any(mask) else 0.0
                brightness_limit = float(box_settings.get("force_tip_min_brightness", 0.0))
                center_y_ratio = (box[1] + box[3]) / 2.0 / max(1, image_height)
                center_y_limit = float(box_settings.get("force_tip_min_center_y_ratio", 0.0))
                box_mode = (
                    "tip"
                    if initial_elongation >= float(force_tip_min_elongation)
                    and initial_brightness >= brightness_limit
                    and (center_y_ratio >= center_y_limit)
                    else "full"
                )
            output_box, tip_extent_ratio = benchmark_box_from_mask(
                mask,
                box,
                box_mode,
                endpoint_quantile=float(box_settings.get("endpoint_quantile", 0.1)),
                prefer_distal_each_frame=bool(box_settings.get("prefer_distal_each_frame", False)),
                frame_bgr=frame_bgr,
                appearance_settings=box_settings,
            )
            track = Track(
                track_id=self.next_id,
                birth_frame=frame_index,
                last_mask=mask,
                last_box=box,
                output_box=output_box,
                box_mode=box_mode,
                active_points=sample_points(mask, 16, self.rng),
                initial_elongation=max(box_width, box_height) / max(1, min(box_width, box_height)),
                initial_brightness=float(gray[mask].mean()) if np.any(mask) else 0.0,
                initial_area_ratio=float(mask.sum() / mask.size),
                initial_box=box,
                tip_extent_ratio=tip_extent_ratio,
            )
            if track.active_points is not None:
                track.trajectories = [
                    deque([point[0].copy()], maxlen=20) for point in track.active_points
                ]
            self.tracks[track.track_id] = track
            print(f"  track {track.track_id}: fixed benchmark box mode={box_mode}")
            self.next_id += 1
            added += 1
        return added

    def restart_tracker(self, frame_index: int) -> None:
        if not self.tracks:
            self.state = None
            return
        self.state = self.predictor.init_state(video_path=str(self.temp_dir), is_realtime=True)
        self.state["num_frames"] = frame_index + 1
        for track in self.tracks.values():
            self.predictor.add_new_mask(
                inference_state=self.state,
                frame_idx=frame_index,
                obj_id=track.track_id,
                mask=track.last_mask,
            )

    def classify_initial(self, frame_index: int, video_name: str) -> None:
        scored = [(track, motion_score(track)) for track in self.tracks.values()]
        scored = [(track, score) for track, score in scored if score is not None]
        score_by_id = {track.track_id: score for track, score in scored}
        print(
            f"[{frame_index:06d}] candidate motion scores: "
            + ", ".join((f"ID{track.track_id}={score:.3f}" for track, score in scored))
        )
        if len(scored) >= 2:
            features = np.asarray([[score] for _, score in scored], dtype=np.float32)
            labels = KMeans(n_clusters=2, n_init=10, random_state=self.args.seed).fit_predict(
                features
            )
            means = [
                (
                    float(features[labels == label].mean())
                    if np.any(labels == label)
                    else -float("inf")
                )
                for label in (0, 1)
            ]
            instrument_label = int(np.argmax(means))
            self.motion_threshold = float(np.mean([m for m in means if np.isfinite(m)]))
            for (track, _), label in zip(scored, labels):
                track.status = "instrument" if label == instrument_label else "background"
        else:
            self.motion_threshold = self.args.fallback_motion_threshold
            for track, score in scored:
                track.status = "instrument" if score >= self.motion_threshold else "background"
        settings = MOTION_CLASSIFICATION_SETTINGS_BY_VIDEO.get(video_name)
        appearance_rejected = []
        if settings:
            min_brightness = settings.get("primary_min_brightness")
            min_elongation = settings.get("primary_min_elongation")
            max_brightness = settings.get("primary_max_brightness")
            max_elongation = settings.get("primary_max_elongation")
            for track in self.tracks.values():
                if track.status != "instrument":
                    continue
                reject_dark_broad = (
                    min_brightness is not None
                    and min_elongation is not None
                    and (track.initial_brightness < float(min_brightness))
                    and (track.initial_elongation < float(min_elongation))
                )
                reject_bright_broad = (
                    max_brightness is not None
                    and max_elongation is not None
                    and (track.initial_brightness > float(max_brightness))
                    and (track.initial_elongation < float(max_elongation))
                )
                if reject_dark_broad or reject_bright_broad:
                    track.status = "background"
                    appearance_rejected.append(track.track_id)
        if appearance_rejected:
            print(
                f"[{frame_index:06d}] rejected appearance-incompatible high-motion candidate IDs={appearance_rejected}"
            )
        rescued = []
        if settings and self.motion_threshold is not None:
            for track in self.tracks.values():
                if track.status != "background" or track.last_box is None:
                    continue
                score = score_by_id.get(track.track_id, 0.0)
                if (
                    score >= self.motion_threshold * float(settings["secondary_threshold_ratio"])
                    and track.initial_elongation >= float(settings["secondary_min_elongation"])
                    and (track.initial_brightness >= float(settings["secondary_min_brightness"]))
                    and (
                        track.initial_area_ratio
                        >= float(settings.get("secondary_min_area_ratio", 0.0))
                    )
                    and (
                        track.initial_area_ratio
                        <= float(settings.get("secondary_max_area_ratio", 1.0))
                    )
                ):
                    track.status = "instrument"
                    rescued.append(track.track_id)
        if rescued:
            print(f"[{frame_index:06d}] retained elongated moving candidate IDs={rescued}")
        duplicate_rejected = []
        if settings and settings.get("classified_duplicate_box_containment") is not None:
            containment_threshold = float(settings["classified_duplicate_box_containment"])
            instruments = [
                track
                for track in self.tracks.values()
                if track.status == "instrument" and track.initial_box is not None
            ]
            instruments.sort(
                key=lambda track: (track.initial_brightness, track.initial_area_ratio), reverse=True
            )
            kept = []
            for track in instruments:
                if any(
                    (
                        box_containment(track.initial_box, other.initial_box)
                        >= containment_threshold
                        for other in kept
                    )
                ):
                    track.status = "background"
                    duplicate_rejected.append(track.track_id)
                else:
                    kept.append(track)
        if duplicate_rejected:
            print(
                f"[{frame_index:06d}] merged overlapping instrument candidate IDs={duplicate_rejected}"
            )
        adjacent_rejected = []
        if settings and settings.get("classified_adjacent_merge_gap") is not None:
            gap_limit = float(settings["classified_adjacent_merge_gap"])
            motion_ratio_limit = float(settings.get("classified_adjacent_motion_ratio", 0.0))
            union_elongation_limit = float(
                settings.get("classified_adjacent_min_union_elongation", 1.0)
            )
            instruments = [
                track
                for track in self.tracks.values()
                if track.status == "instrument" and track.initial_box is not None
            ]
            adjacent_rescued = []
            for candidate in self.tracks.values():
                if candidate.status != "background" or candidate.initial_box is None:
                    continue
                candidate_score = score_by_id.get(candidate.track_id, 0.0)
                if candidate.initial_brightness < float(
                    settings.get("classified_adjacent_min_brightness", 0.0)
                ) or candidate.initial_area_ratio > float(
                    settings.get("classified_adjacent_max_area_ratio", 1.0)
                ):
                    continue
                for confirmed in instruments:
                    confirmed_score = score_by_id.get(confirmed.track_id, 0.0)
                    motion_ratio = min(candidate_score, confirmed_score) / max(
                        1e-06, max(candidate_score, confirmed_score)
                    )
                    combined_box = union_box(candidate.initial_box, confirmed.initial_box)
                    if (
                        box_gap(candidate.initial_box, confirmed.initial_box) <= gap_limit
                        and motion_ratio >= motion_ratio_limit
                        and (box_elongation(combined_box) >= union_elongation_limit)
                    ):
                        candidate.status = "instrument"
                        instruments.append(candidate)
                        adjacent_rescued.append(candidate.track_id)
                        break
            if adjacent_rescued:
                print(
                    f"[{frame_index:06d}] recovered adjacent tool fragments IDs={adjacent_rescued}"
                )
            consumed: Set[int] = set()
            for index, first in enumerate(instruments):
                if first.track_id in consumed:
                    continue
                for second in instruments[index + 1 :]:
                    if second.track_id in consumed:
                        continue
                    first_score = score_by_id.get(first.track_id, 0.0)
                    second_score = score_by_id.get(second.track_id, 0.0)
                    motion_ratio = min(first_score, second_score) / max(
                        1e-06, max(first_score, second_score)
                    )
                    combined_box = union_box(first.initial_box, second.initial_box)
                    if (
                        box_gap(first.initial_box, second.initial_box) > gap_limit
                        or motion_ratio < motion_ratio_limit
                        or box_elongation(combined_box) < union_elongation_limit
                    ):
                        continue
                    if settings.get("classified_adjacent_prefer_lateral_distal"):
                        image_width = first.last_mask.shape[1]
                        enters_from_left = combined_box[0] <= image_width - 1 - combined_box[2]
                        first_center_x = (first.initial_box[0] + first.initial_box[2]) / 2.0
                        second_center_x = (second.initial_box[0] + second.initial_box[2]) / 2.0
                        if enters_from_left:
                            keeper, rejected = (
                                (first, second)
                                if first_center_x >= second_center_x
                                else (second, first)
                            )
                        else:
                            keeper, rejected = (
                                (first, second)
                                if first_center_x <= second_center_x
                                else (second, first)
                            )
                    else:
                        keeper, rejected = (
                            max(
                                (first, second),
                                key=lambda track: (
                                    track.initial_brightness,
                                    track.initial_area_ratio,
                                ),
                            ),
                            min(
                                (first, second),
                                key=lambda track: (
                                    track.initial_brightness,
                                    track.initial_area_ratio,
                                ),
                            ),
                        )
                    distal_anchor = keeper.output_box
                    merged_mask = np.logical_or(keeper.last_mask, rejected.last_mask)
                    merged_box = mask_to_box(merged_mask)
                    if merged_box is not None:
                        keeper.last_mask = merged_mask
                        keeper.last_box = merged_box
                        keeper.output_box = merged_box
                        keeper.initial_box = union_box(keeper.initial_box, rejected.initial_box)
                        keeper.initial_area_ratio = float(merged_mask.sum() / merged_mask.size)
                        keeper.initial_elongation = box_elongation(keeper.initial_box)
                        box_settings = BENCHMARK_BOX_SETTINGS_BY_VIDEO.get(video_name, {})
                        force_tip = box_settings.get("force_tip_min_elongation")
                        keeper.box_mode = (
                            "tip"
                            if force_tip is not None
                            and keeper.initial_elongation >= float(force_tip)
                            else "full"
                        )
                        keeper.output_box, keeper.tip_extent_ratio = benchmark_box_from_mask(
                            merged_mask,
                            merged_box,
                            keeper.box_mode,
                            previous_box=distal_anchor,
                            endpoint_quantile=float(box_settings.get("endpoint_quantile", 0.1)),
                            prefer_distal_each_frame=bool(
                                box_settings.get("prefer_distal_each_frame", False)
                            ),
                        )
                        keeper.active_points = sample_points(merged_mask, 16, self.rng)
                        if keeper.active_points is not None:
                            keeper.trajectories = [
                                deque([point[0].copy()], maxlen=20)
                                for point in keeper.active_points
                            ]
                    rejected.status = "background"
                    consumed.add(rejected.track_id)
                    consumed.add(keeper.track_id)
                    adjacent_rejected.append((rejected.track_id, keeper.track_id))
                    if rejected is first:
                        break
        if adjacent_rejected:
            print(
                f"[{frame_index:06d}] collapsed adjacent same-motion fragments "
                + ", ".join((f"ID{rejected}->ID{keeper}" for rejected, keeper in adjacent_rejected))
            )
        self.initial_classification_done = True
        instruments = [
            track.track_id for track in self.tracks.values() if track.status == "instrument"
        ]
        print(
            f"[{frame_index:06d}] initial classification: threshold={self.motion_threshold:.3f}, instrument IDs={instruments}"
        )
        self.drop_background_tracks(frame_index)

    def drop_background_tracks(self, frame_index: int) -> None:
        rejected = [
            track_id for track_id, track in self.tracks.items() if track.status == "background"
        ]
        if not rejected:
            return
        for track_id in rejected:
            del self.tracks[track_id]
        print(f"[{frame_index:06d}] removed background candidate IDs={rejected}")
        self.restart_tracker(frame_index)

    def propagate(self, frame_index: int) -> Dict[int, np.ndarray]:
        if self.state is None:
            return {}
        self.state["num_frames"] = frame_index + 1
        result: Dict[int, np.ndarray] = {}
        for out_index, object_ids, logits in self.predictor.propagate_in_video(
            self.state, start_frame_idx=frame_index, max_frame_num_to_track=1
        ):
            if out_index != frame_index:
                continue
            for position, object_id in enumerate(object_ids):
                result[int(object_id)] = logits[position].detach().float().cpu().numpy().squeeze()
            break
        return result

    def update_tracks(
        self,
        logits_by_id: Dict[int, np.ndarray],
        prev_gray: Optional[np.ndarray],
        gray: np.ndarray,
        frame_bgr: np.ndarray,
        frame_index: int,
    ) -> None:
        shift = camera_motion(prev_gray, gray) if prev_gray is not None else np.zeros(2)
        height, width = gray.shape
        min_pixels = max(20, int(height * width * self.args.min_thin_area_ratio * 0.2))
        recovered_any = False
        propagation_settings = MASK_PROPAGATION_SETTINGS_BY_VIDEO.get(
            self.current_video_name or "", {}
        )
        exclusive_masks: Dict[int, np.ndarray] = {}
        if propagation_settings.get("exclusive_logits") and len(logits_by_id) >= 2:
            object_ids = list(logits_by_id)
            logit_stack = np.stack([logits_by_id[object_id] for object_id in object_ids], axis=0)
            winning_object = np.argmax(logit_stack, axis=0)
            for position, object_id in enumerate(object_ids):
                exclusive_masks[object_id] = (logit_stack[position] > 0.0) & (
                    winning_object == position
                )
        for track_id, track in list(self.tracks.items()):
            logits = logits_by_id.get(track_id)
            if logits is None:
                recovered = (
                    flow_recovery_mask(track, prev_gray, gray, shift)
                    if track.status == "instrument"
                    and track.lost_frames < self.args.flow_recovery_frames
                    else None
                )
                if recovered is not None:
                    track.last_mask = recovered
                    track.last_box = mask_to_box(recovered)
                    track.output_box = track.last_box
                    track.last_score = max(0.5, track.last_score * 0.98)
                    track.lost_frames += 1
                    track.active_points = sample_points(recovered, 16, self.rng)
                    recovered_any = True
                    continue
                track.lost_frames += 1
                continue
            mask = largest_component(exclusive_masks.get(track_id, logits > 0.0))
            box = mask_to_box(mask, min_pixels=min_pixels)
            score = confidence_from_logits(logits, mask)
            if box is None or score < 0.52:
                recovered = (
                    flow_recovery_mask(track, prev_gray, gray, shift)
                    if track.status == "instrument"
                    and track.lost_frames < self.args.flow_recovery_frames
                    else None
                )
                if recovered is not None:
                    track.last_mask = recovered
                    track.last_box = mask_to_box(recovered)
                    track.output_box = track.last_box
                    track.last_score = max(0.5, track.last_score * 0.98)
                    track.lost_frames += 1
                    track.active_points = sample_points(recovered, 16, self.rng)
                    recovered_any = True
                    continue
                track.lost_frames += 1
            else:
                track.last_mask = mask
                track.last_box = box
                track.output_box, track.tip_extent_ratio = benchmark_box_from_mask(
                    mask,
                    box,
                    track.box_mode,
                    previous_box=track.output_box,
                    endpoint_quantile=float(
                        BENCHMARK_BOX_SETTINGS_BY_VIDEO.get(self.current_video_name or "", {}).get(
                            "endpoint_quantile", 0.1
                        )
                    ),
                    prefer_distal_each_frame=bool(
                        BENCHMARK_BOX_SETTINGS_BY_VIDEO.get(self.current_video_name or "", {}).get(
                            "prefer_distal_each_frame", False
                        )
                    ),
                    frame_bgr=frame_bgr,
                    appearance_settings=BENCHMARK_BOX_SETTINGS_BY_VIDEO.get(
                        self.current_video_name or "", {}
                    ),
                    previous_extent_ratio=track.tip_extent_ratio,
                )
                if track.box_mode == "full":
                    box_settings = BENCHMARK_BOX_SETTINGS_BY_VIDEO.get(
                        self.current_video_name or "", {}
                    )
                    track.output_box = preserve_full_box_short_side(
                        track.output_box,
                        track.initial_box,
                        float(box_settings.get("full_min_short_side_ratio", 0.0)),
                        mask.shape,
                    )
                track.last_score = score
                track.lost_frames = 0
                track.age += 1
                if prev_gray is not None:
                    update_motion(track, prev_gray, gray, mask, shift, self.rng)
        if recovered_any:
            self.restart_tracker(frame_index)
        max_lost_frames = int(
            MAX_LOST_FRAMES_BY_VIDEO.get(self.current_video_name or "", self.args.max_lost_frames)
        )
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.lost_frames > max_lost_frames
        ]
        for track_id in expired:
            del self.tracks[track_id]
            print(f"track {track_id} expired after {max_lost_frames} lost frames")

    def assign_yolo_boxes(self, frame_index: int) -> None:
        """Assign shared detector boxes to stTAM identities one-to-one."""
        if self.yolo_detections is None or frame_index >= len(self.yolo_detections):
            return
        active = [
            track
            for track in self.tracks.values()
            if not track.lost_frames and track.output_box is not None
        ]
        detections = self.yolo_detections[frame_index]
        if not active or not len(detections):
            return
        track_boxes = np.asarray(
            [
                [t.output_box[0], t.output_box[1], t.output_box[2] + 1, t.output_box[3] + 1]
                for t in active
            ],
            dtype=np.float32,
        )
        det_boxes = detections[:, :4]
        left = np.maximum(track_boxes[:, None, 0], det_boxes[None, :, 0])
        top = np.maximum(track_boxes[:, None, 1], det_boxes[None, :, 1])
        right = np.minimum(track_boxes[:, None, 2], det_boxes[None, :, 2])
        bottom = np.minimum(track_boxes[:, None, 3], det_boxes[None, :, 3])
        intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
        track_area = (track_boxes[:, 2] - track_boxes[:, 0]) * (
            track_boxes[:, 3] - track_boxes[:, 1]
        )
        det_area = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
        union = track_area[:, None] + det_area[None, :] - intersection
        similarities = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union > 0
        )
        rows, columns = linear_sum_assignment(1.0 - similarities)
        for row, column in zip(rows, columns):
            if similarities[row, column] < self.args.box_association_iou:
                continue
            box = det_boxes[column]
            active[row].output_box = (
                int(round(box[0])),
                int(round(box[1])),
                int(round(box[2] - 1)),
                int(round(box[3] - 1)),
            )

    def prune_state(self, frame_index: int, keep_frames: int = 32) -> None:
        if self.state is None:
            return
        cutoff = frame_index - keep_frames
        for output in [self.state["output_dict"], *self.state["output_dict_per_obj"].values()]:
            old = [index for index in output["non_cond_frame_outputs"] if index < cutoff]
            for index in old:
                output["non_cond_frame_outputs"].pop(index, None)
        for index in list(self.state["frames_already_tracked"]):
            if index < cutoff:
                self.state["frames_already_tracked"].pop(index, None)

    def mot_lines(self, frame_index: int) -> List[str]:
        lines = []
        for track in self.tracks.values():
            if track.status != "instrument" or track.lost_frames or track.output_box is None:
                continue
            x1, y1, x2, y2 = track.output_box
            width, height = (x2 - x1 + 1, y2 - y1 + 1)
            lines.append(
                f"{frame_index},{track.track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{track.last_score:.6f},-1,-1,-1\n"
            )
        return lines

    def render_frame(self, frame: np.ndarray, local_frame_index: int) -> np.ndarray:
        canvas = frame.copy()
        palette = ((0, 0, 255), (255, 128, 0), (0, 220, 255), (255, 0, 180))
        instrument_ids = []
        for track in self.tracks.values():
            if track.output_box is None or track.lost_frames:
                continue
            if track.status == "background":
                continue
            x1, y1, x2, y2 = track.output_box
            if track.status == "instrument":
                instrument_ids.append(track.track_id)
                color = palette[(track.track_id - 1) % len(palette)]
                label = f"ID {track.track_id} [{track.box_mode}]"
            else:
                color = (180, 180, 180)
                label = f"candidate {track.track_id} [{track.box_mode}]"
            mask = np.asarray(track.last_mask, dtype=bool)
            if mask.shape == canvas.shape[:2] and np.any(mask):
                pixels = canvas[mask].astype(np.float32)
                tint = np.asarray(color, dtype=np.float32)
                canvas[mask] = (
                    pixels * (1.0 - self.args.mask_alpha) + tint * self.args.mask_alpha
                ).astype(np.uint8)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas,
                label,
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        state_text = "classified" if self.initial_classification_done else "classifying"
        status = f"local frame {local_frame_index} | {state_text} | IDs {instrument_ids}"
        cv2.putText(
            canvas,
            status,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA
        )
        return canvas


class EndoVisRunner(STTAMRunner):
    """EndoVis automatic candidates and two-dimensional motion features."""

    def discover(self, frame_bgr, video_name):
        settings = self.args.endovis
        return [
            m["segmentation"].astype(bool)
            for m in self.mask_generator.generate(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if settings["candidate_min_area"] < m["area"] < settings["candidate_max_area"]
        ]

    def add_candidates(self, masks, frame_index, frame_bgr):
        self.initial_prompts = {}
        for mask in masks:
            positive, negative = sample_prompt_points(
                mask, self.args.endovis["initial_positive_points"], 5
            )
            if positive is None:
                continue
            track = Track(self.next_id, frame_index, mask, last_box=mask_to_box(mask))
            track.output_box = track.last_box
            track.active_points = positive[:TRACKING_POINTS].reshape(-1, 1, 2)
            track.trajectories = [[p[0].tolist()] for p in track.active_points]
            self.initial_prompts[self.next_id] = build_prompt_arrays(positive, negative)
            self.tracks[self.next_id] = track
            self.next_id += 1
        return len(self.tracks)

    def restart_tracker(self, frame_index):
        if not self.tracks:
            self.state = None
            return
        self.state = self.predictor.init_state(str(self.temp_dir), is_realtime=True)
        self.state["num_frames"] = frame_index + 1
        for track in self.tracks.values():
            points, labels = self.initial_prompts[track.track_id]
            self.predictor.add_new_points_or_box(
                self.state, frame_index, track.track_id, points=points, labels=labels
            )

    def update_tracks(self, logits_by_id, prev_gray, gray, frame_bgr, frame_index):
        lk = dict(
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        for track in self.tracks.values():
            logits = logits_by_id.get(track.track_id)
            gate = None if logits is None else logits > self.args.endovis["track_mask_threshold"]
            if prev_gray is not None:
                track.active_points, track.trajectories, _ = update_anchor_points(
                    prev_gray, gray, track.active_points, track.trajectories, gate, lk
                )
            if logits is None:
                track.lost_frames += 1
                continue
            mask = logits > 0.0
            track.last_mask = mask
            track.last_box = track.output_box = mask_to_box(mask)
            track.last_score = confidence_from_logits(logits, mask)
            track.lost_frames = 0 if track.last_box is not None else track.lost_frames + 1

    def classify_initial(self, frame_index, video_name):
        tracks = list(self.tracks.values())
        labels, details = classify_instrument_candidates(
            [t.trajectories for t in tracks], len(tracks)
        )
        for track, label in zip(tracks, labels):
            track.status = "instrument" if label == 0 else "background"
        self.initial_classification_done = True
        print("Motion classification:", details)


def load_detections(path, frame_count, video_path):
    """Load frame-aligned shared boxes; no detector training happens here."""
    if path is None or not path.is_file():
        raise FileNotFoundError(
            "Shared mode needs --detection-cache with one detection list per input frame. Use --box-source mask for the detector-free demo."
        )
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("detections")
    if not isinstance(rows, list) or len(rows) < frame_count:
        raise ValueError(f"Detection cache must cover at least {frame_count} input frames.")
    parsed = []
    for index, row in enumerate(rows[:frame_count]):
        det = np.asarray(row, dtype=np.float32)
        if not det.size:
            det = np.empty((0, 6), dtype=np.float32)
        if det.ndim != 2 or det.shape[1] != 6 or not np.isfinite(det).all():
            raise ValueError(f"Invalid detections at frame {index}; expected Nx6 finite values.")
        if len(det) and (
            (det[:, 2:4] <= det[:, :2]).any() or (det[:, 4] < 0).any() or (det[:, 4] > 1).any()
        ):
            raise ValueError(f"Invalid box coordinates or confidence at frame {index}.")
        parsed.append(det)
    # Original caches may name a moved source file. Frame alignment is the
    # caller's responsibility; never silently shift an absolute-frame cache.
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--config", type=Path, default=root / "configs/endovis2017.yaml")
    parser.add_argument(
        "--input", required=True, type=Path, help="Video or naturally sorted frame directory"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--box-source", choices=["mask", "yolo11"])
    parser.add_argument("--detection-cache", type=Path)
    parser.add_argument("--show", action="store_true")
    cli = parser.parse_args(argv)
    with cli.config.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if cli.max_frames is not None and cli.max_frames < 1:
        parser.error("--max-frames must be positive")
    if not cli.checkpoint.is_file():
        parser.error(f"EdgeTAM checkpoint not found: {cli.checkpoint}")
    if not cli.input.exists():
        parser.error(f"Input does not exist: {cli.input}")
    cli.output.mkdir(parents=True, exist_ok=True)
    if cli.input.is_file() and cli.input.resolve() == (cli.output / "tracking.mp4").resolve():
        parser.error("The output video cannot overwrite the input video")
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if cli.device == "auto" else cli.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation")
    global RANDOM_SEED, HISTORY_LENGTH, TRACKING_POINTS, MIN_VALID_POINTS
    RANDOM_SEED = int(config["seed"])
    settings = config.get("endovis", {})
    HISTORY_LENGTH = int(settings.get("history_length", 20))
    TRACKING_POINTS = int(settings.get("tracking_points", 10))
    MIN_VALID_POINTS = int(settings.get("min_valid_points", 5))
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    args = argparse.Namespace(**config["tracker"])
    args.seed = RANDOM_SEED
    args.endovis = settings
    args.benchmark_box_source = cli.box_source or config["box_source"]
    args.mask_alpha = 0.35
    for name, mapping in [
        ("discovery", DISCOVERY_FILTER_SETTINGS_BY_VIDEO),
        ("motion", MOTION_CLASSIFICATION_SETTINGS_BY_VIDEO),
        ("boxes", BENCHMARK_BOX_SETTINGS_BY_VIDEO),
        ("propagation", MASK_PROPAGATION_SETTINGS_BY_VIDEO),
    ]:
        mapping.clear()
        mapping["demo"] = config.get(name, {})
    capture = None
    if cli.input.is_dir():
        paths = sorted(
            [
                p
                for p in cli.input.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            ],
            key=lambda p: [
                int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)
            ],
        )
        total = len(paths)
        fps = float(config.get("frames_fps", 5))
        frames = (cv2.imread(str(p)) for p in paths)
    else:
        capture = cv2.VideoCapture(str(cli.input))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {cli.input}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))

        def video_frames():
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                yield frame

        frames = video_frames()
    limit = min(total, cli.max_frames) if cli.max_frames else total
    if limit <= 0 or not math.isfinite(fps) or fps <= 0:
        if capture is not None:
            capture.release()
        raise ValueError("Input needs decodable frames and a valid FPS")
    observation = int(config["classification_frame"])
    if limit <= observation:
        print(
            f"Smoke run: {limit} frames will not complete the {observation + 1}-frame observation window."
        )
    detections = None
    if args.benchmark_box_source == "yolo11":
        detections = load_detections(cli.detection_cache, limit, cli.input)
    print(f"Loading frozen EdgeTAM on {device}; box source: {args.benchmark_box_source}")
    # Reuse one frozen network for automatic masks and video propagation.
    predictor = build_sam2_video_predictor(
        config["model_cfg"],
        str(cli.checkpoint),
        device=device,
        hydra_overrides_extra=["++model.fill_hole_area=0"],
    )
    predictor.__class__ = StreamingPredictor
    mask_generator = SAM2AutomaticMaskGenerator(model=predictor, **config["mask_generator"])
    cls = EndoVisRunner if config["protocol"] == "endovis2017" else STTAMRunner
    runner = cls(args, predictor, mask_generator)
    runner.current_video_name = "demo"
    runner.yolo_detections = detections
    writer = None
    processed = 0
    previous_gray = None
    try:
        with tempfile.TemporaryDirectory(prefix="sttam_demo_") as temp, (
            cli.output / "tracks.txt"
        ).open("w", encoding="utf-8") as output, torch.inference_mode(), torch.autocast(
            device_type=device, dtype=torch.float16, enabled=device == "cuda"
        ):
            runner.temp_dir = Path(temp)
            for index, frame in enumerate(frames):
                if index >= limit:
                    break
                if frame is None:
                    raise RuntimeError(f"Cannot decode input frame {index}")
                if index == 0:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(cli.output / "tracking.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create output MP4")
                if frame.shape[:2] != (h, w):
                    raise ValueError("All input frames must have the same dimensions")
                if not cv2.imwrite(str(runner.temp_dir / f"{index:06d}.jpg"), frame):
                    raise RuntimeError("Cannot write temporary frame")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if index == 0:
                    candidates = runner.discover(frame, "demo")
                    runner.add_candidates(candidates, 0, frame)
                    print(f"Discovered {len(runner.tracks)} candidates")
                    if not runner.tracks:
                        raise RuntimeError("No automatic candidate found in the first frame")
                    runner.restart_tracker(0)
                logits = runner.propagate(index)
                runner.update_tracks(logits, previous_gray, gray, frame, index)
                runner.assign_yolo_boxes(index)
                if not runner.initial_classification_done and index >= observation:
                    runner.classify_initial(index, "demo")
                if runner.initial_classification_done:
                    output.writelines(runner.mot_lines(index + 1))
                canvas = runner.render_frame(frame, index)
                writer.write(canvas)
                processed += 1
                if cli.show:
                    cv2.imshow("stTAM", canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                runner.prune_state(index)
                old = runner.temp_dir / f"{index-2:06d}.jpg"
                if index > 2 and old.exists():
                    old.unlink()
                previous_gray = gray
                if processed % 25 == 0:
                    print(f"Processed {processed}/{limit} frames", flush=True)
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if cli.show:
            cv2.destroyAllWindows()
        runner.state = None
    print(f"Saved {processed} frames to {cli.output.resolve()}")
    return processed


if __name__ == "__main__":
    main()
