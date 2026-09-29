from __future__ import annotations
import argparse
import json
import math
import os
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
from sam2.modeling.perceiver import PerceiverResampler, window_partition
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import load_video_frames

Box = Tuple[int, int, int, int]
USE_MASK_GATE = True
USE_RESAMPLING = True
HISTORY_LENGTH = 20
MIN_VALID_POINTS = 5
TRACKING_POINTS = 10
RANDOM_SEED = 0


class BatchedPerceiverResampler(PerceiverResampler):
    def forward_2d(self, x):
        batch, channels, height, width = x.shape
        latents = self.latents_2d.unsqueeze(0).expand(batch, -1, -1).reshape(-1, 1, channels)
        windows = int(math.sqrt(self.num_latents_2d))
        x = window_partition(x.permute(0, 2, 3, 1), height // windows).flatten(1, 2)
        for layer in self.layers:
            latents = layer(latents, x)
        latents = latents.reshape(batch, windows, windows, channels).permute(0, 3, 1, 2)
        position = self.position_encoding(latents).permute(0, 2, 3, 1).flatten(1, 2)
        latents = self.norm(latents.permute(0, 2, 3, 1).flatten(1, 2))
        return latents, position


class StreamingPredictor(SAM2VideoPredictor):


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
        }
        if is_realtime:
            inference_state["video_dir"] = video_path
        if not is_realtime or (is_realtime and images is not None):
            self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    def _get_image_feature(self, inference_state, frame_idx, batch_size):

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
                mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
                std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
                image = (image - mean) / std
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
    feature_list = []
    valid_object_indices = []
    feature_details = {}
    for object_index, object_histories in enumerate(all_histories):
        variances = []
        speeds = []
        for trajectory in object_histories:
            points = np.asarray(trajectory, dtype=np.float64)
            if len(points) < 5 or not np.isfinite(points).all():
                continue
            displacement = np.diff(points, axis=0)
            lengths = np.linalg.norm(displacement, axis=1)
            speeds.append(float(lengths.mean()))
            nonzero = displacement[lengths > 0.0]
            if len(nonzero) > 5:
                angles = np.arctan2(nonzero[:, 1], nonzero[:, 0])
                resultant = np.hypot(np.cos(angles).mean(), np.sin(angles).mean())
                variances.append(float(np.clip(1.0 - resultant, 0.0, 1.0)))
        if variances and speeds:
            variance = float(np.median(variances))
            speed = float(np.median(speeds))
            feature_list.append([variance, speed])
            valid_object_indices.append(object_index)
            feature_details[object_index] = {
                "median_circular_variance": variance,
                "median_speed": speed,
                "direction_valid_trajectories": len(variances),
                "displacement_valid_trajectories": len(speeds),
            }
    return np.asarray(feature_list, dtype=np.float64).reshape(-1, 2), valid_object_indices, feature_details


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
    feature_array = (feature_array - feature_array.min(axis=0)) / (
        np.ptp(feature_array, axis=0) + 1e-8
    )
    classification_log["normalized_features"] = feature_array.tolist()
    if np.unique(feature_array, axis=0).shape[0] < 2:
        for index in valid_object_indices:
            final_labels[index] = 0
        classification_log["fallback"] = "identical_features_keep_valid"
        return final_labels, classification_log
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
    output_box: Optional[Tuple[float, float, float, float]] = None
    last_score: float = 0.0
    lost_frames: int = 0
    active_points: Optional[np.ndarray] = None
    trajectories: List[List[List[float]]] = field(default_factory=list)


def mask_to_box(mask: np.ndarray, min_pixels: int = 20) -> Optional[Box]:
    ys, xs = np.where(mask)
    if len(xs) < min_pixels:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def confidence_from_logits(logits: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits[mask], -30.0, 30.0)))
    return float(probability.mean())


class STTAMRunner:

    def __init__(self, args: argparse.Namespace, predictor, mask_generator) -> None:
        self.args = args
        self.predictor = predictor
        self.mask_generator = mask_generator
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.state = None
        self.initial_classification_done = False
        self.temp_dir: Optional[Path] = None
        self.yolo_detections: Optional[List[np.ndarray]] = None
        self.initial_prompts = {}

    def discover(self, frame_bgr, video_name):
        settings = self.args.anchors
        image_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        minimum = settings.get("candidate_min_area", settings.get("candidate_min_area_ratio", 0.002) * image_area)
        maximum = settings.get("candidate_max_area", settings.get("candidate_max_area_ratio", 0.2) * image_area)
        return [
            m["segmentation"].astype(bool)
            for m in self.mask_generator.generate(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if minimum < m["area"] < maximum
        ]

    def add_candidates(self, masks, frame_index, frame_bgr):
        self.initial_prompts = {}
        for mask in masks:
            positive, negative = sample_prompt_points(
                mask, self.args.anchors["initial_positive_points"], 5
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

    def update_tracks(self, logits_by_id, prev_gray, gray, frame_bgr, frame_index):
        lk = dict(
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        for track in self.tracks.values():
            if track.status == "background":
                continue
            logits = logits_by_id.get(track.track_id)
            gate = None if logits is None else logits > self.args.anchors["track_mask_threshold"]
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

            if frame_index > 0 and track.active_points is not None and len(track.active_points):
                points, labels = build_prompt_arrays(track.active_points, None)
                self.predictor.add_new_points_or_box(
                    self.state, frame_index, track.track_id,
                    points=points, labels=labels, clear_old_points=True,
                )

    def classify_initial(self, frame_index, video_name):
        tracks = list(self.tracks.values())
        labels, details = classify_instrument_candidates(
            [t.trajectories for t in tracks], len(tracks)
        )
        for track, label in zip(tracks, labels):
            track.status = "instrument" if label == 0 else "background"
        self.initial_classification_done = True
        print("Motion classification:", details)

    def assign_yolo_boxes(self, frame_index: int) -> None:

        if self.args.benchmark_box_source != "yolo11":
            return
        if self.yolo_detections is None or frame_index >= len(self.yolo_detections):
            for track in self.tracks.values():
                track.output_box = None
            raise ValueError(f"Missing shared detections for frame {frame_index}")
        active = [
            track
            for track in self.tracks.values()
            if track.status != "background" and not track.lost_frames and track.last_box is not None
        ]
        for track in self.tracks.values():
            track.output_box = None
        detections = self.yolo_detections[frame_index]
        if not active or not len(detections):
            return
        track_boxes = np.asarray(
            [
                [t.last_box[0], t.last_box[1], t.last_box[2] + 1, t.last_box[3] + 1]
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
        allowed = similarities >= self.args.box_association_iou
        cost = np.where(allowed, 1.0 - similarities, len(active) + len(detections) + 1.0)
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns):
            if similarities[row, column] < self.args.box_association_iou:
                continue
            box = det_boxes[column]
            active[row].output_box = (
                float(box[0]),
                float(box[1]),
                float(box[2]) - 1.0,
                float(box[3]) - 1.0,
            )
            active[row].last_score = float(detections[column, 4])

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
        old_prompts = {
            index for index in self.state["consolidated_frame_inds"]["non_cond_frame_outputs"]
            if index < cutoff
        }
        self.state["consolidated_frame_inds"]["non_cond_frame_outputs"].difference_update(old_prompts)
        for key in ("point_inputs_per_obj", "mask_inputs_per_obj"):
            for inputs in self.state[key].values():
                for index in old_prompts:
                    inputs.pop(index, None)

    def mot_lines(self, frame_index: int) -> List[str]:
        lines = []
        for track in self.tracks.values():
            if track.status != "instrument" or track.lost_frames or track.output_box is None:
                continue
            x1, y1, x2, y2 = track.output_box
            width, height = (x2 - x1 + 1, y2 - y1 + 1)
            lines.append(
                f"{frame_index},{track.track_id},{x1:.6f},{y1:.6f},{width:.6f},{height:.6f},{track.last_score:.6f},-1,-1,-1\n"
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
            x1, y1, x2, y2 = (int(round(value)) for value in track.output_box)
            if track.status == "instrument":
                instrument_ids.append(track.track_id)
                color = palette[(track.track_id - 1) % len(palette)]
                label = f"ID {track.track_id}"
            else:
                color = (180, 180, 180)
                label = f"candidate {track.track_id}"
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


def load_detections(path, frame_count, video_path):

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
    settings = config["anchors"]
    HISTORY_LENGTH = int(settings.get("history_length", 20))
    TRACKING_POINTS = int(settings.get("tracking_points", 10))
    MIN_VALID_POINTS = int(settings.get("min_valid_points", 5))
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    args = argparse.Namespace(**config["tracker"])
    args.seed = RANDOM_SEED
    args.anchors = settings
    args.benchmark_box_source = cli.box_source or config["box_source"]
    args.mask_alpha = 0.35
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

    predictor = build_sam2_video_predictor(
        config["model_cfg"],
        str(cli.checkpoint),
        device=device,
        hydra_overrides_extra=["++model.fill_hole_area=0"],
    )
    predictor.__class__ = StreamingPredictor
    if predictor.spatial_perceiver is not None:
        predictor.spatial_perceiver.__class__ = BatchedPerceiverResampler
    mask_generator = SAM2AutomaticMaskGenerator(model=predictor, **config["mask_generator"])
    runner = STTAMRunner(args, predictor, mask_generator)
    runner.yolo_detections = detections
    writer = None
    processed = 0
    previous_gray = None
    try:
        with tempfile.TemporaryDirectory(prefix="sttam_demo_") as temp, (
            cli.output / "tracks.txt"
        ).open("w", encoding="utf-8") as output, torch.inference_mode(), torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
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
                if runner.initial_classification_done and index > observation:
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
