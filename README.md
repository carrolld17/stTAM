# stTAM

**Training-Free Surgical Instrument Tracking Based on TAM and Motion-Trajectory Classification**

Gang He, Ning Dai, Xingyu Yang, Hui Chen, Tengfei Chen, Yongbing Chen

Gang He and Ning Dai contributed equally.

![stTAM framework](framework.png)

stTAM uses frozen EdgeTAM, automatic candidate discovery, motion filtering, and mask-guided anchor maintenance. Motion features combine mean displacement and circular variance of `atan2` directions, with per-feature min-max normalization before K-means.

## Installation

Use Python 3.10+ with a compatible [PyTorch](https://pytorch.org/get-started/locally/) installation.

```bash
python -m pip install "git+https://github.com/facebookresearch/EdgeTAM.git"
python -m pip install opencv-python numpy scipy scikit-learn scikit-image PyYAML timm
```

Download `edgetam.pt` from [EdgeTAM](https://github.com/facebookresearch/EdgeTAM). EdgeTAM and SAM 2 share the `sam2` package name; use the EdgeTAM installation.

## VID01 demo

The bundled `tests/cholectrack20_demo.mp4` is a CholecTrack20 VID01 clip.

```bash
python tests/test_tracking.py --checkpoint /path/to/edgetam.pt
```

The default run processes 150 frames in mask-only mode. The first 51 frames form the observation window; tracking rows start at local frame 52. Outputs are `outputs/demo/tracking.mp4` and `outputs/demo/tracks.txt`. Add `--max-frames 0` for the full clip or `--show` for a preview.

## Shared YOLO11n detections

```bash
python tests/test_tracking.py --checkpoint /path/to/edgetam.pt --shared-detections /path/to/detections.json
```

Supply one detection list per local input frame, starting at frame zero. Each detection is `[x1, y1, x2, y2, confidence, class]`, with exclusive upper coordinates:

```json
{"detections": [[[100, 80, 200, 220, 0.95, 0]], []]}
```

Shared mode assigns identities to matched detector boxes and preserves their coordinates and confidence. Unmatched tracks produce no box; mask boxes never fill missing detections. Prepare the cache using YOLO11n with confidence 0.1 and input size 512. The detector is separate from frozen stTAM.

## EndoVis2017

```bash
python sttam.py --config configs/endovis2017.yaml --input /path/to/frames --checkpoint /path/to/edgetam.pt --output outputs/endovis
```

Both presets use the same motion classifier and discover candidates only in the first frame. This release contains inference code; benchmark annotations, detector weights, and evaluation scripts are not included.

## License and acknowledgements

Code: Apache-2.0. The streaming and batched-memory adapters derive from EdgeTAM/SAM 2, copyright Meta Platforms, Inc. and affiliates. Dataset and model rights remain with their respective authors.

Supported by NSFC (12572364, 12272033) and Fujiang Laboratory (2023ZYDF074).
