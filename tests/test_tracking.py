"""Run the bundled video demo (this is an executable smoke test, not pytest)."""

from pathlib import Path
import argparse
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to edgetam.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-frames", type=int, default=150, help="Use 0 for the full clip")
    parser.add_argument("--output", type=Path, default=root / "outputs/demo")
    parser.add_argument(
        "--shared-detections", type=Path, help="Optional frame-aligned YOLO11n cache"
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be zero or positive")
    command = [
        sys.executable,
        "-B",
        str(root / "sttam.py"),
        "--config",
        str(root / "configs/cholectrack20_shared.yaml"),
        "--input",
        str(root / "tests/cholectrack20_demo.mp4"),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--device",
        args.device,
        "--output",
        str(args.output.resolve()),
        "--box-source",
        "yolo11" if args.shared_detections else "mask",
    ]
    if args.max_frames:
        command += ["--max-frames", str(args.max_frames)]
    if args.shared_detections:
        command += ["--detection-cache", str(args.shared_detections.resolve())]
    if args.show:
        command += ["--show"]
    print("Running stTAM demo; shared boxes:", bool(args.shared_detections), flush=True)
    subprocess.run(command, check=True)
    video = args.output / "tracking.mp4"
    tracks = args.output / "tracks.txt"
    if not video.is_file() or video.stat().st_size <= 1024 or not tracks.is_file():
        raise RuntimeError("The demo did not produce its expected outputs")
    print(f"Demo completed: {video.resolve()}")


if __name__ == "__main__":
    main()
