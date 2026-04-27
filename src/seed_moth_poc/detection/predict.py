"""Run YOLO inference on new images, count moths, and save visual previews."""

import argparse
import json
from pathlib import Path

from seed_moth_poc.data_prep.commons import (
    Box,
    ImageBuffer,
    draw_line,
    ensure_directory,
    load_image,
    save_png,
)
from seed_moth_poc.detection.detector import (
    DetectionResult,
    predict_images,
    save_prediction_outputs,
)

BOX_COLOR = (44, 160, 82, 255)
BOX_THICKNESS = 3


def draw_box_outline(
    image: ImageBuffer,
    box: Box,
    color: tuple[int, int, int, int],
) -> None:
    """Draw a rectangular outline around one predicted bounding box."""
    ordered = box.ordered()
    x1 = int(round(ordered.x1))
    y1 = int(round(ordered.y1))
    x2 = int(round(ordered.x2))
    y2 = int(round(ordered.y2))
    draw_line(image, x1, y1, x2, y1, color, thickness=BOX_THICKNESS)
    draw_line(image, x2, y1, x2, y2, color, thickness=BOX_THICKNESS)
    draw_line(image, x2, y2, x1, y2, color, thickness=BOX_THICKNESS)
    draw_line(image, x1, y2, x1, y1, color, thickness=BOX_THICKNESS)


def save_prediction_previews(
    results: list[DetectionResult],
    output_dir: Path,
) -> Path:
    """Render box overlays for each prediction result and save preview images."""
    preview_root = ensure_directory(output_dir / "preview")
    for result in results:
        image = load_image(result.image_path)
        for box in result.boxes:
            draw_box_outline(image, box, BOX_COLOR)
        save_png(image, preview_root / f"{result.image_path.stem}.png")
    return preview_root


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for detector inference."""
    parser = argparse.ArgumentParser(
        description="Run a YOLO moth detector on images or directories.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("results/models/detection/best.pt"),
        help="Path to the trained YOLO weights.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more image files or directories to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/predictions/detection"),
        help="Directory where predicted labels, summaries, and previews will be saved.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold for predictions.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional YOLO device setting, such as cpu, mps, or 0.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="predictions.json",
        help="Name of the JSON summary file written to the output directory.",
    )
    return parser


def main() -> None:
    """Load the trained model, predict boxes, and write count outputs."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Model weights not found: {args.model}. "
            "Run detection/train.py first."
        )

    results = predict_images(
        args.model,
        args.inputs,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    output_dir = ensure_directory(args.output_dir)
    summary_path = save_prediction_outputs(
        results,
        output_dir,
        summary_name=args.summary_name,
    )
    preview_dir = save_prediction_previews(results, output_dir)
    total_count = sum(result.count for result in results)
    summary = [
        {
            "image": str(result.image_path),
            "count": result.count,
        }
        for result in results
    ]
    (output_dir / "counts.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Processed {len(results)} images.")
    print(f"Total predicted moths: {total_count}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved counts: {output_dir / 'counts.json'}")
    print(f"Saved previews: {preview_dir}")


if __name__ == "__main__":
    main()
