"""Evaluate YOLO detection and counting quality on labeled images."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from seed_moth_poc.data_prep.commons import (
    Box,
    collect_image_files,
    ensure_directory,
    load_image,
    read_yolo_boxes,
)
from seed_moth_poc.detection.detector import predict_images

DEFAULT_MODEL = Path("results/models/detection/best.pt")
DEFAULT_INPUTS = [Path("results/detection/yolo_dataset/images/val")]
DEFAULT_LABELS_ROOT = Path("results/detection/yolo_dataset/labels/val")
DEFAULT_OUTPUT_DIR = Path("results/eval/val")


@dataclass(slots=True)
class MatchCounts:
    """Aggregate detection matches for one evaluation set."""

    true_positives: int
    false_positives: int
    false_negatives: int
    matched_ious: list[float]

    @property
    def precision(self) -> float:
        """Return detection precision."""
        denominator = self.true_positives + self.false_positives
        if denominator <= 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def recall(self) -> float:
        """Return detection recall."""
        denominator = self.true_positives + self.false_negatives
        if denominator <= 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def f1(self) -> float:
        """Return detection F1 score."""
        precision = self.precision
        recall = self.recall
        denominator = precision + recall
        if denominator <= 0:
            return 0.0
        return 2.0 * precision * recall / denominator

    @property
    def mean_iou(self) -> float:
        """Return the mean IoU of matched boxes."""
        if not self.matched_ious:
            return 0.0
        return sum(self.matched_ious) / len(self.matched_ious)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO moth detections against YOLO labels.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the trained YOLO weights.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUTS,
        help="One or more image files or directories to evaluate.",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=DEFAULT_LABELS_ROOT,
        help="Root directory with ground-truth YOLO label files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where metrics and per-image results will be saved.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used to match predictions to ground truth.",
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
    return parser


def box_iou(left: Box, right: Box) -> float:
    """Compute IoU for two pixel-space boxes."""
    left_box = left.ordered()
    right_box = right.ordered()
    x1 = max(left_box.x1, right_box.x1)
    y1 = max(left_box.y1, right_box.y1)
    x2 = min(left_box.x2, right_box.x2)
    y2 = min(left_box.y2, right_box.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_left = left_box.width() * left_box.height()
    area_right = right_box.width() * right_box.height()
    union = area_left + area_right - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def greedy_match(
    predicted: list[Box],
    ground_truth: list[Box],
    *,
    iou_threshold: float,
) -> MatchCounts:
    """Match predicted boxes to ground-truth boxes with a greedy IoU rule."""
    matched_ious: list[float] = []
    used_truth: set[int] = set()
    true_positives = 0

    scored_pairs: list[tuple[float, int, int]] = []
    for pred_index, predicted_box in enumerate(predicted):
        for truth_index, truth_box in enumerate(ground_truth):
            scored_pairs.append((box_iou(predicted_box, truth_box), pred_index, truth_index))
    scored_pairs.sort(reverse=True, key=lambda item: item[0])

    used_predicted: set[int] = set()
    for iou, pred_index, truth_index in scored_pairs:
        if iou < iou_threshold:
            break
        if pred_index in used_predicted or truth_index in used_truth:
            continue
        used_predicted.add(pred_index)
        used_truth.add(truth_index)
        true_positives += 1
        matched_ious.append(iou)

    false_positives = len(predicted) - true_positives
    false_negatives = len(ground_truth) - true_positives
    return MatchCounts(true_positives, false_positives, false_negatives, matched_ious)


def mean(values: list[float]) -> float:
    """Return the arithmetic mean for a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def evaluate_predictions(
    model_path: Path,
    inputs: list[Path],
    labels_root: Path,
    *,
    iou_threshold: float,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
) -> dict[str, object]:
    """Run inference and compare predictions against ground truth labels."""
    image_paths = collect_image_files(inputs, recursive=True)
    if not image_paths:
        raise ValueError("No input images found for evaluation.")

    results = predict_images(
        model_path,
        image_paths,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
    )

    per_image: list[dict[str, object]] = []
    tp = fp = fn = 0
    count_errors: list[float] = []
    count_absolute_errors: list[float] = []
    count_squared_errors: list[float] = []
    gt_counts: list[int] = []
    pred_counts: list[int] = []
    matched_ious: list[float] = []

    for result in results:
        image = load_image(result.image_path)
        label_path = labels_root / f"{result.image_path.stem}.txt"
        ground_truth = read_yolo_boxes(label_path, width=image.width, height=image.height)

        counts = greedy_match(result.boxes, ground_truth, iou_threshold=iou_threshold)
        tp += counts.true_positives
        fp += counts.false_positives
        fn += counts.false_negatives
        matched_ious.extend(counts.matched_ious)

        count_error = result.count - len(ground_truth)
        count_errors.append(float(count_error))
        count_absolute_errors.append(abs(float(count_error)))
        count_squared_errors.append(float(count_error * count_error))
        gt_counts.append(len(ground_truth))
        pred_counts.append(result.count)

        per_image.append(
            {
                "image": str(result.image_path),
                "file": result.image_path.name,
                "ground_truth_count": len(ground_truth),
                "predicted_count": result.count,
                "count_error": count_error,
                "count_abs_error": abs(count_error),
                "count_sq_error": count_error * count_error,
                "tp": counts.true_positives,
                "fp": counts.false_positives,
                "fn": counts.false_negatives,
                "precision": counts.precision,
                "recall": counts.recall,
                "f1": counts.f1,
                "mean_iou": counts.mean_iou,
            }
        )

    metrics = MatchCounts(tp, fp, fn, matched_ious)
    count_mae = mean(count_absolute_errors)
    count_mse = mean(count_squared_errors)
    count_rmse = math.sqrt(count_mse)
    count_bias = mean(count_errors)

    return {
        "model": str(model_path),
        "iou_threshold": iou_threshold,
        "num_images": len(results),
        "num_ground_truth_boxes": sum(gt_counts),
        "num_predicted_boxes": sum(pred_counts),
        "detection": {
            "true_positives": metrics.true_positives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "mean_iou": metrics.mean_iou,
        },
        "counting": {
            "bias": count_bias,
            "mae": count_mae,
            "mse": count_mse,
            "rmse": count_rmse,
            "min_gt_count": min(gt_counts) if gt_counts else 0,
            "max_gt_count": max(gt_counts) if gt_counts else 0,
            "min_pred_count": min(pred_counts) if pred_counts else 0,
            "max_pred_count": max(pred_counts) if pred_counts else 0,
        },
        "per_image": per_image,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model weights not found: {args.model}. Run detection/train.py first.")
    if not args.labels_root.exists():
        raise SystemExit(f"Labels root not found: {args.labels_root}.")

    output_dir = ensure_directory(args.output_dir)
    report = evaluate_predictions(
        args.model,
        args.inputs,
        args.labels_root,
        iou_threshold=args.iou_threshold,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    metrics_path = output_dir / "metrics.json"
    per_image_path = output_dir / "per_image.json"
    metrics_path.write_text(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=2) + "\n")
    per_image_path.write_text(json.dumps(report["per_image"], indent=2) + "\n")

    detection = report["detection"]
    counting = report["counting"]

    print(f"Images: {report['num_images']}")
    print(f"Detection precision: {detection['precision']:.4f}")
    print(f"Detection recall: {detection['recall']:.4f}")
    print(f"Detection F1: {detection['f1']:.4f}")
    print(f"Mean IoU: {detection['mean_iou']:.4f}")
    print(f"Count bias: {counting['bias']:.4f}")
    print(f"Count MAE: {counting['mae']:.4f}")
    print(f"Count MSE: {counting['mse']:.4f}")
    print(f"Count RMSE: {counting['rmse']:.4f}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved per-image results: {per_image_path}")


if __name__ == "__main__":
    main()
