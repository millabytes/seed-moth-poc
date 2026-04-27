"""Train a YOLO detector on synthetic seed moth data."""

import argparse
import json
import shutil
from pathlib import Path

from seed_moth_poc.data_prep.commons import ensure_directory
from seed_moth_poc.detection.detector import (
    prepare_yolo_dataset,
    train_yolo_detector,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a YOLO moth detector from synthetic images and labels.",
    )
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=Path("data/synthetic"),
        help="Root directory containing synthetic/images and synthetic/labels.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("results/detection/yolo_dataset"),
        help="Directory where the YOLO train/val split will be prepared.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/models/detection"),
        help="Directory where the YOLO training run and best.pt will be saved.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("assets/pretrained/yolo11n.pt"),
        help=(
            "Pretrained YOLO weights to fine-tune. "
            "If the local asset exists it is used directly; otherwise the script "
            "falls back to Ultralytics' yolo11n.pt checkpoint name."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Training batch size.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of synthetic samples reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting and training.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="train",
        help="Name of the YOLO training run directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional YOLO device setting, such as cpu, mps, or 0.",
    )
    return parser


def main() -> None:
    """Prepare the dataset split, train YOLO, and copy the best weights."""
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_layout = prepare_yolo_dataset(
        args.synthetic_root,
        args.dataset_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=True,
    )

    models_root = ensure_directory(args.output_dir)
    run = train_yolo_detector(
        dataset_layout.dataset_yaml,
        models_root,
        weights=args.weights,
        run_name=args.run_name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        exist_ok=True,
    )

    stable_best = models_root / "best.pt"
    shutil.copy2(run.best_weights, stable_best)

    training_summary = {
        "dataset_root": str(dataset_layout.dataset_root),
        "dataset_yaml": str(dataset_layout.dataset_yaml),
        "split_manifest": str(dataset_layout.split_manifest),
        "requested_weights": str(args.weights),
        "pretrained_weights": run.pretrained_weights,
        "train_images": dataset_layout.train_images,
        "val_images": dataset_layout.val_images,
        "run_dir": str(run.run_dir),
        "best_weights": str(run.best_weights),
        "stable_best_weights": str(stable_best),
    }
    (models_root / "training.json").write_text(
        json.dumps(training_summary, indent=2) + "\n"
    )

    print(
        f"Prepared dataset: {dataset_layout.train_images} train / "
        f"{dataset_layout.val_images} val images."
    )
    print(f"YOLO best weights: {run.best_weights}")
    print(f"Stable copy: {stable_best}")


if __name__ == "__main__":
    main()
