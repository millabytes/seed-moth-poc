"""YOLO-based training and inference helpers for seed moth detection."""

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ultralytics import YOLO

from seed_moth_poc.data_prep.commons import (
    Box,
    collect_image_files,
    ensure_directory,
    read_yolo_boxes,
    write_yolo_boxes,
)

DEFAULT_CLASS_NAME = "moth"
DEFAULT_MODEL_WEIGHTS = Path("assets/pretrained/yolo11n.pt")
DEFAULT_PRETRAINED_MODEL_NAME = "yolo11n.pt"
DEFAULT_VAL_RATIO = 0.2
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.7
DEFAULT_IMAGE_SIZE = 640


@dataclass(slots=True)
class DatasetLayout:
    """Prepared YOLO dataset layout for train and validation splits."""

    dataset_root: Path
    dataset_yaml: Path
    split_manifest: Path
    train_images: int
    val_images: int


@dataclass(slots=True)
class TrainingRun:
    """Metadata for one YOLO training run."""

    run_dir: Path
    best_weights: Path
    final_weights: Path | None
    dataset_yaml: Path
    pretrained_weights: str


@dataclass(slots=True)
class DetectionResult:
    """Predictions for one image."""

    image_path: Path
    width: int
    height: int
    boxes: list[Box]
    confidences: list[float]
    class_ids: list[int]

    @property
    def count(self) -> int:
        """Return the number of predicted moths."""
        return len(self.boxes)


def collect_synthetic_pairs(synthetic_root: Path) -> list[tuple[Path, Path]]:
    """Match synthetic images with their YOLO label files."""
    images_root = synthetic_root / "images"
    labels_root = synthetic_root / "labels"
    image_paths = collect_image_files([images_root], recursive=True)
    pairs: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        label_path = labels_root / f"{image_path.stem}.txt"
        pairs.append((image_path, label_path))
    return pairs


def split_pairs(
    pairs: Sequence[tuple[Path, Path]],
    *,
    val_ratio: float,
    seed: int | None,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Split labelled images into train and validation subsets."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in the range [0.0, 1.0).")
    if not pairs:
        raise ValueError("Cannot split an empty dataset.")

    ordered = list(pairs)
    random.Random(seed).shuffle(ordered)

    if val_ratio <= 0.0 or len(ordered) == 1:
        val_count = 0
    else:
        val_count = int(round(len(ordered) * val_ratio))
        val_count = max(1, min(len(ordered) - 1, val_count))

    val_pairs = ordered[:val_count]
    train_pairs = ordered[val_count:]
    return train_pairs, val_pairs


def copy_pair(
    image_path: Path,
    label_path: Path,
    images_root: Path,
    labels_root: Path,
) -> None:
    """Copy one image and its label file into a YOLO split directory."""
    ensure_directory(images_root)
    ensure_directory(labels_root)
    shutil.copy2(image_path, images_root / image_path.name)
    if label_path.exists():
        shutil.copy2(label_path, labels_root / label_path.name)
    else:
        (labels_root / f"{image_path.stem}.txt").write_text("")


def write_dataset_yaml(dataset_root: Path, yaml_path: Path) -> None:
    """Write a YOLO dataset.yaml file for one-class moth detection."""
    content = "\n".join(
        [
            f"path: {dataset_root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "nc: 1",
            "names:",
            f"  0: {DEFAULT_CLASS_NAME}",
            "",
        ]
    )
    yaml_path.write_text(content)


def write_split_manifest(
    manifest_path: Path,
    *,
    train_pairs: Sequence[tuple[Path, Path]],
    val_pairs: Sequence[tuple[Path, Path]],
) -> None:
    """Write a JSON summary of the train/validation split."""
    manifest = {
        "train": [
            {"image": str(image_path), "label": str(label_path)}
            for image_path, label_path in train_pairs
        ],
        "val": [
            {"image": str(image_path), "label": str(label_path)}
            for image_path, label_path in val_pairs
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def prepare_yolo_dataset(
    synthetic_root: Path,
    dataset_root: Path,
    *,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int | None = None,
    overwrite: bool = True,
) -> DatasetLayout:
    """Prepare a YOLO dataset directory from synthetic images and labels."""
    pairs = collect_synthetic_pairs(synthetic_root)
    if not pairs:
        raise ValueError(f"No synthetic images found in {synthetic_root}.")

    train_pairs, val_pairs = split_pairs(pairs, val_ratio=val_ratio, seed=seed)

    if overwrite and dataset_root.exists():
        shutil.rmtree(dataset_root)

    train_images_root = dataset_root / "images" / "train"
    train_labels_root = dataset_root / "labels" / "train"
    val_images_root = dataset_root / "images" / "val"
    val_labels_root = dataset_root / "labels" / "val"
    ensure_directory(train_images_root)
    ensure_directory(train_labels_root)
    ensure_directory(val_images_root)
    ensure_directory(val_labels_root)

    for image_path, label_path in train_pairs:
        copy_pair(image_path, label_path, train_images_root, train_labels_root)
    for image_path, label_path in val_pairs:
        copy_pair(image_path, label_path, val_images_root, val_labels_root)

    dataset_yaml = dataset_root / "dataset.yaml"
    write_dataset_yaml(dataset_root, dataset_yaml)
    split_manifest = dataset_root / "split.json"
    write_split_manifest(split_manifest, train_pairs=train_pairs, val_pairs=val_pairs)

    return DatasetLayout(
        dataset_root=dataset_root,
        dataset_yaml=dataset_yaml,
        split_manifest=split_manifest,
        train_images=len(train_pairs),
        val_images=len(val_pairs),
    )


def resolve_pretrained_weights(weights: str | Path) -> str:
    """Resolve a pretrained checkpoint path or name for YOLO initialization."""
    candidate = Path(weights)
    if candidate.exists():
        return candidate.as_posix()
    if candidate == DEFAULT_MODEL_WEIGHTS:
        return DEFAULT_PRETRAINED_MODEL_NAME
    if candidate.suffix == ".pt" and candidate.parent != Path("."):
        raise FileNotFoundError(
            f"Pretrained weights not found at {candidate}. "
            f"Place the file there, or pass {DEFAULT_PRETRAINED_MODEL_NAME} "
            "to let Ultralytics resolve the checkpoint from its cache or download it."
        )
    return str(weights)


def resolve_best_weights(model: YOLO, project_dir: Path, run_name: str) -> Path:
    """Resolve the best weights path produced by a YOLO training run."""
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        best_attr = getattr(trainer, "best", None)
        if best_attr:
            return Path(best_attr)
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir is not None:
            candidate = Path(save_dir) / "weights" / "best.pt"
            if candidate.exists():
                return candidate
    candidate = project_dir / run_name / "weights" / "best.pt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Could not resolve best.pt from the YOLO training run.")


def train_yolo_detector(
    dataset_yaml: Path,
    output_dir: Path,
    *,
    weights: str | Path = DEFAULT_MODEL_WEIGHTS,
    run_name: str = "train",
    epochs: int = 100,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    batch: int = 8,
    patience: int = 30,
    device: str | None = None,
    seed: int | None = None,
    exist_ok: bool = True,
) -> TrainingRun:
    """Train a YOLO detector and return the produced weights."""
    resolved_weights = resolve_pretrained_weights(weights)
    model = YOLO(resolved_weights)
    train_kwargs: dict[str, object] = {
        "data": str(dataset_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(output_dir),
        "name": run_name,
        "exist_ok": exist_ok,
        "patience": patience,
        "verbose": False,
    }
    if device is not None:
        train_kwargs["device"] = device
    if seed is not None:
        train_kwargs["seed"] = seed

    model.train(**train_kwargs)
    run_dir = output_dir / run_name
    best_weights = resolve_best_weights(model, output_dir, run_name)
    final_weights = run_dir / "weights" / "last.pt"
    if not final_weights.exists():
        final_weights = None

    return TrainingRun(
        run_dir=run_dir,
        best_weights=best_weights,
        final_weights=final_weights,
        dataset_yaml=dataset_yaml,
        pretrained_weights=resolved_weights,
    )


def collect_input_images(inputs: Sequence[Path | str]) -> list[Path]:
    return collect_image_files(inputs, recursive=True)


def predict_images(
    model_path: Path,
    inputs: Sequence[Path | str],
    *,
    conf: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    device: str | None = None,
) -> list[DetectionResult]:
    """Run YOLO inference on images and collect box predictions."""
    model = YOLO(str(model_path))
    image_paths = collect_input_images(inputs)
    results: list[DetectionResult] = []

    for image_path in image_paths:
        predict_kwargs: dict[str, object] = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "verbose": False,
        }
        if device is not None:
            predict_kwargs["device"] = device

        predictions = model.predict(source=str(image_path), **predict_kwargs)
        if not predictions:
            results.append(
                DetectionResult(
                    image_path=image_path,
                    width=0,
                    height=0,
                    boxes=[],
                    confidences=[],
                    class_ids=[],
                )
            )
            continue

        result = predictions[0]
        height, width = result.orig_shape[:2]
        boxes: list[Box] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            for coords, score, class_id in zip(xyxy, confs, classes):
                boxes.append(
                    Box(
                        float(coords[0]),
                        float(coords[1]),
                        float(coords[2]),
                        float(coords[3]),
                    )
                )
                confidences.append(float(score))
                class_ids.append(int(class_id))

        results.append(
            DetectionResult(
                image_path=image_path,
                width=width,
                height=height,
                boxes=boxes,
                confidences=confidences,
                class_ids=class_ids,
            )
        )

    return results


def save_prediction_outputs(
    results: Sequence[DetectionResult],
    output_dir: Path,
    *,
    summary_name: str = "predictions.json",
) -> Path:
    labels_root = ensure_directory(output_dir / "labels")
    summary: list[dict[str, object]] = []

    for result in results:
        label_path = labels_root / f"{result.image_path.stem}.txt"
        write_yolo_boxes(
            label_path,
            result.boxes,
            width=result.width,
            height=result.height,
        )
        summary.append(
            {
                "image": str(result.image_path),
                "file": result.image_path.name,
                "width": result.width,
                "height": result.height,
                "count": result.count,
                "boxes": [
                    [box.x1, box.y1, box.x2, box.y2] for box in result.boxes
                ],
                "confidences": result.confidences,
                "class_ids": result.class_ids,
                "label": str(label_path),
            }
        )

    summary_path = output_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path
