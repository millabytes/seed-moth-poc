#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

SKIP_SYNC="${SKIP_SYNC:-0}"
FORCE="${FORCE:-0}"
STEP_INDEX=0
STEP_TOTAL=9

# Tunable defaults
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PATIENCE="${PATIENCE:-15}"
IMG_SIZE="${IMG_SIZE:-640}"
VAL_RATIO="${VAL_RATIO:-0.2}"
BACKGROUND_COUNT="${BACKGROUND_COUNT:-24}"
SYNTHETIC_COUNT="${SYNTHETIC_COUNT:-120}"
SEED="${SEED:-42}"
PREDICT_CONF="${PREDICT_CONF:-0.25}"
PREDICT_IOU="${PREDICT_IOU:-0.7}"
EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.5}"

TARGET_IMAGES_DIR="data/reference/target/images"
TARGET_LABELS_DIR="data/reference/target/labels"
MORPHOLOGY_DIR="data/reference/target/morphology"
DERIVED_ROOT="data/reference/derived"
BACKGROUND_ROOT="data/backgrounds/generated"
SYNTHETIC_ROOT="data/synthetic"
TEST_IMAGES_DIR="data/test_images"

DATASET_ROOT="results/detection/yolo_dataset"
MODELS_ROOT="results/models/detection"
PREDICTIONS_ROOT="results/predictions/detection"
EVAL_VAL_ROOT="results/eval/val"
EVAL_TEST_ROOT="results/eval/test"
TEST_LABELS_ROOT="results/test_images/labels"
PRETRAINED_WEIGHTS="assets/pretrained/yolo11n.pt"

log() {
  printf '\n[%s] %s\n' "$1" "$2"
}

step_banner() {
  STEP_INDEX=$((STEP_INDEX + 1))
  printf '\n[%s/%s] %s\n' "$STEP_INDEX" "$STEP_TOTAL" "$1"
}

count_images() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi
  find "$dir" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
    -print | wc -l | tr -d '[:space:]'
}

count_txt_files() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi
  find "$dir" -type f -iname '*.txt' -print | wc -l | tr -d '[:space:]'
}

has_files() {
  local dir="$1"
  [[ -d "$dir" ]] && find "$dir" -type f -print -quit >/dev/null
}

annotation_complete() {
  local image_count label_count
  image_count="$(count_images "$TARGET_IMAGES_DIR")"
  label_count="$(count_txt_files "$TARGET_LABELS_DIR")"
  [[ "$image_count" -gt 0 && "$label_count" -ge "$image_count" ]]
}

mask_extraction_complete() {
  local cutout_count mask_count
  cutout_count="$(count_images "$DERIVED_ROOT/cutouts/images")"
  mask_count="$(count_images "$DERIVED_ROOT/masks/images")"
  [[ -f "$DERIVED_ROOT/manifest.json" && "$cutout_count" -gt 0 && "$cutout_count" -eq "$mask_count" ]]
}

backgrounds_complete() {
  local background_count
  background_count="$(count_images "$BACKGROUND_ROOT")"
  [[ -f "$BACKGROUND_ROOT/manifest.json" && "$background_count" -gt 0 ]]
}

synthetic_complete() {
  local image_count label_count
  image_count="$(count_images "$SYNTHETIC_ROOT/images")"
  label_count="$(count_txt_files "$SYNTHETIC_ROOT/labels")"
  [[ -f "$SYNTHETIC_ROOT/manifest.json" && "$image_count" -gt 0 && "$image_count" -eq "$label_count" ]]
}

dataset_complete() {
  [[ -f "$DATASET_ROOT/dataset.yaml" && -f "$DATASET_ROOT/split.json" ]]
}

training_complete() {
  [[ -f "$MODELS_ROOT/best.pt" && -f "$MODELS_ROOT/training.json" ]]
}

evaluation_val_complete() {
  [[ -f "$EVAL_VAL_ROOT/metrics.json" && -f "$EVAL_VAL_ROOT/per_image.json" ]]
}

prediction_complete() {
  [[ -f "$PREDICTIONS_ROOT/predictions.json" && -f "$PREDICTIONS_ROOT/counts.json" ]]
}

real_evaluation_complete() {
  [[ -f "$EVAL_TEST_ROOT/metrics.json" && -f "$EVAL_TEST_ROOT/per_image.json" ]]
}

auto_detect_device() {
  uv run python - <<'PY'
import platform

try:
    import torch
except Exception:
    print("cpu")
    raise SystemExit(0)

if torch.cuda.is_available():
    print("0")
elif platform.system() == "Darwin" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
}

resolve_device() {
  if [[ "$DEVICE" == "auto" ]]; then
    auto_detect_device
  else
    printf '%s\n' "$DEVICE"
  fi
}

run_step() {
  local label="$1"
  shift
  step_banner "$label"
  local start_ts
  start_ts="$(date +%s)"
  "$@"
  local end_ts
  end_ts="$(date +%s)"
  local elapsed
  elapsed="$((end_ts - start_ts))"
  log "done" "$label (${elapsed}s)"
}

skip_step() {
  local label="$1"
  step_banner "$label"
  log "skip" "$label"
}

require_dir() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "[error] Missing required directory: $dir" >&2
    exit 1
  fi
}

require_dir "$TARGET_IMAGES_DIR"
require_dir "$MORPHOLOGY_DIR"
require_dir "$TEST_IMAGES_DIR"

if [[ "$SKIP_SYNC" != "1" ]]; then
  run_step "Sync uv environment" uv sync --all-extras --locked
else
  skip_step "Sync uv environment (SKIP_SYNC=1)"
fi

RESOLVED_DEVICE="$(resolve_device)"
log "info" "Training/inference device resolved to: $RESOLVED_DEVICE"
log "info" "Config: epochs=$EPOCHS batch=$BATCH_SIZE patience=$PATIENCE img_size=$IMG_SIZE val_ratio=$VAL_RATIO background_count=$BACKGROUND_COUNT synthetic_count=$SYNTHETIC_COUNT seed=$SEED conf=$PREDICT_CONF iou=$PREDICT_IOU eval_iou=$EVAL_IOU_THRESHOLD"

if [[ "$FORCE" != "1" && annotation_complete ]]; then
  skip_step "Manual annotation is already complete"
else
  run_step "Manual annotation" \
    uv run seed-moth-annotate \
      --images "$TARGET_IMAGES_DIR" \
      --labels "$TARGET_LABELS_DIR"
fi

if [[ "$FORCE" != "1" && mask_extraction_complete ]]; then
  skip_step "Mask/cutout extraction is already complete"
else
  run_step "Extract masks and cutouts" \
    uv run seed-moth-extract \
      --inputs "$TARGET_IMAGES_DIR" "$MORPHOLOGY_DIR" \
      --output-root "$DERIVED_ROOT" \
      --labels-root "$TARGET_LABELS_DIR"
fi

if [[ "$FORCE" != "1" && backgrounds_complete ]]; then
  skip_step "Background generation is already complete"
else
  run_step "Generate trap-like backgrounds" \
    uv run seed-moth-backgrounds \
      --output-dir "$BACKGROUND_ROOT" \
      --count "$BACKGROUND_COUNT" \
      --seed "$SEED"
fi

if [[ "$FORCE" != "1" && synthetic_complete ]]; then
  skip_step "Synthetic data generation is already complete"
else
  run_step "Generate synthetic dataset" \
    uv run seed-moth-synthetic \
      --backgrounds "$BACKGROUND_ROOT" \
      --sources-root "$DERIVED_ROOT/cutouts/images" \
      --output-root "$SYNTHETIC_ROOT" \
      --count "$SYNTHETIC_COUNT" \
      --seed "$SEED"
fi

if [[ -f "$PRETRAINED_WEIGHTS" ]]; then
  log "info" "Using local pretrained checkpoint: $PRETRAINED_WEIGHTS"
else
  log "warn" "Local pretrained checkpoint not found at $PRETRAINED_WEIGHTS; Ultralytics will fall back to yolo11n.pt resolution."
fi

if [[ "$FORCE" != "1" && dataset_complete && training_complete ]]; then
  skip_step "Detection training is already complete"
else
  run_step "Train YOLO detector" \
    uv run seed-moth-train \
      --synthetic-root "$SYNTHETIC_ROOT" \
      --dataset-root "$DATASET_ROOT" \
      --output-dir "$MODELS_ROOT" \
      --weights "$PRETRAINED_WEIGHTS" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --batch "$BATCH_SIZE" \
      --imgsz "$IMG_SIZE" \
      --val-ratio "$VAL_RATIO" \
      --seed "$SEED" \
      --device "$RESOLVED_DEVICE"
fi

if [[ "$FORCE" != "1" && evaluation_val_complete ]]; then
  skip_step "Synthetic validation evaluation is already complete"
else
  run_step "Evaluate on synthetic validation split" \
    uv run seed-moth-evaluate \
      --model "$MODELS_ROOT/best.pt" \
      --iou-threshold "$EVAL_IOU_THRESHOLD" \
      --imgsz "$IMG_SIZE" \
      --device "$RESOLVED_DEVICE"
fi

if [[ "$FORCE" != "1" && prediction_complete ]]; then
  skip_step "Inference on test images is already complete"
else
  run_step "Predict and count on unlabeled test images" \
    uv run seed-moth-predict \
      --model "$MODELS_ROOT/best.pt" \
      --inputs "$TEST_IMAGES_DIR" \
      --output-dir "$PREDICTIONS_ROOT" \
      --conf "$PREDICT_CONF" \
      --iou "$PREDICT_IOU" \
      --imgsz "$IMG_SIZE" \
      --device "$RESOLVED_DEVICE"
fi

if [[ -d "$TEST_LABELS_ROOT" && "$(count_txt_files "$TEST_LABELS_ROOT")" -gt 0 ]]; then
  if [[ "$FORCE" != "1" && real_evaluation_complete ]]; then
    skip_step "Real-image evaluation is already complete"
  else
    run_step "Evaluate on labeled real test images" \
      uv run seed-moth-evaluate \
        --model "$MODELS_ROOT/best.pt" \
        --inputs "$TEST_IMAGES_DIR" \
        --labels-root "$TEST_LABELS_ROOT" \
        --output-dir "$EVAL_TEST_ROOT" \
        --iou-threshold "$EVAL_IOU_THRESHOLD" \
        --imgsz "$IMG_SIZE" \
        --device "$RESOLVED_DEVICE"
  fi
else
  skip_step "Real-image evaluation (no labels found in $TEST_LABELS_ROOT)"
fi

log "done" "Pipeline complete"
