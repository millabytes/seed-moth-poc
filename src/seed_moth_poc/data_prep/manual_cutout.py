"""Browser-based manual cutout reviewer for extracted moth sprites.

This tool is meant for the small number of cutouts that are still poor after
automatic extraction. It opens the extracted cutout PNGs one by one, lets the
user trace a polygon around the moth, and writes a reviewed mask/cutout pair
to a separate output root.
"""

import argparse
import html
import json
import mimetypes
import math
import threading
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from seed_moth_poc.data_prep.commons import (
    ImageBuffer,
    collect_image_files,
    crop_image,
    ensure_directory,
    load_image,
    save_png,
)
from seed_moth_poc.data_prep.mask_extractor import (
    bbox_from_mask,
    cutout_from_mask,
    mask_from_alpha,
    mask_to_image,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 0
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_INPUTS = [Path("data/reference/derived/cutouts/images")]
DEFAULT_OUTPUT_ROOT = Path("data/reference/reviewed")


@dataclass(slots=True)
class ReviewItem:
    """Metadata for one cutout review item."""

    input_path: Path
    output_cutout_path: Path
    output_mask_path: Path
    width: int
    height: int
    display_name: str
    current_path: Path
    already_reviewed: bool = False
    points: list[tuple[float, float]] = field(default_factory=list)
    mode: str = "replace"


class ReviewSession:
    """Mutable review session state shared by the HTTP handler."""

    def __init__(self, items: list[ReviewItem], output_root: Path) -> None:
        """Create a session from a list of review items."""
        self.items = items
        self.output_root = output_root
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._manifest: list[dict[str, Any]] = []

    def item_count(self) -> int:
        """Return the number of items in the session."""
        return len(self.items)

    def get_item(self, index: int) -> ReviewItem:
        """Return one item or raise `IndexError` if the index is invalid."""
        return self.items[index]

    def build_state(self, index: int) -> dict[str, Any]:
        """Build a JSON-serializable state payload for one item."""
        item = self.get_item(index)
        return {
            "index": index,
            "count": self.item_count(),
            "display_name": item.display_name,
            "image_url": f"/api/image?index={index}",
            "input_path": str(item.input_path),
            "output_cutout_path": str(item.output_cutout_path),
            "output_mask_path": str(item.output_mask_path),
            "width": item.width,
            "height": item.height,
            "mode": item.mode,
            "points": [[x, y] for x, y in item.points],
        }

    def _write_manifest(self) -> None:
        """Persist the current review manifest to disk."""
        manifest_path = self.output_root / "manifest.json"
        ensure_directory(manifest_path.parent)
        manifest_path.write_text(json.dumps(self._manifest, indent=2) + "\n")

    def save_review(
        self,
        index: int,
        points: list[tuple[float, float]],
        *,
        mode: str,
    ) -> None:
        """Save one reviewed cutout and its mask."""
        with self._lock:
            item = self.get_item(index)
            source_image = load_image(item.current_path)
            base_mask = mask_from_alpha(source_image)
            has_manual_points = len(points) >= 3
            if has_manual_points:
                manual_mask = polygon_mask_from_points(points, source_image.width, source_image.height)
                if mode == "replace":
                    final_mask = manual_mask
                else:
                    final_mask = union_masks(base_mask, manual_mask)
            else:
                final_mask = base_mask

            final_bbox = bbox_from_mask(final_mask, source_image.width, source_image.height)
            keep_existing_review = item.already_reviewed and not has_manual_points
            reviewed = has_manual_points or item.already_reviewed

            if reviewed and not keep_existing_review:
                mask_image = mask_to_image(final_mask, source_image.width, source_image.height)
                cutout_image = cutout_from_mask(source_image, final_mask)
                if final_bbox != (0, 0, 0, 0):
                    x1, y1, x2, y2 = final_bbox
                    mask_image = crop_image(
                        mask_image,
                        x1,
                        y1,
                        x2 + 1,
                        y2 + 1,
                        fill=(0, 0, 0, 255),
                    )
                    cutout_image = crop_image(
                        cutout_image,
                        x1,
                        y1,
                        x2 + 1,
                        y2 + 1,
                        fill=(0, 0, 0, 0),
                    )

                ensure_directory(item.output_mask_path.parent)
                ensure_directory(item.output_cutout_path.parent)
                save_png(mask_image, item.output_mask_path)
                save_png(cutout_image, item.output_cutout_path)
                item.current_path = item.output_cutout_path
            else:
                if not item.already_reviewed:
                    item.output_mask_path.unlink(missing_ok=True)
                    item.output_cutout_path.unlink(missing_ok=True)
                    item.current_path = item.input_path
                else:
                    item.current_path = item.output_cutout_path

            item.points = []
            item.mode = mode

            if index < len(self._manifest):
                self._manifest[index] = self._build_manifest_record(
                    item,
                    points,
                    mode,
                    final_bbox,
                    reviewed=reviewed,
                )
            else:
                self._manifest.append(
                    self._build_manifest_record(
                        item,
                        points,
                        mode,
                        final_bbox,
                        reviewed=reviewed,
                    )
                )

            self._write_manifest()

    def _build_manifest_record(
        self,
        item: ReviewItem,
        points: list[tuple[float, float]],
        mode: str,
        final_bbox: tuple[int, int, int, int],
        *,
        reviewed: bool,
    ) -> dict[str, Any]:
        """Build one manifest record for a reviewed item."""
        return {
            "input_path": str(item.input_path),
            "current_path": str(item.current_path),
            "output_cutout": str(item.output_cutout_path),
            "output_mask": str(item.output_mask_path),
            "display_name": item.display_name,
            "width": item.width,
            "height": item.height,
            "mode": mode,
            "reviewed": reviewed,
            "points": [[x, y] for x, y in points],
            "bbox": list(final_bbox),
        }

    def request_shutdown(self) -> None:
        """Signal that the server should stop."""
        self.stop_event.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review and manually refine extracted moth cutouts.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="One or more extracted cutout files or directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where reviewed cutouts and masks will be written.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host interface to bind to. Defaults to {DEFAULT_HOST}.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to bind to. Defaults to 0, which asks the OS for a free port.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Collect images recursively from directory inputs.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the server without opening a browser tab.",
    )
    parser.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Also include cutouts that already have a reviewed override.",
    )
    parser.add_argument(
        "--prune-identical",
        action="store_true",
        help=(
            "Remove reviewed cutouts and masks that are byte-identical to the original input cutouts. "
            "Useful when a previous review session saved unchanged copies."
        ),
    )
    return parser.parse_args()


def _collect_items(
    inputs: list[str],
    output_root: Path,
    *,
    recursive: bool,
    include_reviewed: bool,
) -> list[ReviewItem]:
    """Collect review items from extracted cutout inputs."""
    items: list[ReviewItem] = []
    output_root = ensure_directory(output_root)

    for raw_input in inputs:
        source = Path(raw_input)
        if source.is_file():
            if source.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            roots = [(source.parent, [source])]
        elif source.is_dir():
            roots = [(source, collect_image_files([source], recursive=recursive))]
        else:
            continue

        for root, files in roots:
            for input_path in files:
                try:
                    image = load_image(input_path)
                except Exception as exc:  # pragma: no cover - defensive guard
                    print(f"[cutout] Skipping {input_path}: failed to load ({exc}).")
                    continue

                relative_path = input_path.name
                try:
                    relative_path = input_path.relative_to(root).as_posix()
                except ValueError:
                    relative_path = input_path.name

                relative = Path(relative_path).with_suffix(".png")
                output_cutout_path = output_root / "cutouts" / "images" / relative
                output_mask_path = output_root / "masks" / "images" / relative
                already_reviewed = output_cutout_path.exists() and output_mask_path.exists()
                if already_reviewed and not include_reviewed:
                    continue
                current_path = output_cutout_path if output_cutout_path.exists() else input_path
                items.append(
                    ReviewItem(
                        input_path=input_path,
                        output_cutout_path=output_cutout_path,
                        output_mask_path=output_mask_path,
                        width=image.width,
                        height=image.height,
                        display_name=relative_path,
                        current_path=current_path,
                        already_reviewed=already_reviewed,
                    )
                )

    items.sort(key=lambda item: item.display_name.lower())
    return items


def _prune_identical_outputs(output_root: Path) -> int:
    """Remove reviewed outputs that are identical to their original cutouts."""
    manifest_path = output_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest file: {manifest_path}")

    try:
        records = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse {manifest_path}: {exc}") from exc

    if not isinstance(records, list):
        raise SystemExit(f"Unexpected manifest format in {manifest_path}.")

    kept: list[dict[str, Any]] = []
    pruned = 0
    for record in records:
        if not isinstance(record, dict):
            continue

        input_path = Path(str(record.get("input_path", "")))
        output_cutout = Path(str(record.get("output_cutout", "")))
        output_mask = Path(str(record.get("output_mask", "")))

        if not input_path.exists() or not output_cutout.exists():
            kept.append(record)
            continue

        if input_path.read_bytes() == output_cutout.read_bytes():
            output_cutout.unlink(missing_ok=True)
            output_mask.unlink(missing_ok=True)
            pruned += 1
            continue

        record["reviewed"] = True
        kept.append(record)

    manifest_path.write_text(json.dumps(kept, indent=2) + "\n")
    return pruned


def _load_html_template() -> str:
    """Return the HTML document used by the review UI."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Seed moth cutout reviewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3efe6;
      --panel: rgba(255, 252, 245, 0.96);
      --panel-border: rgba(89, 70, 30, 0.16);
      --ink: #2b2215;
      --muted: #6e6250;
      --accent: #8e5f14;
      --accent-soft: rgba(142, 95, 20, 0.12);
      --warn: #a05428;
      --ok: #2f6a4f;
      --shadow: 0 18px 48px rgba(78, 56, 18, 0.16);
      --add: rgba(52, 128, 89, 0.86);
      --replace: rgba(170, 92, 19, 0.92);
    }

    * { box-sizing: border-box; }

    html, body {
      height: 100%;
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 255, 255, 0.68), transparent 36%),
        linear-gradient(180deg, #f8f3ea, #eee1cf 55%, #e6d7c0);
    }

    body {
      display: flex;
      flex-direction: column;
      gap: 14px;
      padding: 16px;
    }

    .shell {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 14px;
      height: 100%;
      min-height: 0;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: center;
      padding: 14px 16px;
    }

    .brand {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 0.01em;
      margin-right: 8px;
    }

    .status {
      flex: 1 1 320px;
      min-width: 220px;
      color: var(--muted);
      line-height: 1.35;
    }

    .status[data-kind="warn"] { color: var(--warn); }
    .status[data-kind="ok"] { color: var(--ok); }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    button {
      appearance: none;
      border: 1px solid rgba(121, 88, 30, 0.24);
      background: linear-gradient(180deg, #fffaf1, #f3e6d1);
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
      transition: transform 0.12s ease, box-shadow 0.12s ease;
    }

    button:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 18px rgba(120, 89, 31, 0.12);
    }

    button:active {
      transform: translateY(0);
      box-shadow: none;
    }

    .stage {
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(240px, 0.7fr);
      gap: 12px;
      align-items: start;
      min-height: 0;
      padding: 14px;
    }

    .canvas-wrap,
    .preview-wrap {
      display: flex;
      justify-content: center;
      align-items: flex-start;
      overflow: auto;
      min-height: 0;
    }

    canvas,
    .preview {
      display: block;
      border-radius: 12px;
      border: 1px solid rgba(91, 66, 21, 0.25);
      box-shadow: 0 10px 34px rgba(75, 51, 13, 0.16);
    }

    canvas {
      background: #fffdf7;
      cursor: crosshair;
    }

    .preview {
      width: 100%;
      max-width: 360px;
      min-height: 180px;
      padding: 14px;
      background:
        linear-gradient(45deg, #ece4d4 25%, transparent 25%),
        linear-gradient(-45deg, #ece4d4 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #ece4d4 75%),
        linear-gradient(-45deg, transparent 75%, #ece4d4 75%);
      background-size: 24px 24px;
      background-position: 0 0, 0 12px, 12px -12px, -12px 0px;
    }

    .preview img {
      display: block;
      width: 100%;
      height: auto;
      object-fit: contain;
    }

    .footer {
      padding: 0 4px 2px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }

    .footer code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      background: rgba(255, 255, 255, 0.56);
      padding: 0 5px;
      border-radius: 6px;
      border: 1px solid rgba(101, 82, 37, 0.12);
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="panel toolbar">
      <div class="brand">Seed moth cutout reviewer</div>
      <div class="status" id="status" data-kind="info">Loading session...</div>
      <div class="controls">
        <button id="prev-btn" type="button">Prev</button>
        <button id="next-btn" type="button">Next</button>
        <button id="save-btn" type="button">Save</button>
        <button id="undo-btn" type="button">Undo</button>
        <button id="clear-btn" type="button">Clear</button>
        <button id="mode-btn" type="button">Mode: Add</button>
        <button id="quit-btn" type="button">Quit</button>
      </div>
    </div>

    <div class="panel stage">
      <div>
        <div class="footer" id="info">Click on the cutout to add polygon points. The manual polygon can be
          merged with the existing alpha mask or used as a replacement. Shortcuts:
          <code>n</code> or <code>Enter</code> next/save, <code>p</code> prev,
          <code>s</code> save, <code>u</code> or <code>Backspace</code> undo,
          <code>r</code> clear, <code>m</code> toggle mode, <code>q</code> quit.
        </div>
        <div class="canvas-wrap">
          <canvas id="canvas"></canvas>
        </div>
      </div>

      <div class="preview-wrap">
        <div class="preview">
          <img id="preview-img" alt="Reviewed cutout preview">
        </div>
      </div>
    </div>
  </div>

  <script id="seed-moth-config" type="application/json">__CONFIG_JSON__</script>
  <script>
    const CONFIG = JSON.parse(document.getElementById("seed-moth-config").textContent);
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const infoEl = document.getElementById("info");
    const previewImg = document.getElementById("preview-img");
    const image = new Image();

    const state = {
      index: 0,
      count: CONFIG.count,
      width: 1,
      height: 1,
      displayName: "",
      points: [],
      mode: "add",
      loadToken: 0,
    };

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function setStatus(message, kind = "info") {
      statusEl.textContent = message;
      statusEl.dataset.kind = kind;
    }

    function updateInfo() {
      infoEl.innerHTML = [
        `Cutout <strong>${state.index + 1}/${state.count}</strong>:`,
        `<code>${state.displayName}</code>`,
        `&middot; points: <strong>${state.points.length}</strong>`,
        `&middot; mode: <strong>${state.mode}</strong>`,
      ].join(" ");
    }

    function imageToCanvasX(x) {
      return (x / state.width) * canvas.width;
    }

    function imageToCanvasY(y) {
      return (y / state.height) * canvas.height;
    }

    function pointerToImage(event) {
      const rect = canvas.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * state.width;
      const y = ((event.clientY - rect.top) / rect.height) * state.height;
      return {
        x: clamp(x, 0, state.width),
        y: clamp(y, 0, state.height),
      };
    }

    function resizeCanvas() {
      if (!image.complete || !image.naturalWidth) {
        return;
      }
      const maxWidth = Math.max(320, Math.floor((window.innerWidth - 72) * 0.62));
      const maxHeight = Math.max(240, window.innerHeight - 220);
      const scale = Math.min(
        maxWidth / image.naturalWidth,
        maxHeight / image.naturalHeight,
        1.0,
      );
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      canvas.style.width = `${canvas.width}px`;
      canvas.style.height = `${canvas.height}px`;
      render();
    }

    function drawCheckerboard(width, height, size = 24) {
      ctx.save();
      for (let y = 0; y < height; y += size) {
        for (let x = 0; x < width; x += size) {
          const even = ((x / size) + (y / size)) % 2 === 0;
          ctx.fillStyle = even ? "#e7dccb" : "#f7f2e8";
          ctx.fillRect(x, y, size, size);
        }
      }
      ctx.restore();
    }

    function renderPolygon(points, color) {
      if (!points.length) {
        return;
      }
      ctx.save();
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.fillStyle = color.replace("0.92", "0.20").replace("0.86", "0.20");
      ctx.beginPath();
      ctx.moveTo(imageToCanvasX(points[0].x), imageToCanvasY(points[0].y));
      for (let index = 1; index < points.length; index += 1) {
        ctx.lineTo(imageToCanvasX(points[index].x), imageToCanvasY(points[index].y));
      }
      if (points.length >= 3) {
        ctx.closePath();
        ctx.fill();
      }
      ctx.stroke();
      for (const point of points) {
        ctx.beginPath();
        ctx.arc(imageToCanvasX(point.x), imageToCanvasY(point.y), 3.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
      ctx.restore();
    }

    function render() {
      if (!image.complete || !image.naturalWidth) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawCheckerboard(canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      const color = state.mode === "add" ? "rgba(52, 128, 89, 0.86)" : "rgba(170, 92, 19, 0.92)";
      renderPolygon(state.points, color);
    }

    function syncPreview() {
      previewImg.src = `${image.src}&preview=${Date.now()}`;
    }

    async function loadState(index) {
      const token = ++state.loadToken;
      setStatus(`Loading cutout ${index + 1} of ${state.count}...`, "info");
      const response = await fetch(`/api/state?index=${index}`, {
        headers: {"Cache-Control": "no-cache"},
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      if (token !== state.loadToken) {
        return;
      }
      state.index = payload.index;
      state.count = payload.count;
      state.width = payload.width;
      state.height = payload.height;
      state.displayName = payload.display_name;
      state.points = (payload.points || []).map((point) => ({
        x: point[0],
        y: point[1],
      }));
      state.mode = payload.mode || "add";
      image.onload = () => {
        if (token !== state.loadToken) {
          return;
        }
        resizeCanvas();
      };
      image.src = `${payload.image_url}&t=${Date.now()}`;
      previewImg.src = `${payload.image_url}&preview=${Date.now()}`;
      window.location.hash = `i=${payload.index}`;
      document.getElementById("mode-btn").textContent = `Mode: ${state.mode === "add" ? "Add" : "Replace"}`;
      updateInfo();
      setStatus(`Loaded ${payload.display_name}`, "ok");
    }

    function currentPointsPayload() {
      return state.points.map((point) => [point.x, point.y]);
    }

    async function saveCurrent() {
      const response = await fetch("/api/save", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          index: state.index,
          width: state.width,
          height: state.height,
          points: currentPointsPayload(),
          mode: state.mode,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const suffix = state.points.length === 1 ? "" : "s";
      setStatus(
        `Saved ${state.points.length} point${suffix} in ${state.mode} mode for ${state.displayName}`,
        "ok",
      );
    }

    async function step(delta) {
      await saveCurrent();
      const nextIndex = state.index + delta;
      if (nextIndex < 0 || nextIndex >= state.count) {
        setStatus("Reached the end of the cutout list.", "warn");
        return;
      }
      await loadState(nextIndex);
    }

    async function quitSession() {
      try {
        await saveCurrent();
      } catch (error) {
        setStatus(`Save failed before quit: ${error.message}`, "warn");
        return;
      }
      try {
        await fetch("/api/quit", {method: "POST"});
      } catch (error) {
        // The server may already be closing; ignore the network error.
      }
      setStatus("Server stopped. You can close this tab.", "ok");
    }

    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || !image.complete || !image.naturalWidth) {
        return;
      }
      event.preventDefault();
      state.points.push(pointerToImage(event));
      updateInfo();
      render();
    });

    document.addEventListener("keydown", async (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey) {
        return;
      }
      if (event.key === "n" || event.key === "Enter") {
        event.preventDefault();
        try {
          await step(1);
        } catch (error) {
          setStatus(`Next failed: ${error.message}`, "warn");
        }
        return;
      }
      if (event.key === "p") {
        event.preventDefault();
        try {
          await step(-1);
        } catch (error) {
          setStatus(`Previous failed: ${error.message}`, "warn");
        }
        return;
      }
      if (event.key === "s") {
        event.preventDefault();
        try {
          await saveCurrent();
        } catch (error) {
          setStatus(`Save failed: ${error.message}`, "warn");
        }
        return;
      }
      if (event.key === "u" || event.key === "Backspace" || event.key === "Delete") {
        event.preventDefault();
        state.points.pop();
        updateInfo();
        render();
        setStatus(`Removed last point. Remaining: ${state.points.length}`, "warn");
        return;
      }
      if (event.key === "r") {
        event.preventDefault();
        state.points = [];
        updateInfo();
        render();
        setStatus("Cleared all polygon points.", "warn");
        return;
      }
      if (event.key === "m") {
        event.preventDefault();
        state.mode = state.mode === "add" ? "replace" : "add";
        document.getElementById("mode-btn").textContent = `Mode: ${state.mode === "add" ? "Add" : "Replace"}`;
        render();
        setStatus(`Switched mode to ${state.mode}.`, "info");
        return;
      }
      if (event.key === "q") {
        event.preventDefault();
        await quitSession();
      }
    });

    document.getElementById("prev-btn").addEventListener("click", async () => {
      try {
        await step(-1);
      } catch (error) {
        setStatus(`Previous failed: ${error.message}`, "warn");
      }
    });

    document.getElementById("next-btn").addEventListener("click", async () => {
      try {
        await step(1);
      } catch (error) {
        setStatus(`Next failed: ${error.message}`, "warn");
      }
    });

    document.getElementById("save-btn").addEventListener("click", async () => {
      try {
        await saveCurrent();
      } catch (error) {
        setStatus(`Save failed: ${error.message}`, "warn");
      }
    });

    document.getElementById("undo-btn").addEventListener("click", () => {
      state.points.pop();
      updateInfo();
      render();
      setStatus(`Removed last point. Remaining: ${state.points.length}`, "warn");
    });

    document.getElementById("clear-btn").addEventListener("click", () => {
      state.points = [];
      updateInfo();
      render();
      setStatus("Cleared all polygon points.", "warn");
    });

    document.getElementById("mode-btn").addEventListener("click", () => {
      state.mode = state.mode === "add" ? "replace" : "add";
      document.getElementById("mode-btn").textContent = `Mode: ${state.mode === "add" ? "Add" : "Replace"}`;
      render();
      setStatus(`Switched mode to ${state.mode}.`, "info");
    });

    document.getElementById("quit-btn").addEventListener("click", quitSession);
    window.addEventListener("resize", resizeCanvas);

    function initialIndexFromHash() {
      const match = window.location.hash.match(/i=(\\d+)/);
      if (!match) {
        return 0;
      }
      return clamp(parseInt(match[1], 10), 0, Math.max(0, CONFIG.count - 1));
    }

    (async () => {
      try {
        if (CONFIG.count <= 0) {
          setStatus("No cutouts were found.", "warn");
          infoEl.textContent = "No cutouts found.";
          return;
        }
        await loadState(initialIndexFromHash());
      } catch (error) {
        setStatus(`Failed to start: ${error.message}`, "warn");
        infoEl.textContent = "The reviewer failed to load.";
      }
    })();
  </script>
</body>
</html>
"""


def _build_html(config: dict[str, Any]) -> str:
    """Render the HTML UI with an embedded JSON configuration blob."""
    config_json = html.escape(json.dumps(config), quote=False)
    return _load_html_template().replace("__CONFIG_JSON__", config_json)


def _parse_index(query: dict[str, list[str]], default: int = 0) -> int:
    """Parse a cutout index from a query dictionary."""
    values = query.get("index")
    if not values:
        return default
    try:
        return int(values[0])
    except ValueError:
        return default


def _point_from_payload(payload: Any) -> tuple[float, float]:
    """Convert a JSON point payload into an `(x, y)` tuple."""
    if isinstance(payload, dict):
        try:
            return float(payload["x"]), float(payload["y"])
        except KeyError as exc:
            raise ValueError("Points must contain x and y values.") from exc

    if isinstance(payload, list) and len(payload) == 2:
        return float(payload[0]), float(payload[1])

    raise ValueError("Points must be two-number arrays or x/y objects.")


def point_in_polygon(x: float, y: float, points: list[tuple[float, float]]) -> bool:
    """Return `True` when a point lies inside a polygon."""
    inside = False
    count = len(points)
    if count < 3:
        return False

    j = count - 1
    for i in range(count):
        xi, yi = points[i]
        xj, yj = points[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            denominator = yj - yi
            if denominator == 0:
                denominator = 1e-9
            x_intersect = (xj - xi) * (y - yi) / denominator + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def polygon_mask_from_points(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> bytearray:
    """Rasterize a polygon into a binary mask."""
    mask = bytearray(width * height)
    if len(points) < 3:
        return mask

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x1 = max(0, int(math.floor(min(x_values))))
    y1 = max(0, int(math.floor(min(y_values))))
    x2 = min(width - 1, int(math.ceil(max(x_values))))
    y2 = min(height - 1, int(math.ceil(max(y_values))))

    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            if point_in_polygon(x + 0.5, y + 0.5, points):
                mask[y * width + x] = 1
    return mask


def union_masks(primary: bytearray, secondary: bytearray) -> bytearray:
    """Merge two masks by keeping every foreground pixel from either mask."""
    return bytearray(max(a, b) for a, b in zip(primary, secondary))


def _make_handler(session: ReviewSession) -> type[BaseHTTPRequestHandler]:
    """Create an HTTP handler class bound to one review session."""

    class CutoutReviewHandler(BaseHTTPRequestHandler):
        """HTTP handler for the browser-based reviewer."""

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            """Emit quieter access logs for the local reviewer."""
            print(f"[cutout] {self.address_string()} - {format % args}")

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            """Send one JSON response."""
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, status: HTTPStatus, text: str) -> None:
            """Send one plain-text response."""
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(
            self,
            status: HTTPStatus,
            body: bytes,
            *,
            content_type: str,
        ) -> None:
            """Send one binary response body."""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            """Serve the HTML UI, review state, or cutout image bytes."""
            parsed = urlparse(self.path)
            if parsed.path == "/":
                config = {"count": session.item_count()}
                body = _build_html(config).encode("utf-8")
                self._send_bytes(
                    HTTPStatus.OK,
                    body,
                    content_type="text/html; charset=utf-8",
                )
                return

            if parsed.path == "/api/state":
                index = _parse_index(parse_qs(parsed.query))
                try:
                    payload = session.build_state(index)
                except IndexError:
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        {"error": "Image index out of range."},
                    )
                    return
                self._send_json(HTTPStatus.OK, payload)
                return

            if parsed.path == "/api/image":
                index = _parse_index(parse_qs(parsed.query))
                try:
                    item = session.get_item(index)
                except IndexError:
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        {"error": "Image index out of range."},
                    )
                    return
                content_type = (
                    mimetypes.guess_type(item.current_path.name)[0]
                    or "application/octet-stream"
                )
                self._send_bytes(
                    HTTPStatus.OK,
                    item.current_path.read_bytes(),
                    content_type=content_type,
                )
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})

        def do_POST(self) -> None:  # noqa: N802
            """Handle save and quit requests from the browser UI."""
            parsed = urlparse(self.path)
            if parsed.path == "/api/quit":
                session.request_shutdown()
                self._send_json(HTTPStatus.OK, {"ok": True})
                return

            if parsed.path != "/api/save":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
                index = int(payload["index"])
                width = int(payload["width"])
                height = int(payload["height"])
                points = [_point_from_payload(point) for point in payload.get("points", [])]
                mode = str(payload.get("mode", "add"))
                if mode not in {"add", "replace"}:
                    raise ValueError("Mode must be either 'add' or 'replace'.")

                item = session.get_item(index)
                if width != item.width or height != item.height:
                    raise ValueError(
                        "Image dimensions in the save request do not match the session."
                    )
                session.save_review(index, points, mode=mode)
            except (KeyError, TypeError, ValueError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.OK, {"ok": True, "points": len(points), "mode": mode})

    return CutoutReviewHandler


class CutoutReviewServer(ThreadingHTTPServer):
    """HTTP server that handles reviewer requests in background threads."""

    daemon_threads = True


def _prepare_session(
    inputs: list[str],
    output_root: Path,
    recursive: bool,
    include_reviewed: bool,
) -> ReviewSession:
    """Collect cutouts and create a review session object."""
    items = _collect_items(
        inputs,
        output_root,
        recursive=recursive,
        include_reviewed=include_reviewed,
    )
    if not items:
        raise SystemExit("No supported cutout images were found in the provided inputs.")
    return ReviewSession(items, output_root)


def serve(args: argparse.Namespace) -> None:
    """Start the reviewer server and optionally open the browser UI."""
    output_root = Path(args.output_root)
    session = _prepare_session(
        args.inputs,
        output_root,
        args.recursive,
        args.include_reviewed,
    )
    handler = _make_handler(session)
    server = CutoutReviewServer((args.host, args.port), handler)
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"[cutout] Serving {session.item_count()} cutout(s) at {url}")

    if not args.no_browser:
        opened = webbrowser.open(url, new=1, autoraise=True)
        if not opened:
            print(f"[cutout] Browser did not open automatically. Visit {url}")
    else:
        print(f"[cutout] Browser auto-open disabled. Visit {url}")

    try:
        while not session.stop_event.wait(timeout=0.25):
            pass
    except KeyboardInterrupt:
        session.request_shutdown()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        print("[cutout] Server stopped.")


def main() -> None:
    args = parse_args()
    if args.prune_identical:
        pruned = _prune_identical_outputs(Path(args.output_root))
        print(f"[cutout] Pruned {pruned} identical reviewed cutout(s).")
        return
    serve(args)


if __name__ == "__main__":
    main()
