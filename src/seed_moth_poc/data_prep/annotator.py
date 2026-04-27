"""Browser-based bounding-box annotator for reference images.

The annotator runs a local HTTP server, opens a browser UI, and writes YOLO
labels beside the reference images. It avoids GUI toolkits like Tkinter, which
are brittle in some macOS environments.
"""

import argparse
import html
import json
import mimetypes
import threading
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from seed_moth_poc.data_prep.commons import (
    Box,
    collect_image_files,
    ensure_directory,
    load_image,
    read_yolo_boxes,
    write_yolo_boxes,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 0
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg"}
MIN_BOX_SIZE = 2.0
DEFAULT_IMAGES = [Path("data/reference/target/images")]
DEFAULT_LABELS = Path("data/reference/target/labels")


@dataclass(slots=True)
class AnnotationItem:
    """Metadata and labels for one image in the annotation session."""

    image_path: Path
    label_path: Path
    width: int
    height: int
    display_name: str
    boxes: list[Box] = field(default_factory=list)


class AnnotationSession:
    """Mutable annotation session state shared by the HTTP handler."""

    def __init__(self, items: list[AnnotationItem]) -> None:
        """Create a session from a list of annotation items."""
        self.items = items
        self.stop_event = threading.Event()
        self._lock = threading.Lock()

    def item_count(self) -> int:
        """Return the number of images in the session."""
        return len(self.items)

    def get_item(self, index: int) -> AnnotationItem:
        """Return one image item or raise `IndexError` if the index is invalid."""
        return self.items[index]

    def build_state(self, index: int) -> dict[str, Any]:
        """Build a JSON-serializable state payload for one image."""
        item = self.get_item(index)
        return {
            "index": index,
            "count": self.item_count(),
            "display_name": item.display_name,
            "image_url": f"/api/image?index={index}",
            "label_path": str(item.label_path),
            "width": item.width,
            "height": item.height,
            "boxes": [
                [box.x1, box.y1, box.x2, box.y2]
                for box in item.boxes
            ],
        }

    def save_boxes(
        self,
        index: int,
        boxes: list[Box],
        *,
        width: int,
        height: int,
    ) -> None:
        """Persist boxes for one image as a YOLO label file."""
        with self._lock:
            item = self.get_item(index)
            if width != item.width or height != item.height:
                raise ValueError(
                    "Image dimensions in the save request do not match the session."
                )
            item.boxes = list(boxes)
            write_yolo_boxes(
                item.label_path,
                item.boxes,
                width=item.width,
                height=item.height,
            )

    def request_shutdown(self) -> None:
        """Signal that the server should stop."""
        self.stop_event.set()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the annotator."""
    parser = argparse.ArgumentParser(
        description="Annotate reference images with YOLO bounding boxes.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=DEFAULT_IMAGES,
        help="One or more image files or directories to annotate.",
    )
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help="Directory where YOLO label files should be written.",
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
    return parser.parse_args()


def _collect_items(
    inputs: list[str],
    labels_root: Path,
    *,
    recursive: bool,
) -> list[AnnotationItem]:
    """Collect and normalize annotation items from the provided inputs."""
    items: list[AnnotationItem] = []
    labels_root = ensure_directory(labels_root)

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
            for image_path in files:
                try:
                    image = load_image(image_path)
                except Exception as exc:  # pragma: no cover - defensive guard
                    print(
                        f"[annotator] Skipping {image_path}: failed to load ({exc}).",
                        file=sys.stderr,
                    )
                    continue

                relative_path = image_path.name
                try:
                    relative_path = image_path.relative_to(root).as_posix()
                except ValueError:
                    relative_path = image_path.name

                label_path = labels_root / Path(relative_path).with_suffix(".txt")
                boxes = read_yolo_boxes(
                    label_path,
                    width=image.width,
                    height=image.height,
                )
                items.append(
                    AnnotationItem(
                        image_path=image_path,
                        label_path=label_path,
                        width=image.width,
                        height=image.height,
                        display_name=relative_path,
                        boxes=boxes,
                    )
                )

    items.sort(key=lambda item: item.display_name.lower())
    return items


def _load_html_template() -> str:
    """Return the HTML document used by the annotation UI."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Seed moth annotator</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f1e7;
      --panel: rgba(255, 252, 245, 0.94);
      --panel-border: rgba(89, 70, 30, 0.16);
      --ink: #2b2215;
      --muted: #6e6250;
      --accent: #8e5f14;
      --accent-soft: rgba(142, 95, 20, 0.12);
      --warn: #a05428;
      --ok: #2f6a4f;
      --shadow: 0 18px 48px rgba(78, 56, 18, 0.16);
    }

    * {
      box-sizing: border-box;
    }

    html,
    body {
      height: 100%;
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 255, 255, 0.7), transparent 36%),
        linear-gradient(180deg, #f8f3ea, #efe4d3 55%, #e7dbc7);
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

    .status[data-kind="warn"] {
      color: var(--warn);
    }

    .status[data-kind="ok"] {
      color: var(--ok);
    }

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
      grid-template-columns: minmax(0, 1fr);
      gap: 12px;
      align-items: start;
      min-height: 0;
      padding: 14px;
    }

    .canvas-wrap {
      display: flex;
      justify-content: center;
      align-items: flex-start;
      overflow: auto;
      min-height: 0;
    }

    canvas {
      display: block;
      border-radius: 12px;
      border: 1px solid rgba(91, 66, 21, 0.25);
      background: #fffdf7;
      cursor: crosshair;
      box-shadow: 0 10px 34px rgba(75, 51, 13, 0.16);
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
      <div class="brand">Seed moth annotator</div>
      <div class="status" id="status" data-kind="info">Loading session...</div>
      <div class="controls">
        <button id="prev-btn" type="button">Prev</button>
        <button id="next-btn" type="button">Next</button>
        <button id="save-btn" type="button">Save</button>
        <button id="undo-btn" type="button">Undo</button>
        <button id="clear-btn" type="button">Clear</button>
        <button id="quit-btn" type="button">Quit</button>
      </div>
    </div>

    <div class="panel stage">
      <div class="footer" id="info">Use the mouse to draw boxes around the moth.
        Shortcuts: <code>n</code> or <code>Enter</code> next, <code>p</code> prev,
        <code>s</code> save, <code>u</code> or <code>Backspace</code> undo,
        <code>r</code> clear, <code>q</code> quit.
      </div>
      <div class="canvas-wrap">
        <canvas id="canvas"></canvas>
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
    const image = new Image();
    const MIN_BOX_SIZE = 2.0;

    const state = {
      index: 0,
      count: CONFIG.count,
      width: 1,
      height: 1,
      displayName: "",
      boxes: [],
      draft: null,
      dragging: false,
      startPoint: null,
      loadToken: 0,
    };

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function setStatus(message, kind = "info") {
      statusEl.textContent = message;
      statusEl.dataset.kind = kind;
    }

    function normalizeBox(box) {
      const x1 = clamp(Math.min(box.x1, box.x2), 0, state.width);
      const y1 = clamp(Math.min(box.y1, box.y2), 0, state.height);
      const x2 = clamp(Math.max(box.x1, box.x2), 0, state.width);
      const y2 = clamp(Math.max(box.y1, box.y2), 0, state.height);
      return {x1, y1, x2, y2};
    }

    function boxSize(box) {
      return {
        width: Math.abs(box.x2 - box.x1),
        height: Math.abs(box.y2 - box.y1),
      };
    }

    function updateInfo() {
      infoEl.innerHTML = [
        `Image <strong>${state.index + 1}/${state.count}</strong>:`,
        `<code>${state.displayName}</code>`,
        `&middot; boxes: <strong>${state.boxes.length}</strong>`,
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
      const maxWidth = Math.max(320, window.innerWidth - 56);
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

    function drawBox(box, color, dash = false) {
      const left = imageToCanvasX(box.x1);
      const top = imageToCanvasY(box.y1);
      const width = imageToCanvasX(box.x2) - left;
      const height = imageToCanvasY(box.y2) - top;
      ctx.save();
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      if (dash) {
        ctx.setLineDash([8, 6]);
      }
      ctx.strokeRect(left, top, width, height);
      ctx.fillRect(left, top, 42, 18);
      ctx.fillStyle = "#fffdf7";
      ctx.font =
        "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
      ctx.textBaseline = "middle";
      ctx.fillText(`${Math.round(box.x1)},${Math.round(box.y1)}`, left + 5, top + 9);
      ctx.restore();
    }

    function render() {
      if (!image.complete || !image.naturalWidth) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

      state.boxes.forEach((box) => drawBox(box, "rgba(144, 75, 18, 0.95)"));

      if (state.draft) {
        drawBox(state.draft, "rgba(44, 104, 78, 0.95)", true);
      }
    }

    async function loadState(index) {
      const token = ++state.loadToken;
      setStatus(`Loading image ${index + 1} of ${state.count}...`, "info");
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
      state.boxes = payload.boxes.map((box) => ({
        x1: box[0],
        y1: box[1],
        x2: box[2],
        y2: box[3],
      }));
      state.draft = null;
      state.dragging = false;
      state.startPoint = null;
      image.onload = () => {
        if (token !== state.loadToken) {
          return;
        }
        resizeCanvas();
      };
      image.src = `${payload.image_url}&t=${Date.now()}`;
      window.location.hash = `i=${payload.index}`;
      updateInfo();
      setStatus(`Loaded ${payload.display_name}`, "ok");
    }

    async function saveCurrent() {
      const boxes = state.boxes.map((box) => {
        const normalized = normalizeBox(box);
        return [normalized.x1, normalized.y1, normalized.x2, normalized.y2];
      });
      const response = await fetch("/api/save", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          index: state.index,
          width: state.width,
          height: state.height,
          boxes,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const suffix = boxes.length === 1 ? "" : "es";
      setStatus(
        `Saved ${boxes.length} box${suffix} to ${state.displayName}`,
        "ok",
      );
    }

    async function step(delta) {
      await saveCurrent();
      const nextIndex = state.index + delta;
      if (nextIndex < 0 || nextIndex >= state.count) {
        setStatus("Reached the end of the image list.", "warn");
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
      canvas.setPointerCapture(event.pointerId);
      state.dragging = true;
      state.startPoint = pointerToImage(event);
      state.draft = {
        x1: state.startPoint.x,
        y1: state.startPoint.y,
        x2: state.startPoint.x,
        y2: state.startPoint.y,
      };
      render();
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!state.dragging || !state.startPoint) {
        return;
      }
      event.preventDefault();
      const point = pointerToImage(event);
      state.draft = {
        x1: state.startPoint.x,
        y1: state.startPoint.y,
        x2: point.x,
        y2: point.y,
      };
      render();
    });

    function finishDraft() {
      if (!state.draft) {
        return;
      }
      const box = normalizeBox(state.draft);
      const size = boxSize(box);
      state.dragging = false;
      state.startPoint = null;
      state.draft = null;
      if (size.width >= MIN_BOX_SIZE && size.height >= MIN_BOX_SIZE) {
        state.boxes.push(box);
        updateInfo();
        setStatus(`Box added. Total boxes: ${state.boxes.length}`, "info");
      }
      render();
    }

    canvas.addEventListener("pointerup", (event) => {
      if (!state.dragging) {
        return;
      }
      event.preventDefault();
      state.dragging = false;
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      finishDraft();
    });

    canvas.addEventListener("pointercancel", () => {
      state.dragging = false;
      state.startPoint = null;
      state.draft = null;
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
        state.boxes.pop();
        updateInfo();
        render();
        setStatus(`Removed last box. Remaining: ${state.boxes.length}`, "warn");
        return;
      }
      if (event.key === "r") {
        event.preventDefault();
        state.boxes = [];
        updateInfo();
        render();
        setStatus("Cleared all boxes for the current image.", "warn");
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
      state.boxes.pop();
      updateInfo();
      render();
      setStatus(`Removed last box. Remaining: ${state.boxes.length}`, "warn");
    });

    document.getElementById("clear-btn").addEventListener("click", () => {
      state.boxes = [];
      updateInfo();
      render();
      setStatus("Cleared all boxes for the current image.", "warn");
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
          setStatus("No images were found.", "warn");
          infoEl.textContent = "No images found.";
          return;
        }
        await loadState(initialIndexFromHash());
      } catch (error) {
        setStatus(`Failed to start: ${error.message}`, "warn");
        infoEl.textContent = "The annotator failed to load.";
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
    """Parse an image index from a query dictionary."""
    values = query.get("index")
    if not values:
        return default
    try:
        return int(values[0])
    except ValueError:
        return default


def _box_from_payload(payload: Any) -> Box:
    """Convert a JSON box payload into a `Box` instance."""
    if isinstance(payload, dict):
        try:
            return Box(
                float(payload["x1"]),
                float(payload["y1"]),
                float(payload["x2"]),
                float(payload["y2"]),
            )
        except KeyError as exc:
            raise ValueError("Boxes must contain x1, y1, x2, and y2.") from exc

    if isinstance(payload, list) and len(payload) == 4:
        return Box(
            float(payload[0]),
            float(payload[1]),
            float(payload[2]),
            float(payload[3]),
        )

    raise ValueError("Boxes must be four-number arrays or x1/y1/x2/y2 objects.")


def _make_handler(session: AnnotationSession) -> type[BaseHTTPRequestHandler]:
    """Create an HTTP handler class bound to one annotation session."""

    class AnnotatorHandler(BaseHTTPRequestHandler):
        """HTTP handler for the browser-based annotator."""

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            """Emit quieter access logs for the local annotator."""
            print(f"[annotator] {self.address_string()} - {format % args}")

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
            """Serve the HTML UI, session state, or image bytes."""
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
                    mimetypes.guess_type(item.image_path.name)[0]
                    or "application/octet-stream"
                )
                self._send_bytes(
                    HTTPStatus.OK,
                    item.image_path.read_bytes(),
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
                boxes = [_box_from_payload(box) for box in payload.get("boxes", [])]
                session.save_boxes(index, boxes, width=width, height=height)
            except (KeyError, TypeError, ValueError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.OK, {"ok": True, "boxes": len(boxes)})

    return AnnotatorHandler


class AnnotatorServer(ThreadingHTTPServer):
    """HTTP server that handles annotator requests in background threads."""

    daemon_threads = True


def _prepare_session(
    inputs: list[str],
    labels_root: Path,
    recursive: bool,
) -> AnnotationSession:
    """Collect images, hydrate existing labels, and create a session object."""
    items = _collect_items(inputs, labels_root, recursive=recursive)
    if not items:
        raise SystemExit("No supported images were found in the provided inputs.")
    return AnnotationSession(items)


def serve(args: argparse.Namespace) -> None:
    """Start the annotator server and optionally open the browser UI."""
    labels_root = Path(args.labels)
    session = _prepare_session(args.images, labels_root, args.recursive)
    handler = _make_handler(session)
    server = AnnotatorServer((args.host, args.port), handler)
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(
        f"[annotator] Serving {session.item_count()} image(s) at {url}",
        file=sys.stderr,
    )

    if not args.no_browser:
        opened = webbrowser.open(url, new=1, autoraise=True)
        if not opened:
            print(
                f"[annotator] Browser did not open automatically. Visit {url}",
                file=sys.stderr,
            )
    else:
        print(f"[annotator] Browser auto-open disabled. Visit {url}", file=sys.stderr)

    try:
        while not session.stop_event.wait(timeout=0.25):
            pass
    except KeyboardInterrupt:
        session.request_shutdown()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        print("[annotator] Server stopped.", file=sys.stderr)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    serve(args)


if __name__ == "__main__":
    main()
