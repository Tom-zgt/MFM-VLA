"""Local HTTP inference server for GroundingDINO-Base (`IDEA-Research/grounding-dino-base`).

Pure open-vocabulary 2D bounding-box detector with zero LLM reasoning or task heuristics:
given an RGB PNG/JPEG image and a text query, returns detected bounding boxes sorted by
detection score, together with pure bounding-box geometric center `(x0+x1)/2, (y0+y1)/2`
and bottom-edge midpoint `(x0+x1)/2, y1`.
"""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class GroundingDINORunner:
    def __init__(self, model_dir: str, device: str = "cuda:0") -> None:
        self.device = device
        self.lock = threading.Lock()
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_dir, local_files_only=True)
            .to(device)
            .eval()
        )

    def detect(self, image: Image.Image, query: str, box_threshold: float = 0.05) -> list[dict]:
        q = query.strip()
        if not q.endswith("."):
            q = q + "."
        w, h = image.size
        with self.lock, torch.inference_mode():
            inputs = self.processor(images=image, text=q, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            res = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=box_threshold,
                target_sizes=[(h, w)],
            )[0]
        scores = res["scores"].detach().cpu().tolist()
        boxes = res["boxes"].detach().cpu().tolist()
        labels = res.get("text_labels") or res.get("labels") or [""] * len(scores)
        items = []
        for sc, bx, lb in sorted(zip(scores, boxes, labels), key=lambda x: x[0], reverse=True):
            x0, y0, x1, y1 = [float(v) for v in bx]
            items.append(
                {
                    "score": round(float(sc), 4),
                    "bbox": [x0, y0, x1, y1],
                    "center_uv": [0.5 * (x0 + x1), 0.5 * (y0 + y1)],
                    "base_uv": [0.5 * (x0 + x1), y1],
                    "label": str(lb),
                }
            )
        return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("GDINO_MODEL_DIR", "IDEA-Research/grounding-dino-base"))
    parser.add_argument("--port", type=int, default=9130)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    runner = GroundingDINORunner(args.model_dir, args.device)
    print(f"GroundingDINO server ready on 127.0.0.1:{args.port} ({args.device})", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            return

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            raw = base64.b64decode(payload["image_b64"])
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            dets = runner.detect(
                img,
                str(payload["query"]),
                box_threshold=float(payload.get("box_threshold", 0.05)),
            )
            body = json.dumps({"detections": dets}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
