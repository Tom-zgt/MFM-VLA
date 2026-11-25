"""Local OpenAI-compatible HTTP server (`/v1/chat/completions`) for Qwen3-VL-30B-A3B-Instruct
visual grasp verification (`GRASP_CHECK_PROMPT`).
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
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration


class LocalVLMVerifier:
    def __init__(self, model_dir: str, device: str = "cuda:0") -> None:
        self.lock = threading.Lock()
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map=device,
            local_files_only=True,
        ).eval()

    def chat(self, messages: list[dict], max_tokens: int = 256) -> dict:
        hf_msgs = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                hf_msgs.append({"role": role, "content": [{"type": "text", "text": content}]})
            else:
                parts = []
                for item in content:
                    t = item.get("type")
                    if t == "text":
                        parts.append({"type": "text", "text": item.get("text", "")})
                    elif t == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if "," in url:
                            b64 = url.split(",", 1)[1]
                            im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                            parts.append({"type": "image", "image": im})
                hf_msgs.append({"role": role, "content": parts})

        with self.lock, torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                hf_msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            in_len = inputs["input_ids"].shape[-1]
            gen = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            out_ids = gen[0][in_len:]
            text = self.processor.decode(out_ids, skip_special_tokens=True)
            out_len = int(out_ids.shape[-1])
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {
                "prompt_tokens": int(in_len),
                "completion_tokens": out_len,
                "total_tokens": int(in_len) + out_len,
            },
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("QWEN_VL_MODEL_DIR", "Qwen/Qwen3-VL-30B-A3B-Instruct"))
    ap.add_argument("--port", type=int, default=9290)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    runner = LocalVLMVerifier(args.model_dir, args.device)
    print(f"Local Qwen3-VL OpenAI server ready on 127.0.0.1:{args.port}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            return

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n))
            resp = runner.chat(req.get("messages", []), max_tokens=int(req.get("max_tokens", 256)))
            body = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
