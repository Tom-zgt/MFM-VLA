"""Token-budgeted and disk-cached wrapper around MFMClientSession."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from mfm_client import DEFAULT_BASE_URL, MFMClientSession


class AtomicTokenLedger:
    """Process-safe JSON token counter for capability calibration and routing."""

    def __init__(self, path: Path, budget_tokens: Optional[int] = None) -> None:
        self.path = Path(path)
        self.budget_tokens = budget_tokens
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                json.dumps(
                    {
                        "requests": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    indent=2,
                )
                + "\n"
            )

    def read(self) -> dict:
        if not self.path.exists():
            return {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return json.loads(self.path.read_text())

    def record(self, usage: dict) -> dict:
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("w") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                state = self.read()
                state["requests"] = int(state.get("requests", 0)) + 1
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    state[key] = int(state.get(key, 0)) + int(usage.get(key, 0) or 0)
                self.path.write_text(json.dumps(state, indent=2) + "\n")
                return state
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


class BudgetedMFMSession:
    """Cached MFM session recording exact token usage to AtomicTokenLedger."""

    def __init__(
        self,
        ledger: AtomicTokenLedger,
        cache_dir: Path,
        *,
        model: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 1000,
        reserve_tokens: int = 8000,
    ) -> None:
        self.ledger = ledger
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reserve_tokens = reserve_tokens
        self._inner = MFMClientSession(
            model=model or os.environ.get("MFM_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o")),
            base_url=base_url,
            max_tokens=max_tokens,
        )
        self.last_call: Optional[dict] = None

    def _hash_call(self, prompt: str, rgb: Dict[str, np.ndarray]) -> str:
        digest = hashlib.sha256()
        digest.update(self._inner.model.encode("utf-8"))
        digest.update(prompt.encode("utf-8"))
        for name, img in sorted(rgb.items()):
            digest.update(name.encode("utf-8"))
            arr = np.ascontiguousarray(img, dtype=np.uint8)
            digest.update(str(arr.shape).encode("ascii"))
            digest.update(arr.tobytes())
        return digest.hexdigest()

    def decide_json(self, prompt: str, rgb: Dict[str, np.ndarray]) -> dict:
        input_hash = self._hash_call(prompt, rgb)
        cache_file = self.cache_dir / f"{input_hash}.json"
        if cache_file.is_file():
            cached = json.loads(cache_file.read_text())
            self.last_call = {
                "input_hash": input_hash,
                "cache_hit": True,
                "usage": cached.get("usage") or {},
            }
            return cached["decision"]

        if self.ledger.budget_tokens is not None:
            used = int(self.ledger.read().get("total_tokens", 0))
            if used + self.reserve_tokens > self.ledger.budget_tokens:
                raise RuntimeError(
                    f"Token budget exceeded ({used} + {self.reserve_tokens} > {self.ledger.budget_tokens})"
                )

        decision = self._inner.decide_json(prompt, rgb)
        usage = self._inner.decisions[-1].get("usage") or {}
        self.ledger.record(usage)
        cache_file.write_text(
            json.dumps({"decision": decision, "usage": usage}, ensure_ascii=False, indent=2) + "\n"
        )
        self.last_call = {
            "input_hash": input_hash,
            "cache_hit": False,
            "usage": usage,
        }
        return decision

    def usage(self) -> dict:
        return self._inner.usage()

    def close(self) -> None:
        self._inner.close()
