from __future__ import annotations

import base64
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "dist" / "models" / "gsxt-models-v0.1.0"
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gsxt_solver import Solver


def _decode_image_base64(payload: str) -> bytes:
    payload = payload.strip()
    if payload.lower().startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


@lru_cache(maxsize=1)
def get_solver(model_dir: str = "", use_gpu: bool = False) -> Solver:
    bundle = Path(model_dir).resolve() if model_dir else DEFAULT_MODEL_DIR
    return Solver.from_bundle(PROJECT_ROOT, bundle, use_gpu=use_gpu)


def solve_image_base64(
    image_base64: str,
    *,
    model_dir: str = "",
    use_gpu: bool = False,
    timeout: int = 300,
    debug_save_path: str | Path | None = None,
) -> dict[str, Any]:
    image_bytes = _decode_image_base64(image_base64)
    if debug_save_path:
        debug_path = Path(debug_save_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_bytes(image_bytes)

    with tempfile.TemporaryDirectory(prefix="gjqyxygs-gsxt-captcha-") as tmp:
        image_path = Path(tmp) / "captcha.png"
        image_path.write_bytes(image_bytes)
        result = get_solver(model_dir, use_gpu).predict(image_path, timeout=timeout)

    if not result.get("success"):
        return result

    body = result.get("result", {})
    return {
        "schema_version": "1.0",
        "success": True,
        "image": result.get("image"),
        "task": result.get("task"),
        "sequence": body.get("sequence", []),
        "points": body.get("points", []),
        "items": body.get("items", []),
    }
