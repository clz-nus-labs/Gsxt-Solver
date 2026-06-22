from __future__ import annotations

from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


def _value(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("label") or "")


def _center(item: dict[str, Any]) -> dict[str, int]:
    center = item.get("center") or [0, 0]
    return {"x": int(center[0]), "y": int(center[1])}


def _bbox(item: dict[str, Any]) -> dict[str, int]:
    bbox = item.get("bbox") or [0, 0, 0, 0]
    return {
        "left": int(bbox[0]),
        "top": int(bbox[1]),
        "right": int(bbox[2]),
        "bottom": int(bbox[3]),
    }


def format_standard_result(
    debug_result: dict[str, Any],
    *,
    image: str | Path,
) -> dict[str, Any]:
    """Convert the backend's diagnostic payload into the stable public schema."""

    image_path = Path(image)
    task_spec = debug_result.get("task_spec") or {}
    raw_items = debug_result.get("items") or []
    items = [
        {
            "index": index,
            "type": str(item.get("kind") or "unknown"),
            "value": _value(item),
            "center": _center(item),
            "bbox": _bbox(item),
        }
        for index, item in enumerate(raw_items, start=1)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "success": True,
        "image": image_path.name,
        "task": {
            "action": str(task_spec.get("action") or "detect_only"),
            "type": str(
                debug_result.get("task_type")
                or task_spec.get("modality")
                or "mixed"
            ),
        },
        "result": {
            "count": len(items),
            "sequence": [item["value"] for item in items],
            "points": [item["center"] for item in items],
            "items": items,
        },
    }


def format_error_result(
    *,
    image: str | Path,
    error: Exception,
) -> dict[str, Any]:
    """Create a stable error payload without exposing internal diagnostics."""

    message = str(error).splitlines()[0].strip() or type(error).__name__
    return {
        "schema_version": SCHEMA_VERSION,
        "success": False,
        "image": Path(image).name,
        "error": {
            "type": type(error).__name__,
            "message": message,
        },
    }
