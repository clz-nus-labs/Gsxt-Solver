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


def _item_type(item: dict[str, Any]) -> str:
    kind = str(item.get("kind") or "").strip().lower()
    return kind if kind in {"char", "icon"} else "unknown"


def _looks_like_chinese_text(value: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in value)


def _task_type(debug_result: dict[str, Any], items: list[dict[str, Any]]) -> str:
    """Return the public task modality.

    Older diagnostic payloads may use ``mixed`` for auto-detection or
    detect-only fallback. The public API intentionally exposes only the
    resolved modality used by downstream callers: ``char`` or ``icon``.
    """

    task_spec = debug_result.get("task_spec") or {}
    raw = str(
        debug_result.get("task_type")
        or task_spec.get("modality")
        or ""
    ).strip().lower()
    if raw in {"char", "icon"}:
        return raw

    counts = {"char": 0, "icon": 0}
    for item in items:
        kind = str(item.get("type") or "").strip().lower()
        if kind in counts:
            counts[kind] += 1
    if counts["char"] > counts["icon"]:
        return "char"
    if counts["icon"] > counts["char"]:
        return "icon"

    values = [str(item.get("value") or "") for item in items]
    if any(_looks_like_chinese_text(value) for value in values):
        return "char"
    if items:
        return "icon"
    return "unknown"


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
            "type": _item_type(item),
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
            "type": _task_type(debug_result, items),
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
