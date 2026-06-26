from __future__ import annotations

import json
import time
from pathlib import Path

from flask import Flask, jsonify, request

from gsxt_solver_bridge import solve_image_base64


app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DEBUG_DIR = BASE_DIR / "logs" / "captcha_debug"

# Keep CPU as the default for browser integration. Change to True only when the
# Paddle/CUDA/CUDNN environment is known to be stable on this machine.
USE_GPU = False

# Empty means: use dist/models/gsxt-models-v0.1.0 under the repository root.
MODEL_DIR = ""


def _point_in_challenge_area(
    point: dict,
    image_size: tuple[int, int] | None,
    *,
    challenge_only_crop: bool = False,
) -> bool:
    if not image_size:
        return True
    width, height = image_size
    x = float(point.get("x", 0))
    y = float(point.get("y", 0))
    if challenge_only_crop:
        return (
            width * 0.02 <= x <= width * 0.98
            and height * 0.05 <= y <= height * 0.95
        )
    return (
        width * 0.04 <= x <= width * 0.96
        and height * 0.16 <= y <= height * 0.70
    )


def postprocess_geetest_result(
    result: dict,
    image_size: tuple[int, int] | None,
    crop_info: dict | None = None,
) -> dict:
    """Filter obvious non-target controls from a Geetest crop.

    The browser extension passes full Geetest panel crops. The model can detect
    the blue confirm button or footer icons as body targets. Keep only points in
    the challenge image area and require exactly three targets before auto-click.
    """
    if not result.get("success") or not image_size:
        return result

    crop_info = crop_info or {}
    challenge_only_crop = crop_info.get("solverCropMode") == "challenge_only"
    points = list(result.get("points") or [])
    items = list(result.get("items") or [])
    if not points:
        return result

    kept_indexes = [
        idx for idx, point in enumerate(points)
        if _point_in_challenge_area(
            point,
            image_size,
            challenge_only_crop=challenge_only_crop,
        )
    ]
    if len(kept_indexes) != len(points):
        result["postprocess"] = {
            **(result.get("postprocess") or {}),
            "filtered_non_challenge_points": len(points) - len(kept_indexes),
            "original_points": points,
            "challenge_only_crop": challenge_only_crop,
        }
        result["points"] = [points[idx] for idx in kept_indexes]
        result["items"] = [items[idx] for idx in kept_indexes if idx < len(items)]
        result["sequence"] = [
            str((result["items"][idx] or {}).get("value") or "")
            for idx in range(len(result.get("items") or []))
        ]

    if len(result.get("points") or []) != 3:
        result["success"] = False
        result["error"] = (
            "geetest target count is not 3 after filtering: "
            f"{len(result.get('points') or [])}"
        )
        result["postprocess"] = {
            **(result.get("postprocess") or {}),
            "requires_manual": True,
            "expected_targets": 3,
            "image_size": {"width": image_size[0], "height": image_size[1]},
            "challenge_only_crop": challenge_only_crop,
        }
    else:
        result["postprocess"] = {
            **(result.get("postprocess") or {}),
            "challenge_only_crop": challenge_only_crop,
            "visual_glyph_mode": result.get("task", {}).get("type") == "icon",
            "labels_untrusted": result.get("task", {}).get("type") == "icon",
            "note": (
                "Geetest colored Chinese glyphs are often classified as icons; "
                "use points/order, not labels."
            ),
        }
    return result


@app.after_request
def add_cors(response):
    """Allow calls from the local Chrome/Edge extension."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "solver": "gsxt_solver",
        "use_gpu": USE_GPU,
        "routes": ["health", "debug-capture", "solve", "click-report"],
    })


@app.route("/debug-capture", methods=["POST", "OPTIONS"])
def debug_capture():
    """Save a cropped captcha image before model inference.

    This makes frontend/click/debug failures easier to diagnose because the
    crop is preserved even if the following /solve call or page click fails.
    """
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}
    image_base64 = data.get("image_base64") or data.get("image") or ""
    debug_id = str(data.get("debug_id") or f"captcha_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 100000}")
    reason = str(data.get("reason") or "pre_solve_capture")
    crop_info = data.get("crop_info") or {}
    debug_png = DEBUG_DIR / f"{debug_id}.png"
    debug_json = DEBUG_DIR / f"{debug_id}.capture.json"

    if not image_base64:
        return jsonify({"success": False, "error": "missing image_base64"}), 400

    try:
        from gsxt_solver_bridge import _decode_image_base64

        image_bytes = _decode_image_base64(image_base64)
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        debug_png.write_bytes(image_bytes)
        debug_json.write_text(
            json.dumps(
                {
                    "success": True,
                    "debug_id": debug_id,
                    "debug_image": str(debug_png),
                    "reason": reason,
                    "crop_info": crop_info,
                    "received_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[GSXT_CAPTURE_DEBUG] image={debug_png} reason={reason}")
        if crop_info:
            print(f"[GSXT_CAPTURE_CROP] {json.dumps(crop_info, ensure_ascii=False)}")
        return jsonify({"success": True, "debug_id": debug_id, "debug_image": str(debug_png)})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc), "debug_id": debug_id}), 500


@app.route("/solve", methods=["POST", "OPTIONS"])
def solve():
    """Solve a cropped GSXT captcha image.

    Request JSON:
      {"image_base64": "data:image/png;base64,...", "timeout": 300}

    Response JSON:
      {"success": true, "points": [{"x": 1, "y": 2}, ...], "sequence": [...]}
    """
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}
    image_base64 = data.get("image_base64") or data.get("image") or ""
    timeout = int(data.get("timeout") or 300)
    debug_id = str(data.get("debug_id") or f"captcha_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 100000}")
    crop_info = data.get("crop_info") or {}
    debug_png = DEBUG_DIR / f"{debug_id}.png"
    debug_json = DEBUG_DIR / f"{debug_id}.json"

    if not image_base64:
        return jsonify({"success": False, "error": "missing image_base64"}), 400

    try:
        result = solve_image_base64(
            image_base64,
            model_dir=MODEL_DIR,
            use_gpu=USE_GPU,
            timeout=timeout,
            debug_save_path=debug_png,
        )
        try:
            from PIL import Image

            with Image.open(debug_png) as img:
                image_size = (int(img.width), int(img.height))
        except Exception:
            image_size = None
        result = postprocess_geetest_result(result, image_size, crop_info)
        result["debug_id"] = debug_id
        result["debug_image"] = str(debug_png)
        result["crop_info"] = crop_info
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        debug_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[GSXT_SOLVER_RESULT] success={result.get('success')}")
        print(f"[GSXT_SOLVER_TASK] {json.dumps(result.get('task', {}), ensure_ascii=False)}")
        print(f"[GSXT_SOLVER_SEQUENCE] {json.dumps(result.get('sequence', []), ensure_ascii=False)}")
        print(f"[GSXT_SOLVER_POINTS] {json.dumps(result.get('points', []), ensure_ascii=False)}")
        if result.get("items"):
            print(f"[GSXT_SOLVER_ITEMS] {json.dumps(result.get('items', []), ensure_ascii=False)}")
        if result.get("postprocess"):
            print(f"[GSXT_SOLVER_POSTPROCESS] {json.dumps(result.get('postprocess', {}), ensure_ascii=False)}")
        if result.get("error"):
            print(f"[GSXT_SOLVER_ERROR_MESSAGE] {result.get('error')}")
        if crop_info:
            print(f"[GSXT_SOLVER_CROP] {json.dumps(crop_info, ensure_ascii=False)}")
        print(f"[GSXT_SOLVER_DEBUG] image={debug_png}")
        return jsonify(result)
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        error_result = {
            "success": False,
            "error": str(exc),
            "traceback": tb,
            "debug_id": debug_id,
            "debug_image": str(debug_png),
            "crop_info": crop_info,
        }
        try:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            debug_json.write_text(json.dumps(error_result, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        print("\n" + "=" * 60)
        print("[GSXT_SOLVER_ERROR]")
        print(tb)
        print(f"[GSXT_SOLVER_DEBUG] image={debug_png}")
        print("=" * 60 + "\n")
        return jsonify(error_result), 500


@app.route("/click-report", methods=["POST", "OPTIONS"])
def click_report():
    """Receive browser-side click diagnostics for terminal visibility."""
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}
    stage = data.get("stage") or "unknown"
    debug_id = data.get("debug_id") or "-"
    print("\n" + "-" * 60)
    print(f"[GSXT_CLICK_REPORT] stage={stage} debug_id={debug_id}")

    if data.get("sequence") is not None:
        print(f"sequence: {' -> '.join(map(str, data.get('sequence') or []))}")
    if data.get("points") is not None:
        point_text = " -> ".join(f"({p.get('x')},{p.get('y')})" for p in data.get("points") or [])
        print(f"image_points: {point_text}")

    click = data.get("click")
    if click:
        print(
            "click: "
            f"#{click.get('index')} "
            f"label={click.get('label') or '-'} "
            f"image=({(click.get('imagePoint') or {}).get('x')},{(click.get('imagePoint') or {}).get('y')}) "
            f"css=({(click.get('cssPoint') or {}).get('x')},{(click.get('cssPoint') or {}).get('y')}) "
            f"page=({(click.get('viewportPoint') or {}).get('x')},{(click.get('viewportPoint') or {}).get('y')}) "
            f"target={click.get('target')}"
        )

    if data.get("click_results"):
        print("click_order:")
        for row in data.get("click_results") or []:
            print(
                "  "
                f"{row.get('index')}. "
                f"{row.get('label') or '-'} "
                f"image=({(row.get('imagePoint') or {}).get('x')},{(row.get('imagePoint') or {}).get('y')}) "
                f"css=({(row.get('cssPoint') or {}).get('x')},{(row.get('cssPoint') or {}).get('y')}) "
                f"page=({(row.get('viewportPoint') or {}).get('x')},{(row.get('viewportPoint') or {}).get('y')}) "
                f"target={row.get('target')}"
            )

    if data.get("confirm_click") is not None:
        confirm = data.get("confirm_click")
        if confirm:
            print(
                "confirm: "
                f"method={confirm.get('method')} "
                f"target={confirm.get('target')} "
                f"text={confirm.get('text') or '-'} "
                f"page=({(confirm.get('viewportPoint') or {}).get('x')},{(confirm.get('viewportPoint') or {}).get('y')})"
            )
        else:
            print("confirm: <not clicked>")

    crop = data.get("crop_info") or {}
    if crop:
        crop_pixels = crop.get("cropPixels") or {}
        viewport = crop.get("viewportRect") or {}
        scale = crop.get("scale") or {}
        print(
            "crop: "
            f"element={crop.get('sourceElement')} "
            f"viewport=({viewport.get('left')},{viewport.get('top')},{viewport.get('width')}x{viewport.get('height')}) "
            f"pixels=({crop_pixels.get('x')},{crop_pixels.get('y')},{crop_pixels.get('width')}x{crop_pixels.get('height')}) "
            f"scale=({scale.get('x')},{scale.get('y')})"
        )
    print("-" * 60 + "\n")
    return jsonify({"success": True})


if __name__ == "__main__":
    print("=" * 50)
    print("GSXT Solver local service started")
    print("Listening: http://127.0.0.1:7755")
    print("Endpoint: POST /solve with image_base64")
    print("=" * 50)
    app.run(host="127.0.0.1", port=7755, debug=False)
