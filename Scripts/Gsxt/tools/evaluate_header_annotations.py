from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for portability.
    tqdm = None


ROOT = Path(__file__).resolve().parents[3]
GSXT = ROOT / "Scripts" / "Gsxt"
DEFAULT_ANNOTATION = GSXT / "data" / "annotations" / "gsxt_200_simple_annotation.json"
DEFAULT_IMAGE_DIR = GSXT / "data" / "images"
DEFAULT_OUTPUT_DIR = GSXT / "output" / "header_eval_200"
DEFAULT_MODEL_ROOT = ROOT / "dist" / "models" / "gsxt-models-v0.1.0"


def load_annotations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items") or []
    return [item for item in items if item.get("usable", True)]


def model_args() -> list[str]:
    det_weights = GSXT / "output" / "training" / "paddledet_external_mixed" / "best_model.pdparams"
    det_dataset = GSXT / "data" / "datasets" / "external_mixed_paddledet"
    rec_config = GSXT / "output" / "training" / "chinese_char_rec_ppocrv4_domain_finetune" / "config.yml"
    rec_weights = GSXT / "output" / "training" / "chinese_char_rec_ppocrv4_domain_finetune" / "best_accuracy.pdparams"
    icon_weights = GSXT / "output" / "training" / "icon_cls_geetest_plus_synthetic_mobilenet_v3_large" / "best_accuracy.pdparams"
    icon_labels = GSXT / "output" / "training" / "icon_cls_geetest_plus_synthetic_mobilenet_v3_large" / "label_list.txt"
    if (DEFAULT_MODEL_ROOT / "det" / "best_model.pdparams").exists():
        det_weights = DEFAULT_MODEL_ROOT / "det" / "best_model.pdparams"
        det_dataset = DEFAULT_MODEL_ROOT / "det" / "dataset"
        rec_config = DEFAULT_MODEL_ROOT / "rec" / "config.yml"
        rec_weights = DEFAULT_MODEL_ROOT / "rec" / "best_accuracy.pdparams"
        icon_weights = DEFAULT_MODEL_ROOT / "icon" / "best_accuracy.pdparams"
        icon_labels = DEFAULT_MODEL_ROOT / "icon" / "label_list.txt"
    return [
        "--det-config",
        str(GSXT / "third_party" / "PaddleDetection" / "configs" / "picodet" / "picodet_s_320_coco_lcnet.yml"),
        "--det-weights",
        str(det_weights),
        "--det-dataset",
        str(det_dataset),
        "--rec-config",
        str(rec_config),
        "--rec-weights",
        str(rec_weights),
        "--icon-weights",
        str(icon_weights),
        "--icon-labels",
        str(icon_labels),
    ]


def expected_sequence(item: dict[str, Any]) -> list[str]:
    return [str(t.get("value_or_desc") or "").strip() for t in item.get("targets") or []]


def predicted_sequence(result: dict[str, Any]) -> list[str]:
    seq: list[str] = []
    for row in result.get("items") or []:
        seq.append(str(row.get("text") or row.get("label") or "").strip())
    return seq


def is_semantic_action(result: dict[str, Any]) -> bool:
    task = result.get("task_spec") or {}
    return task.get("action") == "semantic_order"


def normalize_chars(values: list[str]) -> str:
    return "".join(values).replace(",", "").replace("，", "").replace(" ", "")


def make_row(annotation: dict[str, Any], result: dict[str, Any] | None, error: str, elapsed: float) -> dict[str, Any]:
    expected = expected_sequence(annotation)
    if result is None:
        return {
            "image": annotation.get("image"),
            "expected_task_type": annotation.get("task_type"),
            "pred_task_type": "",
            "expected_order_mode": annotation.get("order_mode"),
            "pred_order_mode": "",
            "expected_sequence": " | ".join(expected),
            "pred_sequence": "",
            "target_source_resolved": "",
            "header_instruction_ocr": "",
            "header_target_text_ocr": "",
            "resolved_header_target_text": "",
            "header_icon_count": "",
            "task_type_ok": "no",
            "order_mode_ok": "no",
            "char_sequence_ok": "",
            "final_count_ok": "no",
            "elapsed_sec": f"{elapsed:.2f}",
            "error": error,
        }

    task_spec = result.get("task_spec") or {}
    merge_settings = result.get("merge_settings") or {}
    evidence = task_spec.get("evidence") or {}
    pred = predicted_sequence(result)
    pred_order_mode = "semantic_order" if is_semantic_action(result) else "given_order"
    expected_task = str(annotation.get("task_type") or "")
    pred_task = str(result.get("task_type") or task_spec.get("modality") or "")
    char_ok = ""
    if expected_task == "char":
        char_ok = "yes" if normalize_chars(expected) == normalize_chars(pred[: len(expected)]) else "no"
    return {
        "image": annotation.get("image"),
        "expected_task_type": expected_task,
        "pred_task_type": pred_task,
        "expected_order_mode": annotation.get("order_mode"),
        "pred_order_mode": pred_order_mode,
        "expected_sequence": " | ".join(expected),
        "pred_sequence": " | ".join(pred),
        "target_source_resolved": merge_settings.get("target_source_resolved", ""),
        "header_instruction_ocr": evidence.get("instruction_text", ""),
        "header_instruction_score": evidence.get("instruction_score", ""),
        "header_target_text_ocr": evidence.get("header_target_text", ""),
        "resolved_header_target_text": evidence.get("resolved_header_target_text", ""),
        "header_target_text_score": evidence.get("header_target_text_score", ""),
        "char_body_compatibility": evidence.get("char_body_compatibility", ""),
        "header_icon_count": evidence.get("header_icon_count", ""),
        "target_items": " | ".join(
            str(x.get("matched_label") or x.get("label") or x.get("inferred_label") or "")
            for x in result.get("target_items") or []
        ),
        "task_type_ok": "yes" if expected_task == pred_task else "no",
        "order_mode_ok": "yes" if annotation.get("order_mode") == pred_order_mode else "no",
        "char_sequence_ok": char_ok,
        "final_count_ok": "yes" if len(pred) == len(expected) else "no",
        "elapsed_sec": f"{elapsed:.2f}",
        "error": error,
    }


def run_one(
    python_exe: str,
    image_path: Path,
    run_dir: Path,
    *,
    cpu: bool,
    timeout: int,
    header_intent_model: Path | None,
    header_intent_apply: bool,
    header_intent_threshold: float,
    header_intent_margin: float,
) -> tuple[dict[str, Any] | None, str, float]:
    command = [
        python_exe,
        str(GSXT / "demos" / "dynamic_mixed_infer.py"),
        "--image",
        str(image_path),
        "--output-dir",
        str(run_dir),
        "--threshold",
        "0.3",
        *model_args(),
    ]
    if header_intent_model is not None:
        command.extend(
            [
                "--header-intent-model",
                str(header_intent_model),
                "--header-intent-threshold",
                str(header_intent_threshold),
                "--header-intent-margin",
                str(header_intent_margin),
            ]
        )
        if header_intent_apply:
            command.append("--header-intent-apply")
    if cpu:
        command.append("--cpu")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
    except subprocess.TimeoutExpired as exc:
        return None, f"timeout after {timeout}s: {exc}", time.perf_counter() - start

    if completed.returncode != 0:
        return None, (completed.stderr or completed.stdout or "").strip()[-1200:], elapsed

    result_path = run_dir / "result.json"
    if not result_path.exists():
        return None, "missing result.json", elapsed
    return json.loads(result_path.read_text(encoding="utf-8")), "", elapsed


def load_existing_result(run_dir: Path) -> tuple[dict[str, Any] | None, str, float]:
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return None, "missing result.json", 0.0
    try:
        return json.loads(result_path.read_text(encoding="utf-8")), "", 0.0
    except Exception as exc:
        return None, f"cannot read existing result.json: {exc}", 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(field: str) -> str:
        total = len([r for r in rows if r.get(field) != ""])
        ok = len([r for r in rows if r.get(field) == "yes"])
        return f"{ok}/{total}" if total else "0/0"

    return {
        "rows": len(rows),
        "task_type_ok": rate("task_type_ok"),
        "order_mode_ok": rate("order_mode_ok"),
        "char_sequence_ok": rate("char_sequence_ok"),
        "final_count_ok": rate("final_count_ok"),
        "errors": len([r for r in rows if r.get("error")]),
        "by_task": {
            task: {
                "rows": len(group),
                "task_type_ok": rate_for(group, "task_type_ok"),
                "order_mode_ok": rate_for(group, "order_mode_ok"),
                "char_sequence_ok": rate_for(group, "char_sequence_ok"),
                "final_count_ok": rate_for(group, "final_count_ok"),
            }
            for task, group in group_by(rows, "expected_task_type").items()
        },
    }


def group_by(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get(field) or ""), []).append(row)
    return out


def rate_for(rows: list[dict[str, Any]], field: str) -> str:
    total = len([r for r in rows if r.get(field) != ""])
    ok = len([r for r in rows if r.get(field) == "yes"])
    return f"{ok}/{total}" if total else "0/0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="", help="Comma-separated image names to evaluate.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing runs/<image>/result.json files instead of rerunning them.",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--header-intent-model", type=Path)
    parser.add_argument("--header-intent-apply", action="store_true")
    parser.add_argument("--header-intent-threshold", type=float, default=0.75)
    parser.add_argument("--header-intent-margin", type=float, default=0.25)
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        annotations = [item for item in annotations if item.get("image") in wanted]
    if args.limit:
        annotations = annotations[: args.limit]

    rows: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    iterator = enumerate(annotations, start=1)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(annotations), desc="header eval", unit="img")
    for index, annotation in iterator:
        image = args.image_dir / str(annotation["image"])
        run_dir = args.output_dir / "runs" / image.stem
        if tqdm is None:
            print(f"[{index}/{len(annotations)}] {image.name}", flush=True)
        if args.resume and (run_dir / "result.json").exists():
            result, error, elapsed = load_existing_result(run_dir)
        else:
            result, error, elapsed = run_one(
                args.python,
                image,
                run_dir,
                cpu=args.cpu,
                timeout=args.timeout,
                header_intent_model=args.header_intent_model,
                header_intent_apply=args.header_intent_apply,
                header_intent_threshold=args.header_intent_threshold,
                header_intent_margin=args.header_intent_margin,
            )
        rows.append(make_row(annotation, result, error, elapsed))
        write_csv(args.output_dir / "header_eval.csv", rows)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary(rows), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary(rows), ensure_ascii=False, indent=2))
    print(args.output_dir / "header_eval.csv")


if __name__ == "__main__":
    main()
