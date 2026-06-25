from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional.
    tqdm = None

from predict_header_intent import load_model, predict_one


GSXT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_CSV = GSXT / "output" / "header_eval_200" / "header_eval.csv"
DEFAULT_RUNS_DIR = GSXT / "output" / "header_eval_200" / "runs"
DEFAULT_MODEL = GSXT / "output" / "header_intent_model_200" / "header_intent_model.json"
DEFAULT_OUTPUT = GSXT / "output" / "header_intent_model_200" / "arbitration_report.json"


def read_eval_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def result_path_for(runs_dir: Path, image: str) -> Path:
    return runs_dir / Path(image).stem / "result.json"


def label_from_parts(task_type: str, order_mode: str) -> str:
    return f"{task_type}_{order_mode}"


def split_label(label: str) -> tuple[str, str]:
    if label.endswith("_given_order"):
        return label[: -len("_given_order")], "given_order"
    if label.endswith("_semantic_order"):
        return label[: -len("_semantic_order")], "semantic_order"
    return label, "unknown"


def intent_ok(task: str, order: str, expected_task: str, expected_order: str) -> bool:
    return task == expected_task and order == expected_order


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def acc(field: str) -> str:
        ok = sum(1 for row in rows if row[field])
        return f"{ok}/{len(rows)}"

    by_expected: dict[str, dict[str, int]] = {}
    for row in rows:
        key = row["expected_label"]
        bucket = by_expected.setdefault(
            key,
            {
                "count": 0,
                "rule_ok": 0,
                "model_ok": 0,
                "hybrid_ok": 0,
            },
        )
        bucket["count"] += 1
        bucket["rule_ok"] += int(row["rule_ok"])
        bucket["model_ok"] += int(row["model_ok"])
        bucket["hybrid_ok"] += int(row["hybrid_ok"])

    return {
        "rows": len(rows),
        "rule_intent_accuracy": acc("rule_ok"),
        "model_intent_accuracy": acc("model_ok"),
        "hybrid_intent_accuracy": acc("hybrid_ok"),
        "overrides": sum(1 for row in rows if row["hybrid_overrode_rule"]),
        "helpful_overrides": sum(
            1
            for row in rows
            if row["hybrid_overrode_rule"] and not row["rule_ok"] and row["hybrid_ok"]
        ),
        "harmful_overrides": sum(
            1
            for row in rows
            if row["hybrid_overrode_rule"] and row["rule_ok"] and not row["hybrid_ok"]
        ),
        "by_expected_label": by_expected,
        "rule_errors": [
            row
            for row in rows
            if not row["rule_ok"]
        ],
        "model_errors": [
            row
            for row in rows
            if not row["model_ok"]
        ],
        "hybrid_errors": [
            row
            for row in rows
            if not row["hybrid_ok"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()

    model = load_model(args.model)
    eval_rows = read_eval_rows(args.eval_csv)
    iterator = eval_rows
    if tqdm is not None and not args.no_tqdm:
        iterator = tqdm(eval_rows, desc="intent arbitration", unit="img")

    output_rows: list[dict[str, Any]] = []
    for row in iterator:
        image = row["image"]
        expected_task = row["expected_task_type"]
        expected_order = row["expected_order_mode"]
        rule_task = row["pred_task_type"]
        rule_order = row["pred_order_mode"]
        pred = predict_one(model, result_path_for(args.runs_dir, image))
        probs = pred["probabilities"]
        sorted_probs = sorted(probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else sorted_probs[0]

        should_override = (
            pred["confidence"] >= args.threshold
            and margin >= args.margin
            and (pred["task_type"] != rule_task or pred["order_mode"] != rule_order)
        )
        hybrid_task = pred["task_type"] if should_override else rule_task
        hybrid_order = pred["order_mode"] if should_override else rule_order

        output_rows.append(
            {
                "image": image,
                "expected_label": label_from_parts(expected_task, expected_order),
                "rule_label": label_from_parts(rule_task, rule_order),
                "model_label": pred["label"],
                "hybrid_label": label_from_parts(hybrid_task, hybrid_order),
                "model_confidence": pred["confidence"],
                "model_margin": margin,
                "rule_ok": intent_ok(rule_task, rule_order, expected_task, expected_order),
                "model_ok": intent_ok(pred["task_type"], pred["order_mode"], expected_task, expected_order),
                "hybrid_ok": intent_ok(hybrid_task, hybrid_order, expected_task, expected_order),
                "hybrid_overrode_rule": should_override,
                "expected_sequence": row.get("expected_sequence", ""),
                "rule_sequence": row.get("pred_sequence", ""),
                "target_source_resolved": row.get("target_source_resolved", ""),
                "header_instruction_ocr": row.get("header_instruction_ocr", ""),
                "header_target_text_ocr": row.get("header_target_text_ocr", ""),
                "header_icon_count": row.get("header_icon_count", ""),
            }
        )

    report = {
        "threshold": args.threshold,
        "margin": args.margin,
        **summarize(output_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    printable = {
        key: value
        for key, value in report.items()
        if key not in {"rule_errors", "model_errors", "hybrid_errors"}
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(args.output)
    print(csv_path)


if __name__ == "__main__":
    main()
