from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional.
    tqdm = None


ROOT = Path(__file__).resolve().parents[3]
GSXT = ROOT / "Scripts" / "Gsxt"
DEFAULT_ANNOTATIONS = GSXT / "data" / "annotations" / "gsxt_200_simple_annotation.json"
DEFAULT_RUNS_DIR = GSXT / "output" / "header_eval_200" / "runs"
DEFAULT_OUTPUT_DIR = GSXT / "output" / "header_intent_model_200"


TEXT_TOKENS = [
    "请",
    "按",
    "在",
    "下",
    "图",
    "语",
    "序",
    "顺",
    "词",
    "依",
    "次",
    "点",
    "击",
    "选",
    "择",
    "性",
    "房",
    "决",
    "久",
    "作",
    "人",
    "目",
]

NUMERIC_FEATURES = [
    "instruction_score",
    "header_target_text_score",
    "char_body_compatibility",
    "char_hypothesis_score",
    "icon_hypothesis_score",
    "header_icon_count",
    "prompt_target_overlap",
    "instruction_marker_count",
    "chinese_click_instruction",
    "strong_header_text",
    "icon_prompt_conflict",
    "has_header_target_text",
    "has_resolved_header_target_text",
    "instruction_len",
    "header_target_len",
    "resolved_header_target_len",
    "instruction_cjk_count",
    "header_target_cjk_count",
    "resolved_header_target_cjk_count",
    "raw_item_count",
    "merged_item_count",
    "final_item_count",
    "raw_char_count",
    "raw_icon_count",
    "merged_char_count",
    "merged_icon_count",
    "target_item_count",
    "header_target_in_instruction",
    "resolved_target_in_instruction",
    "ordered_overlap_target",
    "ordered_overlap_resolved",
    "body_phrase_joint",
]


def to_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return 1.0 if value.lower() == "true" else 0.0
        return float(value)
    except (TypeError, ValueError):
        return default


def cjk_text(text: str) -> str:
    return "".join(ch for ch in text or "" if "\u4e00" <= ch <= "\u9fff")


def ordered_overlap_ratio(reference: str, candidate: str) -> float:
    reference = cjk_text(reference)
    candidate = cjk_text(candidate)
    if not candidate:
        return 0.0
    prev = [0] * (len(candidate) + 1)
    for ref_ch in reference:
        cur = prev[:]
        for idx, cand_ch in enumerate(candidate, start=1):
            if ref_ch == cand_ch:
                cur[idx] = max(cur[idx], prev[idx - 1] + 1)
            else:
                cur[idx] = max(cur[idx], cur[idx - 1], prev[idx])
        prev = cur
    return float(prev[-1]) / float(len(candidate))


def load_annotations(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in payload.get("items", []) if item.get("usable", True)]


def load_result(runs_dir: Path, image: str) -> dict[str, Any]:
    result_path = runs_dir / Path(image).stem / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    return json.loads(result_path.read_text(encoding="utf-8"))


def kind_counts(items: list[dict[str, Any]]) -> Counter:
    return Counter(str(item.get("kind") or "") for item in items)


def label_for(annotation: dict[str, Any]) -> str:
    task = str(annotation.get("task_type") or "unknown")
    order = str(annotation.get("order_mode") or "unknown")
    return f"{task}_{order}"


def extract_features(result: dict[str, Any]) -> dict[str, float]:
    task_spec = result.get("task_spec") or {}
    evidence = task_spec.get("evidence") or {}
    instruction = str(evidence.get("instruction_text") or "")
    target = str(evidence.get("header_target_text") or "")
    resolved = str(evidence.get("resolved_header_target_text") or "")
    body_phrase_reason = str(evidence.get("body_phrase_reason") or "")
    raw_items = result.get("raw_items") or []
    merged_items = result.get("merged_items") or []
    target_items = result.get("target_items") or []
    raw_counts = kind_counts(raw_items)
    merged_counts = kind_counts(merged_items)

    features = {
        "instruction_score": to_float(evidence.get("instruction_score")),
        "header_target_text_score": to_float(evidence.get("header_target_text_score")),
        "char_body_compatibility": to_float(evidence.get("char_body_compatibility")),
        "char_hypothesis_score": to_float(evidence.get("char_hypothesis_score")),
        "icon_hypothesis_score": to_float(evidence.get("icon_hypothesis_score")),
        "header_icon_count": to_float(evidence.get("header_icon_count")),
        "prompt_target_overlap": to_float(evidence.get("prompt_target_overlap")),
        "instruction_marker_count": to_float(evidence.get("instruction_marker_count")),
        "chinese_click_instruction": 1.0 if evidence.get("chinese_click_instruction") else 0.0,
        "strong_header_text": 1.0 if evidence.get("strong_header_text") else 0.0,
        "icon_prompt_conflict": 1.0 if evidence.get("icon_prompt_conflict") else 0.0,
        "has_header_target_text": 1.0 if target else 0.0,
        "has_resolved_header_target_text": 1.0 if resolved else 0.0,
        "instruction_len": float(len(instruction)),
        "header_target_len": float(len(target)),
        "resolved_header_target_len": float(len(resolved)),
        "instruction_cjk_count": float(len(cjk_text(instruction))),
        "header_target_cjk_count": float(len(cjk_text(target))),
        "resolved_header_target_cjk_count": float(len(cjk_text(resolved))),
        "raw_item_count": float(len(raw_items)),
        "merged_item_count": float(len(merged_items)),
        "final_item_count": float(len(result.get("items") or [])),
        "raw_char_count": float(raw_counts.get("char", 0)),
        "raw_icon_count": float(raw_counts.get("icon", 0)),
        "merged_char_count": float(merged_counts.get("char", 0)),
        "merged_icon_count": float(merged_counts.get("icon", 0)),
        "target_item_count": float(len(target_items)),
        "header_target_in_instruction": 1.0 if target and target in instruction else 0.0,
        "resolved_target_in_instruction": 1.0 if resolved and resolved in instruction else 0.0,
        "ordered_overlap_target": ordered_overlap_ratio(instruction, target),
        "ordered_overlap_resolved": ordered_overlap_ratio(instruction, resolved),
        "body_phrase_joint": 1.0 if body_phrase_reason.startswith("header-body-joint:") else 0.0,
    }
    for token in TEXT_TOKENS:
        features[f"instruction_has_{token}"] = 1.0 if token in instruction else 0.0
        features[f"target_has_{token}"] = 1.0 if token in target or token in resolved else 0.0
    return features


def build_dataset(
    annotations: list[dict[str, Any]], runs_dir: Path
) -> tuple[list[str], list[dict[str, float]], list[str]]:
    images: list[str] = []
    rows: list[dict[str, float]] = []
    labels: list[str] = []
    iterator = annotations
    for ann in iterator:
        image = str(ann["image"])
        result = load_result(runs_dir, image)
        images.append(image)
        rows.append(extract_features(result))
        labels.append(label_for(ann))
    return images, rows, labels


def feature_names_from(rows: list[dict[str, float]]) -> list[str]:
    names = set(NUMERIC_FEATURES)
    names.update(f"instruction_has_{token}" for token in TEXT_TOKENS)
    names.update(f"target_has_{token}" for token in TEXT_TOKENS)
    for row in rows:
        names.update(row.keys())
    return sorted(names)


def matrix_from(rows: list[dict[str, float]], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(row.get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=np.float64,
    )


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return (x - mean) / std, mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_softmax(
    x: np.ndarray,
    y: np.ndarray,
    *,
    class_count: int,
    epochs: int = 2500,
    lr: float = 0.08,
    l2: float = 0.02,
    patience: int = 120,
    min_delta: float = 1e-6,
    show_progress: bool = False,
    desc: str = "train",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n, dim = x.shape
    w = np.zeros((dim, class_count), dtype=np.float64)
    b = np.zeros(class_count, dtype=np.float64)
    onehot = np.zeros((n, class_count), dtype=np.float64)
    onehot[np.arange(n), y] = 1.0
    class_counts = np.bincount(y, minlength=class_count).astype(np.float64)
    sample_weights = (n / np.maximum(class_counts[y], 1.0)) / class_count
    sample_weights = sample_weights / sample_weights.mean()

    best_loss = float("inf")
    best_epoch = 0
    best_w = w.copy()
    best_b = b.copy()
    wait = 0
    history: list[float] = []

    iterator = range(epochs)
    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(iterator, desc=desc, unit="epoch")
        iterator = progress

    for epoch in iterator:
        logits = x @ w + b
        probs = softmax(logits)
        loss_vec = -np.sum(onehot * np.log(np.maximum(probs, 1e-12)), axis=1)
        loss = float(np.mean(loss_vec * sample_weights) + 0.5 * l2 * np.sum(w * w))
        history.append(loss)

        if loss < best_loss - min_delta:
            best_loss = loss
            best_epoch = epoch + 1
            best_w = w.copy()
            best_b = b.copy()
            wait = 0
        else:
            wait += 1

        if progress is not None and (epoch == 0 or (epoch + 1) % 50 == 0):
            progress.set_postfix(loss=f"{loss:.5f}", best=f"{best_loss:.5f}", wait=wait)
        if patience > 0 and wait >= patience:
            break

        grad = (probs - onehot) * sample_weights[:, None]
        grad_w = (x.T @ grad) / n + l2 * w
        grad_b = grad.mean(axis=0)
        step = lr / math.sqrt(1.0 + epoch / 600.0)
        w -= step * grad_w
        b -= step * grad_b

    return best_w, best_b, {
        "epochs_requested": epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "stopped_early": len(history) < epochs,
        "patience": patience,
        "min_delta": min_delta,
        "loss_tail": history[-10:],
    }


def predict(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = softmax(x @ w + b)
    return probs.argmax(axis=1), probs


def accuracy(labels: list[str], preds: list[str]) -> float:
    return sum(a == b for a, b in zip(labels, preds)) / len(labels) if labels else 0.0


def split_task(label: str) -> tuple[str, str]:
    match = re.match(r"^(.+)_(given_order|semantic_order)$", label)
    if not match:
        return label, ""
    return match.group(1), match.group(2)


def evaluate_leave_one_out(
    x: np.ndarray,
    y: list[str],
    class_names: list[str],
    *,
    epochs: int,
    lr: float,
    l2: float,
    patience: int,
    min_delta: float,
    show_progress: bool,
) -> list[dict[str, Any]]:
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y_idx_all = np.asarray([label_to_idx[label] for label in y], dtype=np.int64)
    records: list[dict[str, Any]] = []
    iterator = range(len(y))
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc="leave-one-out", unit="img")
    for holdout in iterator:
        train_idx = np.asarray([idx for idx in range(len(y)) if idx != holdout], dtype=np.int64)
        test_idx = np.asarray([holdout], dtype=np.int64)
        x_train, mean, std = standardize_fit(x[train_idx])
        x_test = standardize_apply(x[test_idx], mean, std)
        w, b, train_info = train_softmax(
            x_train,
            y_idx_all[train_idx],
            class_count=len(class_names),
            epochs=epochs,
            lr=lr,
            l2=l2,
            patience=patience,
            min_delta=min_delta,
            show_progress=False,
        )
        pred_idx, probs = predict(x_test, w, b)
        pred = class_names[int(pred_idx[0])]
        records.append(
            {
                "index": holdout,
                "actual": y[holdout],
                "predicted": pred,
                "confidence": float(probs[0, int(pred_idx[0])]),
                "epochs_run": train_info["epochs_run"],
                "best_epoch": train_info["best_epoch"],
            }
        )
    return records


def confusion(labels: list[str], preds: list[str], class_names: list[str]) -> dict[str, dict[str, int]]:
    table = {actual: {pred: 0 for pred in class_names} for actual in class_names}
    for actual, pred in zip(labels, preds):
        table.setdefault(actual, {name: 0 for name in class_names})
        table[actual][pred] = table[actual].get(pred, 0) + 1
    return table


def write_feature_csv(
    path: Path,
    images: list[str],
    rows: list[dict[str, float]],
    labels: list[str],
    feature_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", *feature_names])
        for image, label, row in zip(images, labels, rows):
            writer.writerow([image, label, *[row.get(name, 0.0) for name in feature_names]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--loo-epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.02)
    parser.add_argument("--skip-loo", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    images, feature_rows, labels = build_dataset(annotations, args.runs_dir)
    feature_names = feature_names_from(feature_rows)
    class_names = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    x = matrix_from(feature_rows, feature_names)
    y_idx = np.asarray([label_to_idx[label] for label in labels], dtype=np.int64)

    show_progress = not args.no_tqdm
    loo: list[dict[str, Any]] = []
    if not args.skip_loo:
        loo = evaluate_leave_one_out(
            x,
            labels,
            class_names,
            epochs=args.loo_epochs,
            lr=args.lr,
            l2=args.l2,
            patience=args.patience,
            min_delta=args.min_delta,
            show_progress=show_progress,
        )

    loo_preds = [str(row["predicted"]) for row in loo]
    task_labels = [split_task(label)[0] for label in labels]
    order_labels = [split_task(label)[1] for label in labels]
    task_preds = [split_task(label)[0] for label in loo_preds]
    order_preds = [split_task(label)[1] for label in loo_preds]

    x_scaled, mean, std = standardize_fit(x)
    w, b, train_info = train_softmax(
        x_scaled,
        y_idx,
        class_count=len(class_names),
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        patience=args.patience,
        min_delta=args.min_delta,
        show_progress=show_progress,
        desc="final-train",
    )
    train_pred_idx, _train_probs = predict(x_scaled, w, b)
    train_preds = [class_names[int(idx)] for idx in train_pred_idx]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_feature_csv(args.output_dir / "features.csv", images, feature_rows, labels, feature_names)

    model = {
        "schema_version": "header-intent-softmax-v1",
        "classes": class_names,
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": w.tolist(),
        "bias": b.tolist(),
        "text_tokens": TEXT_TOKENS,
        "training": {
            "samples": len(labels),
            "label_counts": dict(Counter(labels)),
            "epochs": args.epochs,
            "loo_epochs": args.loo_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "lr": args.lr,
            "l2": args.l2,
            "final_train_info": train_info,
        },
    }
    (args.output_dir / "header_intent_model.json").write_text(
        json.dumps(model, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "samples": len(labels),
        "label_counts": dict(Counter(labels)),
        "train_accuracy": accuracy(labels, train_preds),
        "train_confusion": confusion(labels, train_preds, class_names),
        "final_train_info": train_info,
    }
    if loo:
        report.update(
            {
                "leave_one_out_accuracy": accuracy(labels, loo_preds),
                "leave_one_out_task_accuracy": accuracy(task_labels, task_preds),
                "leave_one_out_order_accuracy": accuracy(order_labels, order_preds),
                "leave_one_out_confusion": confusion(labels, loo_preds, class_names),
                "leave_one_out_errors": [
                    {
                        "image": images[row["index"]],
                        "actual": row["actual"],
                        "predicted": row["predicted"],
                        "confidence": row["confidence"],
                    }
                    for row in loo
                    if row["actual"] != row["predicted"]
                ],
                "leave_one_out_epochs": {
                    "mean_epochs_run": float(np.mean([row["epochs_run"] for row in loo])),
                    "max_epochs_run": int(max(row["epochs_run"] for row in loo)),
                    "min_epochs_run": int(min(row["epochs_run"] for row in loo)),
                },
            }
        )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(args.output_dir / "header_intent_model.json")


if __name__ == "__main__":
    main()
