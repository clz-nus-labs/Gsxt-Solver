from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_header_intent_model import extract_features, matrix_from, softmax


DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "header_intent_model_200"
    / "header_intent_model.json"
)


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_one(model: dict, result_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    features = extract_features(result)
    feature_names = list(model["feature_names"])
    x = matrix_from([features], feature_names)
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = np.asarray(model["bias"], dtype=np.float64)
    probs = softmax(((x - mean) / std) @ weights + bias)[0]
    classes = list(model["classes"])
    order = np.argsort(-probs)
    best = int(order[0])
    label = classes[best]
    if label.endswith("_given_order"):
        task_type = label[: -len("_given_order")]
        order_mode = "given_order"
    elif label.endswith("_semantic_order"):
        task_type = label[: -len("_semantic_order")]
        order_mode = "semantic_order"
    else:
        task_type = label
        order_mode = "unknown"
    return {
        "result": str(result_path),
        "label": label,
        "task_type": task_type,
        "order_mode": order_mode,
        "confidence": float(probs[best]),
        "probabilities": {classes[idx]: float(probs[idx]) for idx in order},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    model = load_model(args.model)
    payload = [predict_one(model, path) for path in args.result_json]
    print(json.dumps(payload[0] if len(payload) == 1 else payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
