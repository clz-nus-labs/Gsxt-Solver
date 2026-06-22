from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = PROJECT_ROOT / "Scripts" / "Gsxt" / "synthetic" / "output_v4"
DEFAULT_OUTPUT = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "synthetic_mixed_paddledet"

CATEGORIES = [
    {"id": 1, "name": "char", "supercategory": "target"},
    {"id": 2, "name": "icon", "supercategory": "target"},
]


def load_samples(source_dir: Path) -> list[dict]:
    samples_path = source_dir / "samples.json"
    if samples_path.exists():
        return json.loads(samples_path.read_text(encoding="utf-8"))
    jsonl_path = source_dir / "samples.jsonl"
    if jsonl_path.exists():
        return [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raise FileNotFoundError(f"Missing samples.json or samples.jsonl in {source_dir}")


def build_coco(samples: list[dict]) -> dict:
    images = []
    annotations = []
    ann_id = 1
    for sample in samples:
        images.append(
            {
                "id": sample["id"],
                "file_name": sample["file_name"],
                "width": sample["width"],
                "height": sample["height"],
            }
        )
        for obj in sample["objects"]:
            x1, y1, x2, y2 = obj["bbox"]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": sample["id"],
                    "category_id": 1 if obj["label_type"] == "char" else 2,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "label": obj["label"],
                    "label_type": obj["label_type"],
                }
            )
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": CATEGORIES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(source_dir)
    random.Random(args.seed).shuffle(samples)
    val_count = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    (output_dir / "train.json").write_text(
        json.dumps(build_coco(train_samples), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "val.json").write_text(
        json.dumps(build_coco(val_samples), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "source": str(source_dir.resolve()),
                "image_dir": str((source_dir / "images").resolve()),
                "train_count": len(train_samples),
                "val_count": len(val_samples),
                "categories": CATEGORIES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"source={source_dir.resolve()}")
    print(f"output={output_dir.resolve()}")
    print(f"image_dir={(source_dir / 'images').resolve()}")
    print(f"train={len(train_samples)}")
    print(f"val={len(val_samples)}")


if __name__ == "__main__":
    main()
