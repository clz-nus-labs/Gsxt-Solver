from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = PROJECT_ROOT / "Scripts" / "Gsxt" / "synthetic" / "output_v3"
DEFAULT_OUTPUT = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "synthetic_icon_cls"


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


def split_rows(rows: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio)) if shuffled else 0
    return shuffled[val_count:], shuffled[:val_count]


def crop_with_margin(img: Image.Image, bbox: list[int], margin: int) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img.width, x2 + margin)
    y2 = min(img.height, y2 + margin)
    return img.crop((x1, y1, x2, y2)).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--margin", type=int, default=8)
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(source_dir)
    labels: list[str] = []
    rows: list[str] = []
    counter = 0

    for sample in samples:
        full_image = Image.open(source_dir / "images" / sample["file_name"]).convert("RGB")
        for obj_index, obj in enumerate(sample["objects"]):
            if obj.get("label_type") != "icon":
                continue
            label = obj["label"]
            if label not in labels:
                labels.append(label)
            crop = crop_with_margin(full_image, obj["bbox"], args.margin)
            crop_path = image_dir / f"icon_{counter:06d}_{label}.png"
            crop.save(crop_path)
            rows.append(f"{crop_path.resolve().as_posix()}\t{label}")
            counter += 1

    labels = sorted(labels)
    train_rows, val_rows = split_rows(rows, args.val_ratio, args.seed)

    (output_dir / "label_list.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (output_dir / "train.txt").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    (output_dir / "val.txt").write_text("\n".join(val_rows) + "\n", encoding="utf-8")

    print(f"source={source_dir.resolve()}")
    print(f"output={output_dir.resolve()}")
    print(f"classes={len(labels)}")
    print(f"icon_crops={counter}")
    print(f"train={len(train_rows)}")
    print(f"val={len(val_rows)}")
    print(f"label_list={output_dir / 'label_list.txt'}")


if __name__ == "__main__":
    main()
