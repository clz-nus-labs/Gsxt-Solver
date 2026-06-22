from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = PROJECT_ROOT / "Scripts" / "Gsxt" / "synthetic" / "output_v3"
DEFAULT_OUTPUT = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "synthetic_mixed_paddleocr"


def polygon_from_bbox(bbox: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def split_rows(rows: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio)) if shuffled else 0
    return shuffled[val_count:], shuffled[:val_count]


def load_samples(source_dir: Path) -> list[dict]:
    meta_path = source_dir / "samples.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    coco_path = source_dir / "labels_coco.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"Missing labels_coco.json: {coco_path}")

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images_by_id = {item["id"]: item for item in coco["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    samples: list[dict] = []
    for image_id, image in images_by_id.items():
        objects = []
        for ann in anns_by_image.get(image_id, []):
            x, y, w, h = ann["bbox"]
            objects.append(
                {
                    "label_type": ann.get("label_type", "char"),
                    "label": ann.get("label", ""),
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                }
            )
        samples.append(
            {
                "id": image_id,
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
                "objects": objects,
            }
        )
    return samples


def write_detection_labels(source_dir: Path, output_dir: Path, val_ratio: float, seed: int) -> None:
    det_dir = output_dir / "det"
    det_dir.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for sample in load_samples(source_dir):
        image_path = (source_dir / "images" / sample["file_name"]).resolve()
        items = []
        for obj in sample["objects"]:
            items.append(
                {
                    "transcription": obj["label"],
                    "points": polygon_from_bbox(obj["bbox"]),
                    "label_type": obj["label_type"],
                }
            )
        rows.append(f"{image_path.as_posix()}\t{json.dumps(items, ensure_ascii=False)}")

    train_rows, val_rows = split_rows(rows, val_ratio, seed)
    (det_dir / "train_det.txt").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    (det_dir / "val_det.txt").write_text("\n".join(val_rows) + "\n", encoding="utf-8")


def write_recognition_labels(source_dir: Path, output_dir: Path, val_ratio: float, seed: int) -> None:
    rec_dir = output_dir / "rec"
    rec_dir.mkdir(parents=True, exist_ok=True)

    label_path = source_dir / "char_rec_labels.txt"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing char_rec_labels.txt: {label_path}")

    rows = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows, val_rows = split_rows(rows, val_ratio, seed)
    (rec_dir / "train_rec.txt").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    (rec_dir / "val_rec.txt").write_text("\n".join(val_rows) + "\n", encoding="utf-8")


def write_icon_labels(source_dir: Path, output_dir: Path) -> None:
    icon_dir = output_dir / "icon"
    icon_dir.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for sample in load_samples(source_dir):
        image_path = (source_dir / "images" / sample["file_name"]).resolve()
        for index, obj in enumerate(sample["objects"]):
            if obj["label_type"] == "icon":
                rows.append(f"{image_path.as_posix()}\t{obj['label']}\t{json.dumps(obj['bbox'], ensure_ascii=False)}")
    (icon_dir / "icon_boxes.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260604)
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_detection_labels(source_dir, output_dir, args.val_ratio, args.seed)
    write_recognition_labels(source_dir, output_dir, args.val_ratio, args.seed)
    write_icon_labels(source_dir, output_dir)

    print(f"source={source_dir.resolve()}")
    print(f"output={output_dir.resolve()}")
    print(f"det_train={output_dir / 'det' / 'train_det.txt'}")
    print(f"det_val={output_dir / 'det' / 'val_det.txt'}")
    print(f"rec_train={output_dir / 'rec' / 'train_rec.txt'}")
    print(f"rec_val={output_dir / 'rec' / 'val_rec.txt'}")
    print(f"icon_boxes={output_dir / 'icon' / 'icon_boxes.txt'}")


if __name__ == "__main__":
    main()
