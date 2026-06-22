from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHAR_ROOT = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "Chinese_char_1_v1i_coco"
DEFAULT_ICON_ROOT = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "Geetest_Solver_v12i_coco"
DEFAULT_OUTPUT = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "external_mixed_paddledet"
DEFAULT_SYN_DET = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "synthetic_mixed_paddledet"
DEFAULT_SYN_IMAGES = PROJECT_ROOT / "Scripts" / "Gsxt" / "synthetic" / "output_v4" / "images"


SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("valid", "val"),
    "test": ("test",),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def find_split_dir(root: Path, split: str) -> Path | None:
    for name in SPLIT_ALIASES[split]:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def find_annotation(split_dir: Path) -> Path:
    candidates = sorted(split_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No COCO json found under: {split_dir}")
    for path in candidates:
        if "annotation" in path.name.lower() or "coco" in path.name.lower():
            return path
    return candidates[0]


def add_coco_root(
    out: dict,
    out_images_dir: Path,
    root: Path,
    split: str,
    role: str,
    next_image_id: int,
    next_ann_id: int,
) -> tuple[int, int, int, int]:
    split_dir = find_split_dir(root, split)
    if split_dir is None:
        return next_image_id, next_ann_id, 0, 0

    coco = read_json(find_annotation(split_dir))
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("category_id", 0) == 0:
            continue
        anns_by_image[int(ann["image_id"])].append(ann)

    cat_id = 1 if role == "char" else 2
    copied_images = 0
    copied_anns = 0
    for image in coco.get("images", []):
        old_image_id = int(image["id"])
        image_anns = anns_by_image.get(old_image_id, [])
        if not image_anns:
            continue

        source_name = Path(image["file_name"]).name
        source_path = split_dir / source_name
        if not source_path.exists():
            continue

        new_name = f"{root.name}_{split}_{next_image_id:08d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, out_images_dir / new_name)
        out["images"].append(
            {
                "id": next_image_id,
                "file_name": new_name,
                "width": int(image["width"]),
                "height": int(image["height"]),
            }
        )
        for ann in image_anns:
            bbox = [float(v) for v in ann["bbox"]]
            if bbox[2] <= 1 or bbox[3] <= 1:
                continue
            out["annotations"].append(
                {
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": float(ann.get("area", bbox[2] * bbox[3])),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    "segmentation": ann.get("segmentation", []),
                }
            )
            next_ann_id += 1
            copied_anns += 1
        next_image_id += 1
        copied_images += 1

    return next_image_id, next_ann_id, copied_images, copied_anns


def add_synthetic(
    out: dict,
    out_images_dir: Path,
    dataset_dir: Path,
    image_dir: Path,
    split: str,
    next_image_id: int,
    next_ann_id: int,
) -> tuple[int, int, int, int]:
    json_path = dataset_dir / f"{split}.json"
    if not json_path.exists():
        return next_image_id, next_ann_id, 0, 0
    coco = read_json(json_path)
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)

    copied_images = 0
    copied_anns = 0
    for image in coco.get("images", []):
        old_image_id = int(image["id"])
        source_name = Path(image["file_name"]).name
        source_path = image_dir / source_name
        if not source_path.exists():
            continue
        new_name = f"synthetic_{split}_{next_image_id:08d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, out_images_dir / new_name)
        out["images"].append(
            {
                "id": next_image_id,
                "file_name": new_name,
                "width": int(image["width"]),
                "height": int(image["height"]),
            }
        )
        for ann in anns_by_image.get(old_image_id, []):
            bbox = [float(v) for v in ann["bbox"]]
            if bbox[2] <= 1 or bbox[3] <= 1:
                continue
            out["annotations"].append(
                {
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": int(ann["category_id"]),
                    "bbox": bbox,
                    "area": float(ann.get("area", bbox[2] * bbox[3])),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    "segmentation": ann.get("segmentation", []),
                }
            )
            next_ann_id += 1
            copied_anns += 1
        next_image_id += 1
        copied_images += 1
    return next_image_id, next_ann_id, copied_images, copied_anns


def build_split(args: argparse.Namespace, split: str) -> dict:
    out = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "char", "supercategory": "target"},
            {"id": 2, "name": "icon", "supercategory": "target"},
        ],
    }
    images_dir = Path(args.output) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    next_image_id = 1
    next_ann_id = 1
    stats: list[str] = []

    if args.include_synthetic:
        next_image_id, next_ann_id, image_count, ann_count = add_synthetic(
            out,
            images_dir,
            Path(args.synthetic_det),
            Path(args.synthetic_images),
            split,
            next_image_id,
            next_ann_id,
        )
        stats.append(f"synthetic={image_count}/{ann_count}")

    for root_text, role in ((args.char_root, "char"), (args.icon_root, "icon")):
        next_image_id, next_ann_id, image_count, ann_count = add_coco_root(
            out,
            images_dir,
            Path(root_text),
            split,
            role,
            next_image_id,
            next_ann_id,
        )
        stats.append(f"{role}={image_count}/{ann_count}")

    print(f"{split}: images={len(out['images'])} annotations={len(out['annotations'])} {' '.join(stats)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO char/icon datasets to a 2-class PaddleDetection COCO dataset.")
    parser.add_argument("--char-root", default=str(DEFAULT_CHAR_ROOT))
    parser.add_argument("--icon-root", default=str(DEFAULT_ICON_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--synthetic-det", default=str(DEFAULT_SYN_DET))
    parser.add_argument("--synthetic-images", default=str(DEFAULT_SYN_IMAGES))
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        payload = build_split(args, split)
        if payload["images"]:
            write_json(output / f"{split}.json", payload)

    print(f"Saved PaddleDetection dataset to: {output}")


if __name__ == "__main__":
    main()
