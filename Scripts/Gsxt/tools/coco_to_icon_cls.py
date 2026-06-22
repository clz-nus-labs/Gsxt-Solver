from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ICON_ROOT = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "Geetest_Solver_v12i_coco"
DEFAULT_OUTPUT = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "geetest_icon_cls"
DEFAULT_SYN_ICON = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "synthetic_icon_cls"


SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("valid", "val"),
    "test": ("test",),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or "unknown"


def crop_bbox(image_path: Path, bbox: list[float], output_path: Path, pad_ratio: float) -> bool:
    with Image.open(image_path).convert("RGB") as img:
        x, y, w, h = bbox
        if w <= 1 or h <= 1:
            return False
        pad = max(w, h) * pad_ratio
        left = max(0, int(x - pad))
        top = max(0, int(y - pad))
        right = min(img.width, int(x + w + pad))
        bottom = min(img.height, int(y + h + pad))
        if right <= left or bottom <= top:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.crop((left, top, right, bottom)).save(output_path, quality=95)
        return True


def add_external_split(
    root: Path,
    output: Path,
    split: str,
    rows: list[str],
    labels: set[str],
    pad_ratio: float,
) -> tuple[int, int]:
    split_dir = find_split_dir(root, split)
    if split_dir is None:
        return 0, 0
    coco = read_json(find_annotation(split_dir))
    categories = {int(item["id"]): item["name"] for item in coco.get("categories", [])}
    images = {int(item["id"]): item for item in coco.get("images", [])}
    counts: dict[str, int] = defaultdict(int)

    total = 0
    skipped = 0
    for ann in coco.get("annotations", []):
        category_id = int(ann.get("category_id", 0))
        if category_id == 0:
            continue
        image = images.get(int(ann["image_id"]))
        if not image:
            skipped += 1
            continue
        source_path = split_dir / Path(image["file_name"]).name
        if not source_path.exists():
            skipped += 1
            continue
        label = categories.get(category_id, f"class_{category_id}")
        labels.add(label)
        label_dir = safe_label(label)
        counts[label] += 1
        crop_name = f"{split}_{label_dir}_{counts[label]:06d}.jpg"
        crop_path = output / "images" / split / label_dir / crop_name
        if not crop_bbox(source_path, [float(v) for v in ann["bbox"]], crop_path, pad_ratio):
            skipped += 1
            continue
        rows.append(f"{crop_path.resolve()}\t{label}")
        total += 1
    return total, skipped


def add_synthetic_rows(source: Path, split: str, rows: list[str], labels: set[str]) -> int:
    list_name = "val.txt" if split == "val" else f"{split}.txt"
    list_path = source / list_name
    if not list_path.exists():
        return 0
    count = 0
    for line in list_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_path, label = line.split("\t", 1)
        if Path(image_path).exists():
            labels.add(label)
            rows.append(f"{Path(image_path).resolve()}\t{label}")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop COCO icon annotations into the custom icon classification format.")
    parser.add_argument("--icon-root", default=str(DEFAULT_ICON_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--synthetic-icon", default=str(DEFAULT_SYN_ICON))
    parser.add_argument("--pad-ratio", type=float, default=0.15)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    labels: set[str] = set()
    split_rows: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for split in ("train", "val", "test"):
        total, skipped = add_external_split(Path(args.icon_root), output, split, split_rows[split], labels, args.pad_ratio)
        synthetic_count = 0
        if args.include_synthetic and split in ("train", "val"):
            synthetic_count = add_synthetic_rows(Path(args.synthetic_icon), split, split_rows[split], labels)
        print(f"{split}: external={total} skipped={skipped} synthetic={synthetic_count}")

    label_list = sorted(labels)
    (output / "label_list.txt").write_text("\n".join(label_list) + "\n", encoding="utf-8")
    for split, rows in split_rows.items():
        if rows:
            (output / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"labels={len(label_list)}")
    print(f"Saved icon classification dataset to: {output}")


if __name__ == "__main__":
    main()
