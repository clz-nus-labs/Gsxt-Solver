from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "Chinese_char_1_v1i_coco"
DEFAULT_OUTPUT = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "Chinese_char_1_labeling"

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


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def crop_with_padding(image_path: Path, bbox: list[float], pad_ratio: float) -> Image.Image | None:
    with Image.open(image_path).convert("RGB") as img:
        x, y, w, h = bbox
        if w <= 1 or h <= 1:
            return None
        pad = max(w, h) * pad_ratio
        left = max(0, int(x - pad))
        top = max(0, int(y - pad))
        right = min(img.width, int(x + w + pad))
        bottom = min(img.height, int(y + h + pad))
        if right <= left or bottom <= top:
            return None
        return img.crop((left, top, right, bottom))


def collect_samples(root: Path, max_samples: int, pad_ratio: float, output: Path) -> dict[int, dict]:
    by_category: dict[int, dict] = {}
    counts: dict[int, int] = defaultdict(int)

    for split in ("train", "val", "test"):
        split_dir = find_split_dir(root, split)
        if split_dir is None:
            continue
        coco = read_json(find_annotation(split_dir))
        categories = {int(item["id"]): item["name"] for item in coco.get("categories", [])}
        images = {int(item["id"]): item for item in coco.get("images", [])}
        for category_id, encoded_name in categories.items():
            if category_id == 0:
                continue
            by_category.setdefault(
                category_id,
                {
                    "category_id": category_id,
                    "encoded_name": encoded_name,
                    "sample_count": 0,
                    "crops": [],
                },
            )

        for ann in coco.get("annotations", []):
            category_id = int(ann.get("category_id", 0))
            if category_id == 0 or counts[category_id] >= max_samples:
                continue
            image = images.get(int(ann["image_id"]))
            if image is None:
                continue
            image_path = split_dir / Path(image["file_name"]).name
            if not image_path.exists():
                continue
            crop = crop_with_padding(image_path, [float(v) for v in ann["bbox"]], pad_ratio)
            if crop is None:
                continue
            encoded_name = by_category[category_id]["encoded_name"]
            crop_dir = output / "crops" / f"{category_id:04d}_{encoded_name}"
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / f"{counts[category_id]:03d}_{split}.png"
            crop.save(crop_path)
            by_category[category_id]["crops"].append(crop_path)
            by_category[category_id]["sample_count"] += 1
            counts[category_id] += 1

    return by_category


def make_sheet(category: dict, output: Path, thumb_size: int, columns: int) -> Path | None:
    crops = [Path(p) for p in category["crops"]]
    if not crops:
        return None
    title_h = 44
    rows = (len(crops) + columns - 1) // columns
    width = columns * thumb_size
    height = title_h + rows * thumb_size
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 8),
        f"id={category['category_id']} code={category['encoded_name']} samples={category['sample_count']}",
        fill=(0, 0, 0),
        font=font(18),
    )
    for index, crop_path in enumerate(crops):
        crop = Image.open(crop_path).convert("RGB")
        crop.thumbnail((thumb_size - 12, thumb_size - 22), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_size
        y = title_h + (index // columns) * thumb_size
        draw.rectangle((x, y, x + thumb_size - 1, y + thumb_size - 1), outline=(210, 210, 210))
        paste_x = x + (thumb_size - crop.width) // 2
        paste_y = y + 6 + (thumb_size - 22 - crop.height) // 2
        sheet.paste(crop, (paste_x, paste_y))
        draw.text((x + 4, y + thumb_size - 16), str(index), fill=(80, 80, 80), font=font(12))

    sheet_dir = output / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = sheet_dir / f"{category['category_id']:04d}_{category['encoded_name']}.png"
    sheet.save(sheet_path)
    return sheet_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate contact sheets and a CSV template for labeling encoded Chinese char COCO categories.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--pad-ratio", type=float, default=0.12)
    parser.add_argument("--thumb-size", type=int, default=96)
    parser.add_argument("--columns", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    categories = collect_samples(root, args.max_samples, args.pad_ratio, output)
    csv_path = output / "char_label_template.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category_id", "encoded_name", "label", "sample_count", "sheet_path", "crop_dir"],
        )
        writer.writeheader()
        for category_id in sorted(categories):
            item = categories[category_id]
            sheet_path = make_sheet(item, output, args.thumb_size, args.columns)
            crop_dir = output / "crops" / f"{category_id:04d}_{item['encoded_name']}"
            writer.writerow(
                {
                    "category_id": category_id,
                    "encoded_name": item["encoded_name"],
                    "label": "",
                    "sample_count": item["sample_count"],
                    "sheet_path": str(sheet_path.resolve()) if sheet_path else "",
                    "crop_dir": str(crop_dir.resolve()),
                }
            )

    print(f"categories={len(categories)}")
    print(f"CSV template: {csv_path}")
    print(f"Contact sheets: {output / 'sheets'}")


if __name__ == "__main__":
    main()
