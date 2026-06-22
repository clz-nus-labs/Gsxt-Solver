from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
import re
from xml.etree import ElementTree
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "Chinese_char_1_v1i_coco"
DEFAULT_LABELS = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "Chinese_char_1_labeling" / "char_label_template.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "Scripts" / "Gsxt" / "data" / "datasets" / "chinese_char_labeled_paddleocr"

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


def load_csv_labels(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip()
            if label:
                mapping[int(row["category_id"])] = label
    return mapping


def load_xlsx_labels(path: Path) -> dict[int, str]:
    # A tiny XLSX reader is enough for this label template and avoids an extra openpyxl dependency.
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", ns):
                parts = [node.text or "" for node in item.findall(".//main:t", ns)]
                shared_strings.append("".join(parts))

        workbook = ElementTree.fromstring(zf.read("xl/workbook.xml"))
        first_sheet = workbook.find("main:sheets/main:sheet", ns)
        if first_sheet is None:
            raise ValueError("XLSX has no worksheet.")
        rel_id = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        rels = ElementTree.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall("rel:Relationship", ns):
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib["Target"]
                break
        if target is None:
            raise ValueError("Cannot resolve first worksheet path.")
        sheet_path = "xl/" + target.lstrip("/")
        sheet_root = ElementTree.fromstring(zf.read(sheet_path))

    def cell_value(cell: ElementTree.Element) -> str:
        value_node = cell.find("main:v", ns)
        inline_node = cell.find("main:is/main:t", ns)
        if inline_node is not None:
            return inline_node.text or ""
        if value_node is None:
            return ""
        raw = value_node.text or ""
        if cell.attrib.get("t") == "s":
            return shared_strings[int(raw)] if raw else ""
        return raw

    def column_index(cell_ref: str) -> int:
        letters = re.match(r"[A-Z]+", cell_ref)
        if not letters:
            return 0
        value = 0
        for ch in letters.group(0):
            value = value * 26 + (ord(ch) - ord("A") + 1)
        return value - 1

    table: list[list[str]] = []
    for row in sheet_root.findall(".//main:sheetData/main:row", ns):
        cells: dict[int, str] = {}
        max_index = -1
        for cell in row.findall("main:c", ns):
            index = column_index(cell.attrib.get("r", "A1"))
            cells[index] = cell_value(cell)
            max_index = max(max_index, index)
        values = [cells.get(index, "") for index in range(max_index + 1)]
        table.append(values)
    if not table:
        return {}

    headers = [str(value).strip() if value is not None else "" for value in table[0]]
    header_to_index = {name: index for index, name in enumerate(headers)}
    if "category_id" not in header_to_index or "label" not in header_to_index:
        raise ValueError("XLSX must contain category_id and label columns.")

    mapping: dict[int, str] = {}
    for row in table[1:]:
        category_index = header_to_index["category_id"]
        label_index = header_to_index["label"]
        if category_index >= len(row) or label_index >= len(row):
            continue
        category_raw = row[category_index]
        label_raw = row[label_index]
        label = str(label_raw).strip()
        if not label:
            continue
        mapping[int(float(category_raw))] = label
    return mapping


def load_labels(path: Path) -> dict[int, str]:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return load_xlsx_labels(path)
    return load_csv_labels(path)


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


def convert_split(root: Path, output: Path, split: str, labels: dict[int, str], pad_ratio: float) -> tuple[int, int]:
    split_dir = find_split_dir(root, split)
    if split_dir is None:
        return 0, 0
    coco = read_json(find_annotation(split_dir))
    images = {int(item["id"]): item for item in coco.get("images", [])}
    rows: list[str] = []
    skipped = 0
    for index, ann in enumerate(coco.get("annotations", [])):
        category_id = int(ann.get("category_id", 0))
        label = labels.get(category_id)
        if not label:
            skipped += 1
            continue
        image = images.get(int(ann["image_id"]))
        if not image:
            skipped += 1
            continue
        source_path = split_dir / Path(image["file_name"]).name
        if not source_path.exists():
            skipped += 1
            continue
        crop_path = output / "rec" / "images" / split / f"{split}_{index:08d}.jpg"
        if crop_bbox(source_path, [float(v) for v in ann["bbox"]], crop_path, pad_ratio):
            rows.append(f"{crop_path.resolve()}\t{label}")
        else:
            skipped += 1

    list_name = "val_rec.txt" if split == "val" else f"{split}_rec.txt"
    if rows:
        (output / "rec").mkdir(parents=True, exist_ok=True)
        (output / "rec" / list_name).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert manually labeled Chinese char COCO annotations into PaddleOCR recognition crops.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--pad-ratio", type=float, default=0.08)
    args = parser.parse_args()

    labels = load_labels(Path(args.labels))
    if not labels:
        raise ValueError(f"No filled labels found in: {args.labels}")

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        count, skipped = convert_split(Path(args.root), output, split, labels, args.pad_ratio)
        print(f"{split}: crops={count} skipped={skipped}")
    print(f"labels={len(labels)}")
    print(f"Saved PaddleOCR rec dataset to: {output}")


if __name__ == "__main__":
    main()
