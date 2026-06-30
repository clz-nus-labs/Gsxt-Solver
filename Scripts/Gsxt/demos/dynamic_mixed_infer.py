from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import paddle
import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GSXT_ROOT = PROJECT_ROOT / "Scripts" / "Gsxt"
PADDLEDET_ROOT = GSXT_ROOT / "third_party" / "PaddleDetection"
PADDLEOCR_ROOT = GSXT_ROOT / "third_party" / "PaddleOCR"

DEFAULT_IMAGE = GSXT_ROOT / "synthetic" / "output_v4" / "images" / "sample_00001.png"
DEFAULT_OUTPUT = GSXT_ROOT / "output" / "dynamic_mixed_infer"

DEFAULT_DET_CONFIG = PADDLEDET_ROOT / "configs" / "picodet" / "picodet_s_320_coco_lcnet.yml"
DEFAULT_DET_WEIGHTS = GSXT_ROOT / "output" / "training" / "paddledet_external_mixed" / "best_model.pdparams"
DEFAULT_DET_DATASET = GSXT_ROOT / "data" / "datasets" / "external_mixed_paddledet"

DEFAULT_REC_CONFIG = GSXT_ROOT / "output" / "training" / "chinese_char_rec_ppocrv4_domain_finetune" / "config.yml"
DEFAULT_REC_WEIGHTS = GSXT_ROOT / "output" / "training" / "chinese_char_rec_ppocrv4_domain_finetune" / "best_accuracy.pdparams"

DEFAULT_ICON_WEIGHTS = (
    GSXT_ROOT
    / "output"
    / "training"
    / "icon_cls_geetest_plus_synthetic_mobilenet_v3_large"
    / "best_accuracy.pdparams"
)
DEFAULT_ICON_LABELS = (
    GSXT_ROOT
    / "output"
    / "training"
    / "icon_cls_geetest_plus_synthetic_mobilenet_v3_large"
    / "label_list.txt"
)
DEFAULT_SEMANTIC_PHRASES = GSXT_ROOT / "data" / "semantic_phrases.txt"
DEFAULT_SEMANTIC_LEXICON_DIR = GSXT_ROOT / "data" / "semantic_lexicons"
DEFAULT_GENERAL_LEXICON = DEFAULT_SEMANTIC_LEXICON_DIR / "jieba" / "dict.txt"
_SEMANTIC_FREQUENCY_CACHE: dict[tuple[int, tuple[str, ...], bool], dict[str, int]] = {}


@dataclass
class TaskSpec:
    action: str
    modality: str
    target_source: str
    target_order: str = ""
    target_items: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

ICON_TEXT_ALIASES = {
    "三角": "triangle",
    "三角形": "triangle",
    "菱形": "diamond",
    "方框": "box",
    "框": "box",
    "正方形": "square",
    "方形": "square",
    "圆": "circle",
    "圆形": "circle",
    "伞": "umbrella",
    "雨伞": "umbrella",
    "暂停": "pause",
    "播放": "play",
    "十字": "cross",
    "叉": "cross",
    "齿轮": "gear",
    "齿轮组": "gears",
    "烟": "cigarette",
    "香烟": "cigarette",
    "苹果": "apple",
    "星": "star",
    "星星": "stars",
    "太阳": "sun",
    "月亮": "moon",
    "云": "cloud",
    "钟": "clock",
    "时钟": "clock",
    "房子": "house",
    "车": "car",
    "汽车": "car",
    "书": "book",
    "灯": "lamp",
    "钥匙": "key",
    "锁": "lock",
    "叶": "leaf",
    "叶子": "leaf",
    "花": "flower",
    "皇冠": "crown",
    "火箭": "rocket",
    "心": "heart",
    "铃": "bell",
}

BUILTIN_SEMANTIC_PHRASES = [
    "古罗马",
    "马路",
    "罗马",
    "中华",
    "中国",
    "北京",
    "上海",
    "广州",
    "深圳",
    "南宁",
    "广西",
    "山水",
    "风景",
    "天空",
    "太阳",
    "月亮",
    "星星",
    "苹果",
    "香蕉",
    "西瓜",
    "葡萄",
    "暂停",
    "播放",
    "三角",
    "菱形",
    "方框",
]


def as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def clamp_box(box: list[float], width: int, height: int, pad: int = 4) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        max(0, int(round(x1)) - pad),
        max(0, int(round(y1)) - pad),
        min(width, int(round(x2)) + pad),
        min(height, int(round(y2)) + pad),
    ]


def box_area_ratio(box: list[int] | list[float], width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / max(1.0, float(width * height))


def is_header_box(box: list[int] | list[float], height: int, header_ratio: float) -> bool:
    x1, y1, x2, y2 = box
    center_y = (y1 + y2) / 2.0
    return center_y <= height * header_ratio


def spatial_order_key(item: dict[str, Any]) -> tuple[float, float]:
    center = item.get("center") or []
    if len(center) >= 2:
        return float(center[0]), float(center[1])
    box = item.get("bbox") or [0, 0, 0, 0]
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def box_iou(a: list[int] | list[float], b: list[int] | list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_area(box: list[int] | list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_coverage(container: list[int] | list[float], inner: list[int] | list[float]) -> float:
    ix1 = max(container[0], inner[0])
    iy1 = max(container[1], inner[1])
    ix2 = min(container[2], inner[2])
    iy2 = min(container[3], inner[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return intersection / max(1.0, box_area(inner))


def suppress_bridge_boxes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """移除同时覆盖多个分离目标的异常大框，避免其把相邻图标串成一组。"""
    suppressed_ids: set[int] = set()
    for container in items:
        container_area = box_area(container["bbox"])
        if container_area <= 0:
            continue
        contained: list[dict[str, Any]] = []
        for candidate in items:
            if candidate is container or candidate.get("kind") != container.get("kind"):
                continue
            candidate_area = box_area(candidate["bbox"])
            if candidate_area >= container_area * 0.82:
                continue
            if box_coverage(container["bbox"], candidate["bbox"]) >= 0.72:
                contained.append(candidate)
        if len(contained) < 2:
            continue

        has_separate_pair = False
        for first, second in itertools.combinations(contained, 2):
            if box_iou(first["bbox"], second["bbox"]) >= 0.35:
                continue
            dx = float(first["center"][0] - second["center"][0])
            dy = float(first["center"][1] - second["center"][1])
            distance = float(np.hypot(dx, dy))
            container_span = max(
                float(container["bbox"][2] - container["bbox"][0]),
                float(container["bbox"][3] - container["bbox"][1]),
            )
            if distance >= container_span * 0.25:
                has_separate_pair = True
                break
        if has_separate_pair:
            suppressed_ids.add(id(container))

    return [item for item in items if id(item) not in suppressed_ids]


def is_single_cjk(text: str) -> bool:
    text = (text or "").strip()
    return len(text) == 1 and "\u4e00" <= text <= "\u9fff"


def compact_item_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": item.get("kind"),
        "name": item.get("text") if item.get("kind") == "char" else item.get("label"),
        "score": item.get("final_score", item.get("det_score")),
        "det_score": item.get("det_score"),
        "final_score": item.get("final_score"),
        "bbox": item.get("bbox"),
        "center": item.get("center"),
        "label": item.get("label"),
        "text": item.get("text"),
        "cls_score": item.get("cls_score"),
        "rec_score": item.get("rec_score"),
        "target_scores": item.get("target_scores"),
    }


def choose_overlap_item(
    group: list[dict[str, Any]],
    char_rec_threshold: float,
    prefer_kind: str | None = None,
) -> dict[str, Any]:
    """同一位置可能同时被识别为 char/icon，这里选择更可信的一条进入最终结果。"""
    if len(group) == 1:
        item = dict(group[0])
        item["merge_reason"] = "single"
        return item

    chars = [item for item in group if item["kind"] == "char"]
    icons = [item for item in group if item["kind"] == "icon"]
    effective_char_threshold = min(char_rec_threshold, 0.20) if prefer_kind == "char" else char_rec_threshold
    good_chars = [
        item
        for item in chars
        if is_single_cjk(str(item.get("text", "")))
        and float(item.get("rec_score", 0.0)) >= effective_char_threshold
    ]
    if good_chars:
        selected = max(good_chars, key=lambda row: (float(row.get("rec_score", 0.0)), float(row.get("det_score", 0.0))))
        item = dict(selected)
        item["merge_reason"] = "prefer_high_confidence_char"
        item["suppressed"] = [compact_item_snapshot(other) for other in group if other is not selected]
        return item

    if icons:
        selected = max(icons, key=lambda row: (float(row.get("final_score", 0.0)), float(row.get("det_score", 0.0))))
        item = dict(selected)
        item["merge_reason"] = "prefer_icon_when_char_uncertain"
        item["suppressed"] = [compact_item_snapshot(other) for other in group if other is not selected]
        return item

    item = dict(max(group, key=lambda row: float(row.get("final_score", 0.0))))
    item["merge_reason"] = "highest_score"
    return item


def merge_overlapping_items(
    items: list[dict[str, Any]],
    overlap_iou: float,
    char_rec_threshold: float,
    prefer_kind: str | None = None,
) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for item in sorted(items, key=lambda row: float(row.get("final_score", 0.0)), reverse=True):
        matched_group: list[dict[str, Any]] | None = None
        for group in groups:
            if any(box_iou(item["bbox"], other["bbox"]) >= overlap_iou for other in group):
                matched_group = group
                break
        if matched_group is None:
            groups.append([item])
        else:
            matched_group.append(item)

    merged = [choose_overlap_item(group, char_rec_threshold, prefer_kind=prefer_kind) for group in groups]
    for index, item in enumerate(merged, start=1):
        item["index"] = index
    return merged


def restore_icon_candidates_for_header_mode(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    restored: list[dict[str, Any]] = []
    for item in items:
        if item.get("kind") != "char":
            restored.append(item)
            continue
        suppressed_icons = [
            row for row in item.get("suppressed", []) if row.get("kind") == "icon" and row.get("bbox")
        ]
        if not suppressed_icons:
            restored.append(item)
            continue
        selected = max(suppressed_icons, key=lambda row: float(row.get("final_score") or row.get("score") or 0.0))
        replacement = dict(item)
        replacement.update(
            {
                "kind": "icon",
                "label": selected.get("label") or selected.get("name"),
                "cls_score": selected.get("cls_score", 0.0),
                "det_score": selected.get("det_score", item.get("det_score", 0.0)),
                "final_score": selected.get("final_score", selected.get("score", item.get("final_score", 0.0))),
                "bbox": selected.get("bbox"),
                "center": selected.get("center"),
                "merge_reason": "restore_suppressed_icon_for_header_mode",
                "restored_from_char": compact_item_snapshot(item),
            }
        )
        replacement.pop("text", None)
        replacement.pop("rec_score", None)
        restored.append(replacement)

    for index, item in enumerate(restored, start=1):
        item["index"] = index
    return restored


def has_cjk_text(text: str, min_len: int = 2) -> bool:
    return sum(1 for char in str(text or "") if "\u4e00" <= char <= "\u9fff") >= min_len


def infer_char_body_fallback_for_icon_rule(
    *,
    task_spec: TaskSpec,
    raw_results: list[dict[str, Any]],
    overlap_iou: float,
    char_rec_threshold: float,
) -> tuple[TaskSpec, list[dict[str, Any]], str, str] | None:
    """Recover real Geetest character tasks that header-icon rules steal.

    On real challenge-only Geetest crops, distorted Chinese title/body glyphs can
    be classified as icons. If the rule-selected task is icon but the header OCR
    and body OCR both expose clear CJK evidence, prefer the char detections. This
    is deliberately asymmetric: it never turns a char task into icon.
    """
    if task_spec.modality != "icon" or task_spec.action != "explicit_order":
        return None
    evidence = task_spec.evidence or {}
    header_text = str(
        evidence.get("resolved_header_target_text")
        or evidence.get("header_target_text")
        or evidence.get("instruction_text")
        or ""
    )
    if not has_cjk_text(header_text, min_len=2):
        return None

    char_merged = merge_overlapping_items(
        [item for item in raw_results if item.get("kind") == "char"],
        overlap_iou=overlap_iou,
        char_rec_threshold=min(char_rec_threshold, 0.20),
        prefer_kind="char",
    )
    char_candidates = [
        item
        for item in char_merged
        if is_single_cjk(str(item.get("text", "")))
        and float(item.get("rec_score", 0.0)) >= 0.35
        and float(item.get("det_score", 0.0)) >= 0.20
    ]
    if len(char_candidates) < 3:
        return None

    # Confidence selects which glyphs are reliable. If the header already gave
    # a CJK target order, use that order to match body glyphs; do not let body
    # layout or detection confidence rewrite the prompt order.
    selected_by_score = sorted(
        char_candidates,
        key=lambda row: (
            float(row.get("rec_score", 0.0)),
            float(row.get("det_score", 0.0)),
            float(row.get("final_score", 0.0)),
        ),
        reverse=True,
    )[:3]
    header_chars = [char for char in header_text if "\u4e00" <= char <= "\u9fff"][:3]
    if len(header_chars) == 3:
        selected = order_items_by_target(
            selected_by_score,
            ",".join(header_chars),
            normalize_aliases=False,
            keep_unmatched=False,
        )
        if len(selected) < 3:
            selected = selected_by_score
        target_chars_for_order = header_chars
    else:
        selected = sorted(selected_by_score, key=spatial_order_key)
        target_chars_for_order = [str(item.get("text") or "") for item in selected]
    for index, item in enumerate(selected, start=1):
        item["index"] = index
        item["merge_reason"] = item.get("merge_reason") or "char_body_fallback_for_icon_rule"

    target_order = ",".join(target_chars_for_order)
    new_evidence = dict(evidence)
    new_evidence["char_body_fallback_for_icon_rule"] = {
        "previous_target_source": task_spec.target_source,
        "header_text": header_text,
        "candidate_count": len(char_candidates),
    }
    new_spec = TaskSpec(
        action="explicit_order",
        modality="char",
        target_source="char_body_fallback_for_icon_rule",
        target_order=target_order,
        confidence=max(float(task_spec.confidence or 0.0), 0.55),
        evidence=new_evidence,
    )
    return new_spec, selected, target_order, new_spec.target_source


def reinterpret_icon_results_as_char_if_cjk_evidence(
    *,
    task_spec: TaskSpec,
    final_results: list[dict[str, Any]],
    target_source_resolved: str,
) -> tuple[TaskSpec, list[dict[str, Any]], str, str] | None:
    """Keep click points but report char mode when icon labels are likely bogus.

    Some real Geetest character tasks have one body glyph detected only as icon.
    If the header OCR sees a multi-character CJK target and at least two selected
    body points have strong CJK OCR evidence, the safest correction is to keep the
    selected points/order but expose them as a character task instead of arbitrary
    icon labels such as "cross".
    """
    if task_spec.modality != "icon" or task_spec.action != "explicit_order":
        return None
    evidence = task_spec.evidence or {}
    header_text = "".join(
        char
        for char in str(
            evidence.get("resolved_header_target_text")
            or evidence.get("header_target_text")
            or ""
        )
        if "\u4e00" <= char <= "\u9fff"
    )
    header_text_score = float(evidence.get("header_target_text_score") or 0.0)
    header_text_reliable = header_text_score >= 0.85
    icon_header_without_body_support = target_source_resolved == "header_icons_ignored_no_body_icons"
    if len(final_results) < 3:
        return None
    if len(header_text) < 2 and not icon_header_without_body_support:
        return None

    strong_cjk_points = 0
    converted: list[dict[str, Any]] = []
    for index, item in enumerate(final_results, start=1):
        candidate_scores = item.get("candidate_scores") or {}
        best_char = ""
        best_score = 0.0
        for char, score in candidate_scores.items():
            if "\u4e00" <= str(char) <= "\u9fff" and float(score) > best_score:
                best_char = str(char)
                best_score = float(score)
        if not best_char:
            for source_key in ("restored_from_char", "promoted_from_char"):
                source = item.get(source_key) or {}
                source_text = str(source.get("text") or "")
                if is_single_cjk(source_text):
                    best_char = source_text
                    best_score = max(best_score, float(source.get("rec_score") or 0.0))
                    break
        if header_text_reliable and index <= len(header_text):
            best_char = header_text[index - 1]
        if best_score >= 0.70 or item.get("detected_kind") == "char":
            strong_cjk_points += 1
        row = dict(item)
        row["kind"] = "char"
        row["text"] = best_char or "unknown_char"
        row["rec_score"] = best_score
        row["final_score"] = float(row.get("det_score", 0.0)) * max(best_score, 0.5)
        row["merge_reason"] = "reinterpret_icon_as_char_with_cjk_evidence"
        row.pop("label", None)
        row.pop("cls_score", None)
        row["index"] = index
        converted.append(row)

    required_cjk_points = 3 if icon_header_without_body_support else 2
    if strong_cjk_points < required_cjk_points:
        return None

    converted_ordered = converted
    target_order = ",".join(str(item.get("text") or "") for item in converted_ordered)
    new_evidence = dict(evidence)
    new_evidence["reinterpret_icon_as_char_with_cjk_evidence"] = {
        "previous_target_source": target_source_resolved,
        "header_text": header_text,
        "header_text_score": header_text_score,
        "header_text_reliable": header_text_reliable,
        "strong_cjk_points": strong_cjk_points,
        "icon_header_without_body_support": icon_header_without_body_support,
    }
    new_spec = TaskSpec(
        action="semantic_order" if icon_header_without_body_support and not header_text else "explicit_order",
        modality="char",
        target_source="reinterpret_icon_as_char_with_cjk_evidence",
        target_order=target_order,
        confidence=max(float(task_spec.confidence or 0.0), 0.55),
        evidence=new_evidence,
    )
    return new_spec, converted_ordered, target_order, new_spec.target_source


def promote_char_candidates_for_header_mode(
    items: list[dict[str, Any]],
    image_rgb: Image.Image,
    icon_model,
    icon_labels: list[str],
) -> list[dict[str, Any]]:
    """Reclassify char-only detections as icon candidates for icon-order tasks."""
    promoted: list[dict[str, Any]] = []
    for item in items:
        if item.get("kind") != "char":
            promoted.append(item)
            continue
        x1, y1, x2, y2 = item["bbox"]
        area_ratio = box_area(item["bbox"]) / max(
            1.0, float(image_rgb.width * image_rgb.height)
        )
        aspect_ratio = (x2 - x1) / max(1.0, float(y2 - y1))
        if area_ratio > 0.06 or not 0.35 <= aspect_ratio <= 2.8:
            continue
        label, score = classify_icon(
            icon_model,
            icon_labels,
            image_rgb.crop((x1, y1, x2, y2)),
        )
        replacement = dict(item)
        replacement.update(
            {
                "kind": "icon",
                "label": label,
                "cls_score": score,
                "final_score": float(item.get("det_score", 0.0)) * score,
                "merge_reason": "promote_char_candidate_for_header_mode",
                "promoted_from_char": compact_item_snapshot(item),
            }
        )
        replacement.pop("text", None)
        replacement.pop("rec_score", None)
        promoted.append(replacement)
    return promoted


def parse_target_order(target_order: str) -> list[str]:
    target_order = (target_order or "").strip()
    if not target_order:
        return []
    if "," in target_order or "，" in target_order:
        return [part.strip() for part in target_order.replace("，", ",").split(",") if part.strip()]
    return [char for char in target_order if not char.isspace()]


def item_name(item: dict[str, Any]) -> str:
    return str(item.get("text") if item["kind"] == "char" else item.get("label", ""))


def normalize_target_token(token: str) -> str:
    token = token.strip()
    return ICON_TEXT_ALIASES.get(token, token)


def order_items_by_target(
    items: list[dict[str, Any]],
    target_order: str,
    normalize_aliases: bool = True,
    keep_unmatched: bool = True,
) -> list[dict[str, Any]]:
    targets = parse_target_order(target_order)
    if not targets:
        return items

    remaining = list(items)
    normalized_targets = [
        normalize_target_token(target) if normalize_aliases else target.strip()
        for target in targets
    ]
    assignments: dict[int, dict[str, Any]] = {}

    # First lock exact OCR matches. The small target word is usually much
    # clearer than the distorted body characters, so confirmed matches reduce
    # the remaining assignment problem substantially.
    for target_index, target in enumerate(normalized_targets):
        exact_indexes = [
            idx for idx, item in enumerate(remaining) if item_name(item) == target
        ]
        if not exact_indexes:
            continue
        matched_index = max(
            exact_indexes,
            key=lambda idx: (
                float(remaining[idx].get("rec_score", 0.0)),
                float(remaining[idx].get("final_score", 0.0)),
            ),
        )
        selected = remaining.pop(matched_index)
        selected["target_text"] = target
        selected["recognized_text"] = item_name(selected)
        selected["text"] = target
        selected["target_match_reason"] = "exact_ocr"
        selected["target_similarity_score"] = float(
            selected.get("target_scores", {}).get(target, selected.get("rec_score", 0.0))
        )
        assignments[target_index] = selected

    unmatched_target_indexes = [
        idx for idx in range(len(normalized_targets)) if idx not in assignments
    ]
    assign_count = min(len(unmatched_target_indexes), len(remaining))
    if assign_count:
        best_pairs: list[tuple[int, dict[str, Any]]] = []
        best_score = float("-inf")
        candidate_indexes = range(len(remaining))
        for item_indexes in itertools.permutations(candidate_indexes, assign_count):
            score = 0.0
            pairs: list[tuple[int, dict[str, Any]]] = []
            for offset, item_index in enumerate(item_indexes):
                target_index = unmatched_target_indexes[offset]
                target = normalized_targets[target_index]
                item = remaining[item_index]
                probability = float(item.get("target_scores", {}).get(target, 0.0))
                score += float(np.log(max(probability, 1e-8)))
                pairs.append((target_index, item))
            if score > best_score:
                best_score = score
                best_pairs = pairs

        used_ids: set[int] = set()
        for target_index, item in best_pairs:
            target = normalized_targets[target_index]
            selected = dict(item)
            selected["target_text"] = target
            selected["recognized_text"] = item_name(selected)
            selected["text"] = target
            selected["target_match_reason"] = "ctc_target_similarity"
            selected["target_similarity_score"] = float(
                selected.get("target_scores", {}).get(target, 0.0)
            )
            assignments[target_index] = selected
            used_ids.add(id(item))
        remaining = [item for item in remaining if id(item) not in used_ids]

    ordered = [
        assignments[target_index]
        for target_index in range(len(normalized_targets))
        if target_index in assignments
    ]
    if keep_unmatched:
        ordered.extend(remaining)
    for index, item in enumerate(ordered, start=1):
        item["index"] = index
    return ordered


def normalize_phrase_token(token: str) -> str:
    token = re.sub(r"\s+", "", token.strip())
    return "".join(char for char in token if "\u4e00" <= char <= "\u9fff")


def iter_semantic_sources(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        for pattern in ("*.txt", "*.dict", "*.csv", "*.tsv"):
            files.extend(
                file_path
                for file_path in sorted(path.rglob(pattern))
                if "raw_corpus" not in file_path.parts
            )
    return files


def iter_phrase_file(path: Path) -> list[str]:
    phrases: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # THUOCL/jieba/userdict usually stores: word freq tag. CSV/TSV stores word in the first field.
        token = re.split(r"[\t,，\s]+", line, maxsplit=1)[0]
        phrase = normalize_phrase_token(token)
        if phrase:
            phrases.append(phrase)
    return phrases


def iter_phrase_frequencies(path: Path, default_frequency: int = 1000) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\t,，\s]+", line, maxsplit=2)
        phrase = normalize_phrase_token(parts[0])
        if not phrase:
            continue
        frequency = default_frequency
        if len(parts) >= 2:
            try:
                frequency = max(1, int(parts[1]))
            except ValueError:
                pass
        frequencies[phrase] = max(frequencies.get(phrase, 0), frequency)
    return frequencies


def load_jieba_phrases(max_len: int, charset: set[str]) -> list[str]:
    try:
        import jieba  # type: ignore
    except ImportError:
        return []

    try:
        jieba.initialize()
        freq_dict = getattr(jieba.dt, "FREQ", {})
    except Exception:
        return []

    phrases: list[str] = []
    for word in freq_dict:
        phrase = normalize_phrase_token(str(word))
        if 2 <= len(phrase) <= max_len and all(char in charset for char in phrase):
            phrases.append(phrase)
    return phrases


def load_semantic_phrase_frequencies(
    path: Path,
    max_len: int,
    extra_paths: list[Path] | None = None,
    use_jieba: bool = True,
) -> dict[str, int]:
    extra_paths = extra_paths or []
    cache_key = (
        max_len,
        tuple(sorted(str(extra.resolve()) for extra in extra_paths if extra.exists())),
        use_jieba,
    )
    cached = _SEMANTIC_FREQUENCY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    frequencies = {
        normalize_phrase_token(phrase): 100_000
        for phrase in BUILTIN_SEMANTIC_PHRASES
        if normalize_phrase_token(phrase)
    }
    source_paths = [path]
    if DEFAULT_SEMANTIC_LEXICON_DIR.exists():
        source_paths.append(DEFAULT_SEMANTIC_LEXICON_DIR)
    source_paths.extend(extra_paths)
    for source in iter_semantic_sources(source_paths):
        default_frequency = 100_000 if "manual" in source.parts else 1000
        for phrase, frequency in iter_phrase_frequencies(
            source, default_frequency=default_frequency
        ).items():
            if 1 <= len(phrase) <= max_len:
                frequencies[phrase] = max(frequencies.get(phrase, 0), frequency)

    if use_jieba:
        try:
            import jieba  # type: ignore

            jieba.initialize()
            for word, frequency in getattr(jieba.dt, "FREQ", {}).items():
                phrase = normalize_phrase_token(str(word))
                if 1 <= len(phrase) <= max_len:
                    frequencies[phrase] = max(
                        frequencies.get(phrase, 0), max(1, int(frequency or 1))
                    )
        except (ImportError, ValueError, TypeError):
            pass

    _SEMANTIC_FREQUENCY_CACHE[cache_key] = frequencies
    return frequencies


def phrase_language_score(phrase: str, frequencies: dict[str, int]) -> float:
    """Score a short phrase by its best word segmentation using unigram frequencies."""
    corpus_frequency = 60_101_967.0
    length = len(phrase)
    scores = [float("-inf")] * (length + 1)
    scores[0] = 0.0
    for end in range(1, length + 1):
        for start in range(end):
            token = phrase[start:end]
            frequency = frequencies.get(token)
            if frequency is None:
                if len(token) != 1:
                    continue
                frequency = 1
            token_score = float(np.log(max(1, frequency) / corpus_frequency))
            scores[end] = max(scores[end], scores[start] + token_score)
    return scores[length] / max(1, length)


def phrase_syntax_score(phrase: str) -> float:
    """Small general-purpose prior for common Chinese function-word placement."""
    score = 0.0
    final_particles = set("的了着过吗呢吧啊呀")
    degree_adverbs = set("很更最太较挺颇极")
    for index, char in enumerate(phrase):
        if char in final_particles:
            score += 1.0 if index == len(phrase) - 1 else -0.7
        if char in degree_adverbs:
            if index == 0:
                score += 1.0
            elif index < len(phrase) - 1:
                score += 0.3
            else:
                score -= 0.7
    return score


def load_semantic_phrases(
    path: Path,
    max_len: int,
    charset: set[str],
    extra_paths: list[Path] | None = None,
    use_jieba: bool = True,
) -> list[str]:
    phrases = list(BUILTIN_SEMANTIC_PHRASES)
    source_paths = [path]
    if DEFAULT_SEMANTIC_LEXICON_DIR.exists():
        source_paths.append(DEFAULT_SEMANTIC_LEXICON_DIR)
    if extra_paths:
        source_paths.extend(extra_paths)

    for source in iter_semantic_sources(source_paths):
        phrases.extend(iter_phrase_file(source))
    if use_jieba:
        phrases.extend(load_jieba_phrases(max_len=max_len, charset=charset))

    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        phrase = normalize_phrase_token(phrase)
        if len(phrase) != max_len:
            continue
        if not all(char in charset for char in phrase):
            continue
        if phrase not in seen:
            seen.add(phrase)
            unique.append(phrase)
    return unique


def phrase_matches_chars(phrase: str, chars: list[str]) -> bool:
    if len(phrase) != len(chars):
        return False
    remaining = list(chars)
    for char in phrase:
        if char not in remaining:
            return False
        remaining.remove(char)
    return not remaining


def infer_semantic_target_order(
    items: list[dict[str, Any]],
    phrase_path: Path,
    extra_phrase_paths: list[Path] | None = None,
    use_jieba: bool = True,
) -> tuple[str, str]:
    char_items = [item for item in items if item["kind"] == "char" and is_single_cjk(str(item.get("text", "")))]
    if len(char_items) < 2 or len(char_items) != len(items):
        return "", "semantic-not-char-only"

    chars = [str(item["text"]) for item in char_items]
    frequencies = load_semantic_phrase_frequencies(
        phrase_path,
        max_len=len(chars),
        extra_paths=extra_phrase_paths,
        use_jieba=use_jieba,
    )

    candidate_scores = [
        {
            str(char): float(score)
            for char, score in item.get("candidate_scores", {}).items()
            if is_single_cjk(str(char))
        }
        for item in char_items
    ]
    candidate_charset = set().union(*(scores.keys() for scores in candidate_scores))
    if not candidate_charset:
        return "", "semantic-no-match"

    phrase_candidates = {
        phrase
        for phrase in frequencies
        if len(phrase) == len(chars) and all(char in candidate_charset for char in phrase)
    }
    phrase_candidates.update("".join(order) for order in set(itertools.permutations(chars)))
    short_candidates = [
        list(scores)[:3] or [char]
        for scores, char in zip(candidate_scores, chars)
    ]
    for values in itertools.product(*short_candidates):
        phrase_candidates.update("".join(order) for order in set(itertools.permutations(values)))

    ranked: list[tuple[float, float, float, int, str]] = []
    for phrase in phrase_candidates:
        best_log_score = float("-inf")
        best_min_score = 0.0
        for item_order in itertools.permutations(range(len(char_items))):
            scores = [
                candidate_scores[item_index].get(char, 0.0)
                for char, item_index in zip(phrase, item_order)
            ]
            if any(score <= 0.0 for score in scores):
                continue
            log_score = float(np.mean(np.log(np.maximum(scores, 1e-8))))
            if log_score > best_log_score:
                best_log_score = log_score
                best_min_score = min(scores)
        if np.isfinite(best_log_score):
            language_score = phrase_language_score(phrase, frequencies)
            syntax_score = phrase_syntax_score(phrase)
            lexical_frequency = frequencies.get(phrase, 0)
            joint_score = best_log_score + 0.65 * language_score + 0.35 * syntax_score
            ranked.append(
                (joint_score, best_log_score, best_min_score, lexical_frequency, phrase)
            )

    if not ranked:
        return "", "semantic-no-match"
    ranked.sort(reverse=True)
    best_joint, best_visual_log, best_min_score, best_frequency, best_phrase = ranked[0]
    best_mean_score = float(np.exp(best_visual_log))
    second_joint = ranked[1][0] if len(ranked) > 1 else float("-inf")
    margin = best_joint - second_joint
    # Distorted body glyphs often put the correct character far below top-1,
    # while the three-character phrase itself is common and visually plausible
    # as a whole (for example "台北市").  A hard 1e-7 per-character floor was
    # too aggressive and forced semantic tasks back to spatial/OCR order.  Keep
    # the joint visual mean and margin checks as the real guardrails, but allow
    # one weak glyph candidate to survive.
    if best_mean_score >= 0.001 and best_min_score >= 1e-9 and margin >= 0.06:
        return (
            best_phrase,
            f"semantic-joint:{best_phrase}:visual={best_mean_score:.3f}:"
            f"freq={best_frequency}:margin={margin:.3f}",
        )
    return "", "semantic-no-match"


def run_paddledet(
    image: Path,
    output_dir: Path,
    config: Path,
    weights: Path,
    dataset_dir: Path,
    threshold: float,
    use_gpu: bool,
) -> list[dict[str, Any]]:
    det_output = output_dir / "det"
    det_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        as_posix(PADDLEDET_ROOT / "tools" / "infer.py"),
        "-c",
        as_posix(config),
        "--infer_img",
        as_posix(image),
        "--output_dir",
        as_posix(det_output),
        "--draw_threshold",
        str(threshold),
        "--save_threshold",
        str(threshold),
        "--save_results",
        "True",
        "--visualize",
        "False",
        "-o",
        f"weights={as_posix(weights)}",
        f"use_gpu={'true' if use_gpu else 'false'}",
        "num_classes=2",
        f"TestDataset.dataset_dir={as_posix(dataset_dir)}",
        "TestDataset.image_dir=images",
        "TestDataset.anno_path=val.json",
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    json_candidates = sorted(det_output.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_candidates:
        raise FileNotFoundError(f"No PaddleDetection JSON result found under {det_output}")

    payload = json.loads(json_candidates[0].read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("bbox", payload.get("results", payload.get("detections", [])))
    else:
        records = payload

    items: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        score = float(row.get("score", row.get("confidence", 0.0)))
        if score < threshold:
            continue
        raw_bbox = row.get("bbox", row.get("box", row.get("rect")))
        if not raw_bbox:
            continue
        if len(raw_bbox) == 4:
            # PaddleDetection saved results use COCO xywh in most versions.
            box = xywh_to_xyxy([float(v) for v in raw_bbox])
        else:
            continue

        cls = row.get("category", row.get("label", row.get("category_name")))
        cls_id = row.get("category_id", row.get("class_id", row.get("clsid")))
        if cls in {"char", "icon"}:
            label = str(cls)
        elif cls_id in {1, "1"}:
            label = "char"
        elif cls_id in {2, "2"}:
            label = "icon"
        elif cls_id in {0, "0"}:
            label = "char"
        else:
            label = "icon"

        items.append({"kind": label, "score": score, "bbox": box, "raw": row})

    return items


def load_ocr_model(config_path: Path, weights_path: Path, use_gpu: bool):
    sys.path.insert(0, str(PADDLEOCR_ROOT))
    from ppocr.modeling.architectures import build_model  # type: ignore
    from ppocr.postprocess import build_post_process  # type: ignore

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    global_config = config["Global"]
    dict_path = global_config.get("character_dict_path")
    if dict_path and not Path(dict_path).is_absolute():
        candidate = PADDLEOCR_ROOT / dict_path
        if candidate.exists():
            global_config["character_dict_path"] = str(candidate)
    elif not dict_path:
        global_config["character_dict_path"] = str(PADDLEOCR_ROOT / "ppocr" / "utils" / "ppocr_keys_v1.txt")
    post_process = build_post_process(config["PostProcess"], global_config)
    char_num = len(getattr(post_process, "character", []))

    arch = config["Architecture"]
    if arch.get("Head", {}).get("name") == "MultiHead":
        arch["Head"]["out_channels_list"] = {
            "CTCLabelDecode": char_num,
            "NRTRLabelDecode": char_num + 2,
        }
    else:
        arch.setdefault("Head", {})["out_channels"] = char_num

    model = build_model(arch)
    state = paddle.load(str(weights_path))
    model.set_state_dict(state)
    model.eval()
    paddle.set_device("gpu" if use_gpu else "cpu")
    return model, post_process


def preprocess_rec(crop_bgr: np.ndarray, image_shape: tuple[int, int, int] = (3, 48, 320)) -> np.ndarray:
    img_c, img_h, img_w = image_shape
    h, w = crop_bgr.shape[:2]
    ratio = w / float(max(1, h))
    resized_w = min(img_w, max(1, int(np.ceil(img_h * ratio))))
    resized = cv2.resize(crop_bgr, (resized_w, img_h))
    resized = resized.astype("float32") / 255.0
    resized = (resized - 0.5) / 0.5
    resized = resized.transpose(2, 0, 1)
    padded = np.zeros((img_c, img_h, img_w), dtype="float32")
    padded[:, :, :resized_w] = resized
    return padded


@paddle.no_grad()
def recognize_char_with_targets(
    model,
    post_process,
    crop_bgr: np.ndarray,
    target_chars: list[str] | None = None,
    candidate_top_k: int = 256,
) -> tuple[str, float, dict[str, float], dict[str, float]]:
    tensor = paddle.to_tensor(preprocess_rec(crop_bgr)[None, :, :, :])
    preds = model(tensor)
    decoded = post_process(preds)
    if isinstance(decoded, dict):
        decoded = decoded.get("ctc", decoded.get("res", []))
    text = ""
    score = 0.0
    if decoded and isinstance(decoded[0], (list, tuple)) and len(decoded[0]) >= 2:
        text = str(decoded[0][0])
        score = float(decoded[0][1])

    target_scores: dict[str, float] = {}
    candidate_scores: dict[str, float] = {}
    if isinstance(preds, paddle.Tensor) and len(preds.shape) == 3:
        raw_scores = preds.numpy()[0]
        row_sums = raw_scores.sum(axis=1)
        if raw_scores.min() >= 0.0 and np.allclose(row_sums, 1.0, atol=1e-3):
            probabilities = raw_scores
        else:
            probabilities = paddle.nn.functional.softmax(preds, axis=2).numpy()[0]
        character_list = list(getattr(post_process, "character", []))
        character_indexes = {char: idx for idx, char in enumerate(character_list)}
        for char in dict.fromkeys(target_chars or []):
            char_index = character_indexes.get(char)
            if char_index is None or char_index >= probabilities.shape[1]:
                target_scores[char] = 0.0
                continue
            target_scores[char] = float(probabilities[:, char_index].max())
        max_scores = probabilities.max(axis=0)
        ranked_chars = sorted(
            (
                (float(max_scores[index]), str(char))
                for index, char in enumerate(character_list[: len(max_scores)])
                if is_single_cjk(str(char))
            ),
            reverse=True,
        )
        candidate_scores = {
            char: char_score for char_score, char in ranked_chars[: max(1, candidate_top_k)]
        }
    return text, score, target_scores, candidate_scores


def recognize_char(model, post_process, crop_bgr: np.ndarray) -> tuple[str, float]:
    text, score, _target_scores, _candidate_scores = recognize_char_with_targets(
        model, post_process, crop_bgr
    )
    return text, score


def load_icon_model(weights_path: Path, labels_path: Path, model_name: str, use_gpu: bool):
    sys.path.insert(0, str(GSXT_ROOT / "training"))
    from train_synthetic_icon_cls import build_model  # type: ignore

    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    model = build_model(model_name, num_classes=len(labels), pretrained=False)
    model.set_state_dict(paddle.load(str(weights_path)))
    model.eval()
    paddle.set_device("gpu" if use_gpu else "cpu")
    return model, labels


def preprocess_icon(crop_rgb: Image.Image, image_size: int = 128) -> np.ndarray:
    img = crop_rgb.convert("RGB")
    img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    canvas.paste(img, ((image_size - img.width) // 2, (image_size - img.height) // 2))
    arr = np.asarray(canvas).astype("float32") / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype="float32")) / np.array(
        [0.229, 0.224, 0.225], dtype="float32"
    )
    return arr.transpose(2, 0, 1)


@paddle.no_grad()
def classify_icon_with_scores(
    model,
    labels: list[str],
    crop_rgb: Image.Image,
    top_k: int = 20,
) -> tuple[str, float, dict[str, float]]:
    tensor = paddle.to_tensor(preprocess_icon(crop_rgb)[None, :, :, :])
    probs = paddle.nn.functional.softmax(model(tensor), axis=1).numpy()[0]
    idx = int(np.argmax(probs))
    top_indexes = np.argsort(-probs)[: min(top_k, len(labels))]
    scores = {labels[int(index)]: float(probs[int(index)]) for index in top_indexes}
    return labels[idx], float(probs[idx]), scores


def classify_icon(model, labels: list[str], crop_rgb: Image.Image) -> tuple[str, float]:
    label, score, _scores = classify_icon_with_scores(model, labels, crop_rgb)
    return label, score


@paddle.no_grad()
def extract_icon_embeddings(model, crops_rgb: list[Image.Image]) -> np.ndarray:
    """Return normalized penultimate visual features for open-set matching."""
    batch = np.stack([preprocess_icon(crop) for crop in crops_rgb], axis=0)
    tensor = paddle.to_tensor(batch)
    if all(hasattr(model, name) for name in ("conv", "blocks", "lastconv", "avgpool")):
        features = model.conv(tensor)
        features = model.blocks(features)
        features = model.lastconv(features)
        features = model.avgpool(features)
        features = paddle.flatten(features, start_axis=1)
    else:
        features = model(tensor)
    vectors = features.numpy().astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def extract_icon_embedding(model, crop_rgb: Image.Image) -> np.ndarray:
    return extract_icon_embeddings(model, [crop_rgb])[0]


def icon_embedding_for_box(
    model,
    image_rgb: Image.Image,
    box: list[int],
    *,
    header: bool,
) -> np.ndarray:
    """Average original and silhouette views to reduce color/background bias."""
    x1, y1, x2, y2 = box
    crop = image_rgb.crop((x1, y1, x2, y2)).convert("RGB")
    mask = icon_shape_mask(image_rgb, box, header=header)
    variants = [crop]
    binary = mask > 0.25
    if binary.any():
        for color in ((255, 255, 255), (0, 255, 255)):
            canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
            canvas[binary] = color
            variants.append(Image.fromarray(canvas, mode="RGB"))
    embeddings = extract_icon_embeddings(model, variants)
    vector = np.mean(embeddings, axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def adaptive_foreground_mask(crop: np.ndarray, min_delta: int = 38) -> np.ndarray:
    """Extract foreground from black/white/colored prompt areas without assuming a fixed text color."""
    if crop.size == 0:
        return np.zeros(crop.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    border = np.concatenate(
        [
            crop[:3, :, :].reshape(-1, 3),
            crop[-3:, :, :].reshape(-1, 3),
            crop[:, :3, :].reshape(-1, 3),
            crop[:, -3:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border.astype(np.float32), axis=0)
    bg_gray = float(np.median(cv2.cvtColor(border.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY)))
    color_dist = np.linalg.norm(crop.astype(np.float32) - bg[None, None, :], axis=2)

    if bg_gray >= 145:
        mask = np.where(gray < bg_gray - min_delta, 255, 0).astype(np.uint8)
    elif bg_gray <= 110:
        mask = np.where(gray > bg_gray + min_delta, 255, 0).astype(np.uint8)
    else:
        mask = np.where(color_dist > max(42, min_delta), 255, 0).astype(np.uint8)

    color_mask = np.where(color_dist > max(55, min_delta + 10), 255, 0).astype(np.uint8)
    mask = cv2.bitwise_or(mask, color_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    return mask


def header_foreground_mask(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return np.zeros(crop.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    border = np.concatenate(
        [
            crop[:3, :, :].reshape(-1, 3),
            crop[-3:, :, :].reshape(-1, 3),
            crop[:, :3, :].reshape(-1, 3),
            crop[:, -3:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_gray = float(np.median(cv2.cvtColor(border.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY)))
    if bg_gray >= 145:
        mask = np.where(gray < 125, 255, 0).astype(np.uint8)
    elif bg_gray <= 110:
        mask = np.where(gray > 145, 255, 0).astype(np.uint8)
    else:
        mask = adaptive_foreground_mask(crop)
    return mask


def detect_header_icon_boxes(
    image_rgb: Image.Image,
    header_ratio: float = 0.20,
    right_start_ratio: float = 0.45,
) -> list[list[int]]:
    arr = np.asarray(image_rgb.convert("RGB"))
    height, width = arr.shape[:2]
    y1 = 0
    y2 = max(1, int(height * header_ratio))
    x1 = max(0, int(width * right_start_ratio))
    x2 = width

    crop = arr[y1:y2, x1:x2]
    mask = header_foreground_mask(crop)
    col_sum = (mask > 0).sum(axis=0)
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for col_idx, value in enumerate(col_sum):
        if value > 0 and run_start is None:
            run_start = col_idx
        if (value == 0 or col_idx == len(col_sum) - 1) and run_start is not None:
            run_end = col_idx if value == 0 else col_idx + 1
            if run_end - run_start >= 8:
                runs.append((run_start, run_end))
            run_start = None

    def split_wide_header_box(local_start: int, local_end: int, box: list[int]) -> list[list[int]]:
        # Use the unpadded foreground bounds here. The public box includes a
        # six-pixel margin on every side; using that padded height can hide a
        # genuinely wide run containing two touching prompt icons.
        run_mask = mask[:, local_start:local_end]
        run_coords = cv2.findNonZero(run_mask)
        if run_coords is None:
            return [box]
        _, _, foreground_width, foreground_height = cv2.boundingRect(run_coords)
        peer_widths = [
            end - start
            for start, end in runs
            if start != local_start or end != local_end
        ]
        typical_peer_width = (
            float(np.median(peer_widths)) if peer_widths else 0.0
        )
        aspect_wide = (
            foreground_height > 0
            and foreground_width / max(1, foreground_height) >= 1.9
        )
        peer_wide = (
            len(runs) >= 2
            and foreground_width >= 48
            and typical_peer_width > 0
            and foreground_width >= typical_peer_width * 1.65
        )
        if not aspect_wide and not peer_wide:
            return [box]
        # Several black prompt icons often touch after thresholding. When a
        # single run is much wider than one icon, split it into same-size
        # slices, then re-tighten each slice to foreground pixels.
        estimated = int(
            round(
                foreground_width
                / max(28.0, foreground_height * 0.9)
            )
        )
        estimated = max(2, min(5, estimated))
        split_boxes: list[list[int]] = []
        for idx in range(estimated):
            seg_start = int(round(local_start + (local_end - local_start) * idx / estimated))
            seg_end = int(round(local_start + (local_end - local_start) * (idx + 1) / estimated))
            submask = mask[:, seg_start:seg_end]
            coords = cv2.findNonZero(submask)
            if coords is None:
                continue
            sx, sy, sw, sh = cv2.boundingRect(coords)
            if sw < 6 or sh < 8:
                continue
            split_boxes.append(
                [
                    max(0, x1 + seg_start + sx - 6),
                    max(0, y1 + sy - 6),
                    min(width, x1 + seg_start + sx + sw + 6),
                    min(height, y1 + sy + sh + 6),
                ]
            )
        return split_boxes if len(split_boxes) >= 2 else [box]

    projection_boxes: list[list[int]] = []
    for start, end in runs:
        submask = mask[:, start:end]
        coords = cv2.findNonZero(submask)
        if coords is None:
            continue
        bx, by, bw, bh = cv2.boundingRect(coords)
        if bw < 8 or bh < 8:
            continue
        box = [
            max(0, x1 + start + bx - 6),
            max(0, y1 + by - 6),
            min(width, x1 + start + bx + bw + 6),
            min(height, y1 + by + bh + 6),
        ]
        projection_boxes.extend(split_wide_header_box(start, end, box))
    if len(projection_boxes) >= 2:
        return projection_boxes

    mask = cv2.dilate(mask, np.ones((2, 2), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts: list[list[int]] = []
    for contour in contours:
        rx, ry, rw, rh = cv2.boundingRect(contour)
        area = rw * rh
        if area < 30 or area > 8000:
            continue
        if rw < 4 or rh < 4:
            continue
        if rw > width * 0.25 or rh > height * 0.18:
            continue
        parts.append([x1 + rx, y1 + ry, x1 + rx + rw, y1 + ry + rh])

    # Merge small contour pieces into nearby prompt icons without joining adjacent icons.
    parts = sorted(parts, key=lambda box: box[0])
    clusters: list[list[int]] = []
    max_icon_width = max(28, int(width * 0.095))
    for box in parts:
        if not clusters:
            clusters.append(box)
            continue
        prev = clusters[-1]
        new_span_width = max(prev[2], box[2]) - min(prev[0], box[0])
        horizontal_gap = box[0] - prev[2]
        vertical_overlap = min(prev[3], box[3]) - max(prev[1], box[1])
        if new_span_width <= max_icon_width and horizontal_gap <= 16 and vertical_overlap > -14:
            prev[0] = min(prev[0], box[0])
            prev[1] = min(prev[1], box[1])
            prev[2] = max(prev[2], box[2])
            prev[3] = max(prev[3], box[3])
        else:
            clusters.append(box)

    boxes: list[list[int]] = []
    for box in clusters:
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        if bw < 10 or bh < 10:
            continue
        boxes.append(
            [
                max(0, box[0] - 6),
                max(0, box[1] - 6),
                min(width, box[2] + 6),
                min(height, box[3] + 6),
            ]
        )

    # Final pass for accidental duplicate fragments.
    boxes = sorted(boxes, key=lambda box: box[0])
    merged: list[list[int]] = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        prev = merged[-1]
        horizontal_gap = box[0] - prev[2]
        vertical_overlap = min(prev[3], box[3]) - max(prev[1], box[1])
        if horizontal_gap <= 2 and vertical_overlap > 0:
            prev[0] = min(prev[0], box[0])
            prev[1] = min(prev[1], box[1])
            prev[2] = max(prev[2], box[2])
            prev[3] = max(prev[3], box[3])
        else:
            merged.append(box)
    return merged


def has_separate_header_target_group(
    image_rgb: Image.Image,
    header_ratio: float,
    first_target_left: int,
    min_gap_ratio: float = 0.045,
) -> bool:
    """Reject centered instruction text mistaken as right-side prompt targets."""
    arr = np.asarray(image_rgb.convert("RGB"))
    height, width = arr.shape[:2]
    y2 = max(1, int(height * header_ratio))
    first_target_left = int(np.clip(first_target_left, 0, width))
    if first_target_left <= 0:
        return True
    if first_target_left >= int(width * 0.70):
        return True
    header = arr[:y2, :first_target_left]
    mask = header_foreground_mask(header)
    _ys, xs = np.where(mask > 0)
    if not len(xs):
        return True
    gap = first_target_left - int(xs.max())
    return gap >= max(24, int(width * min_gap_ratio))


def classify_header_icons(
    image_rgb: Image.Image,
    icon_model,
    icon_labels: list[str],
    header_ratio: float,
    right_start_ratio: float,
    min_score: float,
) -> list[dict[str, Any]]:
    boxes = detect_header_icon_boxes(image_rgb, header_ratio, right_start_ratio)
    image_arr = np.asarray(image_rgb.convert("RGB"))

    if len(boxes) >= 2:
        widths = [box[2] - box[0] for box in boxes]
        reference_width = float(np.median(sorted(widths)[:-1])) if len(widths) > 2 else float(min(widths))
        refined_boxes: list[list[int]] = []
        for box, box_width in zip(boxes, widths):
            if box_width < max(42, reference_width * 1.35):
                refined_boxes.append(box)
                continue
            x1, y1, x2, y2 = box
            mask = header_foreground_mask(image_arr[y1:y2, x1:x2])
            projection = (mask > 0).sum(axis=0).astype(np.float32)
            start = max(8, int(len(projection) * 0.28))
            end = min(len(projection) - 8, int(len(projection) * 0.72))
            if end <= start:
                refined_boxes.append(box)
                continue
            split = start + int(np.argmin(projection[start:end]))
            peak = max(1.0, float(projection.max()))
            if projection[split] > peak * 0.45:
                refined_boxes.append(box)
                continue
            left_mask = mask[:, :split]
            right_mask = mask[:, split:]
            left_coords = cv2.findNonZero(left_mask)
            right_coords = cv2.findNonZero(right_mask)
            if left_coords is None or right_coords is None:
                refined_boxes.append(box)
                continue
            lx, ly, lw, lh = cv2.boundingRect(left_coords)
            rx, ry, rw, rh = cv2.boundingRect(right_coords)
            refined_boxes.extend(
                [
                    [
                        max(0, x1 + lx - 5),
                        max(0, y1 + ly - 5),
                        min(image_rgb.width, x1 + lx + lw + 5),
                        min(image_rgb.height, y1 + ly + lh + 5),
                    ],
                    [
                        max(0, x1 + split + rx - 5),
                        max(0, y1 + ry - 5),
                        min(image_rgb.width, x1 + split + rx + rw + 5),
                        min(image_rgb.height, y1 + ry + rh + 5),
                    ],
                ]
            )
        boxes = refined_boxes

    merged_boxes: list[list[int]] = []
    for box in sorted(boxes, key=lambda row: row[0]):
        if not merged_boxes:
            merged_boxes.append(box)
            continue
        previous = merged_boxes[-1]
        overlap_x = max(0, min(previous[2], box[2]) - max(previous[0], box[0]))
        overlap_y = max(0, min(previous[3], box[3]) - max(previous[1], box[1]))
        min_width = max(1, min(previous[2] - previous[0], box[2] - box[0]))
        min_height = max(1, min(previous[3] - previous[1], box[3] - box[1]))
        if overlap_x / min_width >= 0.30 and overlap_y / min_height >= 0.60:
            merged_boxes[-1] = [
                min(previous[0], box[0]),
                min(previous[1], box[1]),
                max(previous[2], box[2]),
                max(previous[3], box[3]),
            ]
        else:
            merged_boxes.append(box)

    # Header detector boxes are padded and may include strokes from adjacent
    # targets. Partition overlaps at the midpoint between neighboring centers.
    if len(merged_boxes) >= 2:
        centers = [(box[0] + box[2]) / 2.0 for box in merged_boxes]
        boundaries = [(left + right) / 2.0 for left, right in zip(centers, centers[1:])]
        partitioned: list[list[int]] = []
        for index, box in enumerate(merged_boxes):
            left = max(box[0], int(np.ceil(boundaries[index - 1]))) if index else box[0]
            right = min(box[2], int(np.floor(boundaries[index]))) if index < len(boundaries) else box[2]
            partitioned.append([left, box[1], max(left + 1, right), box[3]])
        merged_boxes = partitioned

    if merged_boxes and not has_separate_header_target_group(
        image_rgb,
        header_ratio,
        min(box[0] for box in merged_boxes),
    ):
        return []

    target_items: list[dict[str, Any]] = []
    for box in merged_boxes:
        x1, y1, x2, y2 = box
        crop = image_rgb.crop((x1, y1, x2, y2)).convert("RGB")
        crop_arr = np.asarray(crop)
        mask = header_foreground_mask(crop_arr)
        variants = [crop]
        ys, xs = np.where(mask > 0)
        if len(xs) and len(ys):
            tight_mask = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            for color in ((0, 255, 255), (255, 230, 0), (255, 120, 0), (255, 255, 255)):
                canvas = np.zeros((*tight_mask.shape, 3), dtype=np.uint8)
                canvas[tight_mask > 0] = color
                variants.append(Image.fromarray(canvas, mode="RGB"))

        combined_scores: dict[str, float] = {}
        for variant in variants:
            _label, _score, variant_scores = classify_icon_with_scores(
                icon_model,
                icon_labels,
                variant,
            )
            for name, probability in variant_scores.items():
                combined_scores[name] = max(combined_scores.get(name, 0.0), probability)
        label, score = max(combined_scores.items(), key=lambda row: row[1])
        class_scores = dict(
            sorted(combined_scores.items(), key=lambda row: row[1], reverse=True)[:20]
        )
        if score < min_score:
            continue
        target_items.append(
            {
                "kind": "icon",
                "label": label,
                "cls_score": score,
                "class_scores": class_scores,
                "bbox": box,
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            }
        )
    return sorted(target_items, key=spatial_order_key)


def icon_shape_mask(image_rgb: Image.Image, box: list[int], header: bool, size: int = 64) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = np.asarray(image_rgb.crop((x1, y1, x2, y2)).convert("RGB"))
    if crop.size == 0:
        return np.zeros((size, size), dtype=np.float32)

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    if header:
        mask = header_foreground_mask(crop)
    else:
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        vivid_threshold = max(75.0, float(np.percentile(sat, 78)))
        vivid_mask = np.where((sat >= vivid_threshold) & (val > 55), 255, 0).astype(np.uint8)
        vivid_mask = cv2.morphologyEx(
            vivid_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        border = np.concatenate(
            [
                crop[:3, :, :].reshape(-1, 3),
                crop[-3:, :, :].reshape(-1, 3),
                crop[:, :3, :].reshape(-1, 3),
                crop[:, -3:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(border.astype(np.float32), axis=0)
        color_dist = np.linalg.norm(crop.astype(np.float32) - bg[None, None, :], axis=2)
        if int(vivid_mask.sum() / 255) >= 40:
            mask = vivid_mask
        else:
            mask = np.where((color_dist > 45) & (sat > 35) & (val > 45), 255, 0).astype(np.uint8)
        if int(mask.sum() / 255) < 40:
            edges = cv2.Canny(gray, 60, 160)
            mask = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8), iterations=1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return np.zeros((size, size), dtype=np.float32)

    bx, by, bw, bh = cv2.boundingRect(coords)
    mask = mask[by : by + bh, bx : bx + bw]
    scale = min((size - 8) / max(1, bw), (size - 8) / max(1, bh))
    new_w = max(1, int(round(bw * scale)))
    new_h = max(1, int(round(bh * scale)))
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    ox = (size - new_w) // 2
    oy = (size - new_h) // 2
    canvas[oy : oy + new_h, ox : ox + new_w] = resized
    return (canvas > 30).astype(np.float32)


def shifted_mask_scores(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float]:
    best_corr = 0.0
    best_iou = 0.0
    height, width = mask_a.shape
    for dy in (-8, -4, 0, 4, 8):
        for dx in (-8, -4, 0, 4, 8):
            shifted = np.zeros_like(mask_b)
            src_x1 = max(0, -dx)
            src_y1 = max(0, -dy)
            src_x2 = min(width, width - dx)
            src_y2 = min(height, height - dy)
            dst_x1 = max(0, dx)
            dst_y1 = max(0, dy)
            dst_x2 = dst_x1 + max(0, src_x2 - src_x1)
            dst_y2 = dst_y1 + max(0, src_y2 - src_y1)
            if src_x2 <= src_x1 or src_y2 <= src_y1:
                continue
            shifted[dst_y1:dst_y2, dst_x1:dst_x2] = mask_b[src_y1:src_y2, src_x1:src_x2]
            inter = float(np.minimum(mask_a, shifted).sum())
            union = float(np.maximum(mask_a, shifted).sum())
            iou = inter / union if union else 0.0
            corr = float(
                (mask_a * shifted).sum()
                / max(1.0, np.sqrt((mask_a * mask_a).sum() * (shifted * shifted).sum()))
            )
            best_corr = max(best_corr, corr)
            best_iou = max(best_iou, iou)
    return best_corr, best_iou


def mask_shape_features(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros(7, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(coords)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = float(binary.sum())
    contour_area = float(sum(cv2.contourArea(contour) for contour in contours))
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    aspect = w / max(1.0, float(h))
    fill = area / max(1.0, float(w * h))
    circularity = (4.0 * np.pi * contour_area / (perimeter * perimeter)) if perimeter > 0 else 0.0
    ys, xs = np.where(binary > 0)
    spread_x = float(np.std(xs) / max(1.0, w))
    spread_y = float(np.std(ys) / max(1.0, h))
    component_count = float(min(5, len(contours))) / 5.0
    return np.array([aspect, fill, circularity, spread_x, spread_y, component_count, area / binary.size], dtype=np.float32)


def infer_header_icon_label(mask: np.ndarray, fallback_label: str) -> str:
    """把顶部白色提示图标先粗分成稳定语义，再和主体图标类别对齐。"""
    aspect, fill, circularity, _spread_x, _spread_y, component_count, _area_ratio = mask_shape_features(mask)
    if circularity >= 0.45 and fill <= 0.45:
        return "ring"
    if component_count <= 0.55 and aspect < 1.35:
        return "box"
    if circularity >= 0.25 and fill >= 0.55:
        return "gears"
    if aspect >= 1.35 and circularity < 0.20:
        return "cigarette"
    return fallback_label


def shape_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.sum() <= 0 or mask_b.sum() <= 0:
        return 0.0
    corr, iou = shifted_mask_scores(mask_a, mask_b)
    features_a = mask_shape_features(mask_a)
    features_b = mask_shape_features(mask_b)
    feature_distance = float(np.linalg.norm(features_a - features_b))
    feature_score = 1.0 / (1.0 + feature_distance)
    return 0.45 * corr + 0.25 * iou + 0.30 * feature_score


def order_items_by_header_shape(
    items: list[dict[str, Any]],
    header_items: list[dict[str, Any]],
    image_rgb: Image.Image,
    icon_model,
    fallback_shape_threshold: float = 0.45,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    icon_items = [item for item in items if item["kind"] == "icon"]
    if len(header_items) < 2 or not icon_items:
        return items, header_items

    sorted_headers = sorted(header_items, key=spatial_order_key)
    candidates = icon_items[:]
    using_fallback_candidates = False
    if not candidates:
        candidates = list(items)
        using_fallback_candidates = True

    header_data: list[dict[str, Any]] = []
    body_masks = {id(item): icon_shape_mask(image_rgb, item["bbox"], header=False) for item in candidates}
    body_embeddings = {
        id(item): icon_embedding_for_box(
            icon_model,
            image_rgb,
            item["bbox"],
            header=False,
        )
        for item in candidates
    }
    for header_item in sorted_headers:
        header_mask = icon_shape_mask(image_rgb, header_item["bbox"], header=True)
        header_embedding = icon_embedding_for_box(
            icon_model,
            image_rgb,
            header_item["bbox"],
            header=True,
        )
        inferred_label = infer_header_icon_label(header_mask, str(header_item.get("label", "")))
        score_map: dict[int, float] = {}
        embedding_scores: dict[int, float] = {}
        for candidate in candidates:
            score = shape_similarity(header_mask, body_masks[id(candidate)])
            embedding_score = float(
                np.clip(
                    (np.dot(header_embedding, body_embeddings[id(candidate)]) + 1.0)
                    / 2.0,
                    0.0,
                    1.0,
                )
            )
            score += 0.20 * embedding_score
            semantic_probability = float(
                header_item.get("class_scores", {}).get(item_name(candidate), 0.0)
            )
            score += 0.8 * semantic_probability
            if item_name(candidate) == inferred_label:
                score += 0.12
            score_map[id(candidate)] = score
            embedding_scores[id(candidate)] = embedding_score
        header_data.append(
            {
                "item": header_item,
                "inferred_label": inferred_label,
                "scores": score_map,
                "embedding_scores": embedding_scores,
            }
        )

    assignments: dict[int, tuple[dict[str, Any], float, list[tuple[float, dict[str, Any]]]]] = {}
    assign_count = min(len(header_data), len(candidates))
    best_total = float("-inf")
    best_score_vector: tuple[float, ...] = ()
    best_candidate_order: tuple[dict[str, Any], ...] = ()
    for candidate_order in itertools.permutations(candidates, assign_count):
        score_vector = tuple(
            header_data[header_index]["scores"][id(candidate)]
            for header_index, candidate in enumerate(candidate_order)
        )
        total = sum(score_vector)
        if total > best_total + 0.01 or (
            abs(total - best_total) <= 0.01 and score_vector > best_score_vector
        ):
            best_total = total
            best_score_vector = score_vector
            best_candidate_order = candidate_order
    for header_index, selected in enumerate(best_candidate_order):
        scored = sorted(
            [
                (header_data[header_index]["scores"][id(candidate)], candidate)
                for candidate in candidates
            ],
            key=lambda pair: pair[0],
            reverse=True,
        )
        assignments[header_index] = (
            selected,
            header_data[header_index]["scores"][id(selected)],
            scored,
        )

    ordered: list[dict[str, Any]] = []
    enriched_header_items: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for header_index, data in enumerate(header_data):
        header_item = data["item"]
        inferred_label = data["inferred_label"]
        assignment = assignments.get(header_index)
        if assignment is None:
            enriched_header_items.append(header_item)
            continue
        selected, score, scored = assignment
        if using_fallback_candidates and score < fallback_shape_threshold:
            matched_label = ""
            matched_center = None
            match_reason = "fallback_shape_below_threshold"
        else:
            ordered.append(selected)
            used_ids.add(id(selected))
            matched_label = item_name(selected)
            matched_center = selected.get("center")
            match_reason = "confidence_first_shape"

        enriched = dict(header_item)
        enriched["inferred_label"] = inferred_label
        enriched["matched_label"] = matched_label
        enriched["matched_center"] = matched_center
        enriched["shape_score"] = score
        enriched["embedding_score"] = data["embedding_scores"][id(selected)]
        enriched["match_reason"] = match_reason
        enriched["shape_candidates"] = [
            {
                "label": item_name(candidate),
                "center": candidate.get("center"),
                "score": candidate_score,
            }
            for candidate_score, candidate in scored
        ]
        enriched_header_items.append(enriched)

    for index, item in enumerate(ordered, start=1):
        item["index"] = index
    return ordered, enriched_header_items


def recognize_header_text(
    bgr: np.ndarray,
    rec_model,
    rec_post,
    header_ratio: float,
    left_end_ratio: float = 0.78,
) -> tuple[str, float]:
    height, width = bgr.shape[:2]
    y2 = max(1, int(height * header_ratio))
    x2 = max(1, int(width * left_end_ratio))
    crop = bgr[:y2, :x2]
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mask = adaptive_foreground_mask(rgb_crop)
    # Ignore thin page borders commonly drawn along the very top edge.
    mask[: max(3, int(mask.shape[0] * 0.08)), :] = 0
    ys, xs = np.where(mask > 0)
    if len(xs) >= 20 and len(ys) >= 20:
        pad_x = max(4, int(width * 0.01))
        pad_y = max(3, int(height * 0.008))
        tx1 = max(0, int(xs.min()) - pad_x)
        ty1 = max(0, int(ys.min()) - pad_y)
        tx2 = min(crop.shape[1], int(xs.max()) + pad_x + 1)
        ty2 = min(crop.shape[0], int(ys.max()) + pad_y + 1)
        if tx2 > tx1 and ty2 > ty1:
            crop = crop[ty1:ty2, tx1:tx2]
    text, score = recognize_char(rec_model, rec_post, crop)
    return text.strip(), score


def recognize_header_instruction_text(
    bgr: np.ndarray,
    rec_model,
    rec_post,
    header_ratio: float,
) -> tuple[str, float]:
    """Recognize the left-side prompt while avoiding right-side target glyphs.

    Real Geetest crops often put colored/distorted target glyphs on the right.
    Feeding the whole header into the recognizer can make the prompt OCR fail
    completely, which then lets those target glyphs be treated as icons. Try a
    few left-only crops first, then fall back to the wider header.
    """
    candidates: list[tuple[str, float]] = []
    for ratio in (0.52, 0.60, 0.68, 0.78, 1.0):
        text, score = recognize_header_text(
            bgr,
            rec_model,
            rec_post,
            header_ratio,
            left_end_ratio=ratio,
        )
        if text:
            candidates.append((text, score))
            if is_semantic_order_instruction(text, score, has_prompt_targets=False):
                return text, score
    if not candidates:
        return "", 0.0
    return max(
        candidates,
        key=lambda row: (
            is_semantic_order_instruction(row[0], row[1], has_prompt_targets=False),
            row[1],
            len(row[0]),
        ),
    )


def recognize_header_target_text(
    bgr: np.ndarray,
    rec_model,
    rec_post,
    header_ratio: float,
    right_start_ratio: float = 0.62,
) -> tuple[str, float]:
    """识别右上角的清晰目标词，避免把汉字连通块误判成提示图标。"""
    height, width = bgr.shape[:2]
    y2 = max(1, int(height * header_ratio))
    start_ratios = sorted(
        {
            max(0.40, right_start_ratio - 0.16),
            max(0.40, right_start_ratio - 0.10),
            max(0.40, right_start_ratio - 0.06),
            right_start_ratio,
            min(0.78, right_start_ratio + 0.04),
        }
    )
    candidates: list[tuple[str, float]] = []
    for start_ratio in start_ratios:
        x1 = min(width - 1, max(0, int(width * start_ratio)))
        text, score = recognize_char(rec_model, rec_post, bgr[:y2, x1:])
        cjk_text = "".join(char for char in text if "\u4e00" <= char <= "\u9fff")
        if cjk_text:
            candidates.append((cjk_text, score))
    if not candidates:
        return "", 0.0

    best_text, best_score = max(candidates, key=lambda row: row[1])

    def contains_in_order(longer: str, shorter: str) -> bool:
        iterator = iter(longer)
        return all(any(value == char for value in iterator) for char in shorter)

    expansions = [
        (text, score)
        for text, score in candidates
        if len(text) > len(best_text)
        and score >= 0.60
        and contains_in_order(text, best_text)
    ]
    if expansions:
        expanded_text, expanded_score = max(expansions, key=lambda row: (len(row[0]), row[1]))
        return expanded_text, max(best_score, expanded_score)
    return best_text, best_score


def recognize_header_target_text_by_boxes(
    bgr: np.ndarray,
    image_rgb: Image.Image,
    rec_model,
    rec_post,
    header_ratio: float,
    right_start_ratio: float,
    max_targets: int,
) -> tuple[str, float, list[dict[str, Any]]]:
    """OCR header targets one visual box at a time, preserving x-order.

    Full-line OCR is useful for content, but for "given order" prompts the
    order is visual: left-to-right target boxes in the prompt header.  Per-box
    OCR keeps that order independent from recognition confidence.
    """
    boxes = detect_header_icon_boxes(
        image_rgb,
        header_ratio=header_ratio,
        right_start_ratio=max(0.40, right_start_ratio - 0.06),
    )
    if not boxes:
        return "", 0.0, []
    if not has_separate_header_target_group(
        image_rgb,
        header_ratio,
        min(box[0] for box in boxes),
    ):
        return "", 0.0, []
    height, width = bgr.shape[:2]
    y_limit = max(1, int(height * header_ratio))
    rows: list[dict[str, Any]] = []
    for box in sorted(boxes, key=lambda row: row[0])[:max_targets]:
        x1, y1, x2, y2 = clamp_box(box, width, height)
        if y1 >= y_limit:
            continue
        y2 = min(y2, y_limit)
        if x2 <= x1 or y2 <= y1:
            continue
        pad_x = max(2, int((x2 - x1) * 0.12))
        pad_y = max(2, int((y2 - y1) * 0.18))
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(width, x2 + pad_x)
        cy2 = min(y_limit, y2 + pad_y)
        text, score = recognize_char(rec_model, rec_post, bgr[cy1:cy2, cx1:cx2])
        cjk_text = "".join(char for char in text if "\u4e00" <= char <= "\u9fff")
        if not cjk_text:
            continue
        rows.append(
            {
                "text": cjk_text[0],
                "raw_text": text,
                "score": float(score),
                "bbox": [cx1, cy1, cx2, cy2],
                "center": [int((cx1 + cx2) / 2), int((cy1 + cy2) / 2)],
            }
        )
    if not rows:
        return "", 0.0, []
    text = "".join(str(row["text"]) for row in rows)
    score = float(np.mean([float(row["score"]) for row in rows]))
    return text, score, rows


def parse_header_text_targets(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", "", text or "")
    if not cleaned:
        return []
    if not any(mark in cleaned for mark in ["：", ":", "、", ",", "，", ";", "；"]):
        return []

    for marker in ["选择：", "选择:", "点击：", "点击:", "依次点击", "顺序选择"]:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[-1]
            break

    cleaned = re.sub(r"^(请|在下图|下图|按照|按|语序|顺序|依次)+", "", cleaned)
    cleaned = cleaned.strip("：:，,。.;；")
    if not cleaned:
        return []

    if "、" in cleaned or "," in cleaned or "，" in cleaned or ";" in cleaned or "；" in cleaned:
        parts = re.split(r"[、,，;；]+", cleaned)
    else:
        parts = list(cleaned)
    return [normalize_target_token(part) for part in parts if part.strip()]


def is_semantic_order_instruction(
    text: str,
    score: float = 1.0,
    *,
    has_prompt_targets: bool = False,
) -> bool:
    """Return True only for explicit semantic-order prompts.

    "依次" and "顺序" are also used by ordinary given-order prompts, so they
    must not trigger semantic reordering.  The reliable semantic signal is
    "语序" (or the equivalent "词序").
    """
    cleaned = re.sub(r"[\s，,。.;；:：、]+", "", text or "")
    if not cleaned:
        return False

    has_semantic_marker = "语序" in cleaned or "词序" in cleaned
    has_action_marker = "点击" in cleaned or "选择" in cleaned or "点选" in cleaned
    return bool(has_semantic_marker and has_action_marker and score >= 0.35)


def has_prompt_header(
    image_rgb: Image.Image,
    header_ratio: float = 0.20,
    right_start_ratio: float = 0.45,
) -> bool:
    arr = np.asarray(image_rgb.convert("RGB"))
    height, width = arr.shape[:2]
    header = arr[: max(1, int(height * header_ratio)), :]
    icon_boxes = detect_header_icon_boxes(image_rgb, header_ratio, right_start_ratio)
    if len(icon_boxes) < 2:
        return False

    # Prompt text is usually on the left side of the same header. Requiring
    # some left-side foreground avoids treating body-only crops as full prompts.
    left = header[:, : max(1, int(width * min(right_start_ratio, 0.55)))]
    left_mask = adaptive_foreground_mask(left)
    foreground = int(left_mask.sum() / 255)
    if foreground < max(30, int(left_mask.size * 0.006)):
        return False

    return True


def has_dark_prompt_header(
    image_rgb: Image.Image,
    header_ratio: float = 0.20,
    max_mean_brightness: float = 115.0,
    max_bright_fraction: float = 0.35,
) -> bool:
    # Kept as a compatibility wrapper for older calls.
    return has_prompt_header(image_rgb, header_ratio)


def analyze_explicit_char_hypothesis(
    target_text: str,
    detections: list[dict[str, Any]],
    bgr: np.ndarray,
    rec_model,
    rec_post,
    header_ratio: float,
    phrase_path: Path,
    extra_phrase_paths: list[Path],
    use_jieba: bool,
    instruction_text: str = "",
) -> tuple[float, str, str]:
    target_chars = list(target_text)
    if len(target_chars) < 2:
        return 0.0, target_text, ""
    height, width = bgr.shape[:2]
    body_detections: list[dict[str, Any]] = []
    for det in detections:
        box = clamp_box(det["bbox"], width, height)
        if is_header_box(box, height, header_ratio):
            continue
        if box_area_ratio(box, width, height) > 0.28:
            continue
        body_detections.append(
            {
                "bbox": box,
                "score": float(det.get("score", 0.0)),
                "kind": det.get("kind"),
            }
        )

    char_detections = [det for det in body_detections if det.get("kind") == "char"]
    if len(char_detections) >= len(target_chars):
        body_detections = char_detections

    clusters: list[list[dict[str, Any]]] = []
    for det in sorted(body_detections, key=lambda row: row["score"], reverse=True):
        group = next(
            (
                cluster
                for cluster in clusters
                if any(
                    box_iou(det["bbox"], other["bbox"]) >= 0.45
                    or box_coverage(det["bbox"], other["bbox"]) >= 0.70
                    or box_coverage(other["bbox"], det["bbox"]) >= 0.70
                    for other in cluster
                )
            ),
            None,
        )
        if group is None:
            clusters.append([det])
        else:
            group.append(det)

    if len(clusters) < len(target_chars):
        return 0.0, target_text, ""

    cluster_scores: list[dict[str, float]] = []
    body_items: list[dict[str, Any]] = []
    for cluster in clusters:
        scores = {char: 0.0 for char in target_chars}
        all_candidate_scores: dict[str, float] = {}
        best_text = ""
        best_rec_score = 0.0
        for det in cluster:
            x1, y1, x2, y2 = det["bbox"]
            text, rec_score, target_scores, candidate_scores = recognize_char_with_targets(
                rec_model,
                rec_post,
                bgr[y1:y2, x1:x2],
                target_chars=target_chars,
                candidate_top_k=256,
            )
            if rec_score > best_rec_score:
                best_text = text
                best_rec_score = rec_score
            for char in target_chars:
                scores[char] = max(scores[char], float(target_scores.get(char, 0.0)))
            for char, probability in candidate_scores.items():
                all_candidate_scores[char] = max(
                    all_candidate_scores.get(char, 0.0), float(probability)
                )
        cluster_scores.append(scores)
        if all_candidate_scores:
            best_text, best_rec_score = max(
                all_candidate_scores.items(), key=lambda row: row[1]
            )
        body_items.append(
            {
                "kind": "char",
                "text": best_text,
                "rec_score": best_rec_score,
                "candidate_scores": all_candidate_scores,
            }
        )

    # Jointly decode individual header glyphs and body glyphs. Both sides may
    # have a wrong top-1 prediction, while their shared candidate is usually
    # much stronger than accidental icon OCR.
    rgb_image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    header_boxes = detect_header_icon_boxes(
        rgb_image,
        header_ratio=header_ratio,
        right_start_ratio=0.45,
    )
    if len(header_boxes) == len(target_chars):
        header_candidates: list[dict[str, float]] = []
        for x1, y1, x2, y2 in sorted(header_boxes, key=lambda box: box[0]):
            _text, _score, _targets, candidates = recognize_char_with_targets(
                rec_model,
                rec_post,
                bgr[y1:y2, x1:x2],
                candidate_top_k=256,
            )
            header_candidates.append(candidates)

        best_joint_log = float("-inf")
        best_joint_visual_log = float("-inf")
        best_joint_text = ""
        frequencies = load_semantic_phrase_frequencies(
            phrase_path,
            max_len=len(target_chars),
            extra_paths=extra_phrase_paths,
            use_jieba=use_jieba,
        )
        for cluster_order in itertools.permutations(range(len(body_items)), len(target_chars)):
            position_candidates: list[list[tuple[str, float]]] = []
            for position, cluster_index in enumerate(cluster_order):
                header_scores = header_candidates[position]
                body_scores = body_items[cluster_index]["candidate_scores"]
                shared = set(header_scores).intersection(body_scores)
                if not shared:
                    break
                ranked_shared = sorted(
                    [
                        (
                            char,
                            float(np.sqrt(header_scores[char] * body_scores[char])),
                        )
                        for char in shared
                    ],
                    key=lambda row: row[1],
                    reverse=True,
                )
                position_candidates.append(ranked_shared[:4])
            if len(position_candidates) != len(target_chars):
                continue
            for combination in itertools.product(*position_candidates):
                chosen_chars = [row[0] for row in combination]
                probabilities = [row[1] for row in combination]
                phrase = "".join(chosen_chars)
                visual_log = float(np.mean(np.log(np.maximum(probabilities, 1e-12))))
                instruction_suffix = normalize_phrase_token(instruction_text)[-len(phrase) :]
                header_prior = sum(
                    0.25
                    for index, char in enumerate(phrase)
                    if index < len(target_text) and char == target_text[index]
                )
                instruction_prior = sum(
                    0.25
                    for index, char in enumerate(phrase)
                    if index < len(instruction_suffix)
                    and char == instruction_suffix[index]
                )
                joint_log = (
                    visual_log
                    + 0.80 * phrase_language_score(phrase, frequencies)
                    + 0.25 * phrase_syntax_score(phrase)
                    + header_prior
                    + instruction_prior
                )
                if joint_log > best_joint_log:
                    best_joint_log = joint_log
                    best_joint_visual_log = visual_log
                    best_joint_text = phrase
        joint_compatibility = (
            float(np.exp(best_joint_visual_log))
            if np.isfinite(best_joint_visual_log)
            else 0.0
        )
        if joint_compatibility >= 0.08:
            return (
                joint_compatibility,
                best_joint_text,
                f"header-body-joint:{best_joint_text}",
            )

    body_phrase_reason = ""
    resolved_text = target_text

    # Cross-correct one weak header character from a strongly matching body
    # assignment. This handles OCR confusions without requiring the whole
    # phrase to exist as a dictionary entry.  It deliberately keeps the header
    # positions fixed: this function is used for "given order" prompts, so the
    # target order is the left-to-right header order, never a dictionary/semantic
    # reordering of the body characters.
    if resolved_text == target_text and len(target_chars) >= 3:
        best_assignment_score = float("-inf")
        best_assignment: tuple[int, ...] = ()
        for cluster_order in itertools.permutations(range(len(clusters)), len(target_chars)):
            probabilities = [
                float(body_items[cluster_index]["candidate_scores"].get(char, 0.0))
                for char, cluster_index in zip(target_chars, cluster_order)
            ]
            score = float(np.mean(np.log(np.maximum(probabilities, 1e-12))))
            if score > best_assignment_score:
                best_assignment_score = score
                best_assignment = cluster_order
        if best_assignment:
            corrected = list(target_text)
            strong_matches = 0
            replacements: list[tuple[int, str]] = []
            for index, (char, cluster_index) in enumerate(zip(target_chars, best_assignment)):
                candidates = body_items[cluster_index]["candidate_scores"]
                target_probability = float(candidates.get(char, 0.0))
                top_char, top_probability = max(candidates.items(), key=lambda row: row[1])
                if top_char == char and top_probability >= 0.10:
                    strong_matches += 1
                elif (
                    top_probability >= 0.35
                    and top_probability >= max(1e-8, target_probability) * 3.0
                ):
                    replacements.append((index, top_char))
            if strong_matches >= len(target_chars) - 1 and len(replacements) == 1:
                replacement_index, replacement_char = replacements[0]
                corrected[replacement_index] = replacement_char
                resolved_text = "".join(corrected)
                body_phrase_reason = (
                    f"header-body-correction:{target_text}->{resolved_text}"
                )

    resolved_chars = list(resolved_text)
    if resolved_text != target_text:
        cluster_scores = [
            {
                char: float(item["candidate_scores"].get(char, 0.0))
                for char in resolved_chars
            }
            for item in body_items
        ]

    best_log_score = float("-inf")
    for cluster_order in itertools.permutations(range(len(clusters)), len(resolved_chars)):
        probabilities = [
            cluster_scores[cluster_index].get(char, 0.0)
            for char, cluster_index in zip(resolved_chars, cluster_order)
        ]
        if any(probability <= 0.0 for probability in probabilities):
            continue
        score = float(np.mean(np.log(np.maximum(probabilities, 1e-12))))
        best_log_score = max(best_log_score, score)
    compatibility = float(np.exp(best_log_score)) if np.isfinite(best_log_score) else 0.0
    return compatibility, resolved_text, body_phrase_reason


def ordered_char_overlap_ratio(reference: str, candidate: str) -> float:
    reference = "".join(ch for ch in reference or "" if "\u4e00" <= ch <= "\u9fff")
    candidate = "".join(ch for ch in candidate or "" if "\u4e00" <= ch <= "\u9fff")
    if not candidate:
        return 0.0
    prev = [0] * (len(candidate) + 1)
    for ref_char in reference:
        current = prev[:]
        for index, cand_char in enumerate(candidate, start=1):
            if ref_char == cand_char:
                current[index] = max(current[index], prev[index - 1] + 1)
            else:
                current[index] = max(current[index], current[index - 1], prev[index])
        prev = current
    return float(prev[-1]) / float(len(candidate))


def resolve_task_spec(
    args,
    rgb_image: Image.Image,
    icon_model,
    icon_labels: list[str],
    bgr: np.ndarray,
    rec_model,
    rec_post,
    detections: list[dict[str, Any]],
) -> TaskSpec:
    if args.target_order:
        return TaskSpec(
            action="explicit_order",
            modality="mixed",
            target_source="manual",
            target_order=args.target_order,
            confidence=1.0,
        )

    has_header = has_prompt_header(rgb_image, args.header_ratio, args.header_right_start)
    instruction_text = ""
    instruction_score = 0.0
    if args.target_source == "auto":
        instruction_text, instruction_score = recognize_header_instruction_text(
            bgr,
            rec_model,
            rec_post,
            args.header_ratio,
        )
        if is_semantic_order_instruction(
            instruction_text,
            instruction_score,
            has_prompt_targets=has_header,
        ):
            return TaskSpec(
                action="semantic_order",
                modality="char",
                target_source="body_chars",
                confidence=instruction_score,
                evidence={
                    "instruction_text": instruction_text,
                    "instruction_score": instruction_score,
                },
            )

    target_text = ""
    target_text_score = 0.0
    boxed_target_text = ""
    boxed_target_text_score = 0.0
    boxed_target_items: list[dict[str, Any]] = []
    char_body_score = 0.0
    resolved_target_text = ""
    body_phrase_reason = ""
    if args.target_source == "auto" and has_header:
        target_text, target_text_score = recognize_header_target_text(
            bgr,
            rec_model,
            rec_post,
            args.header_ratio,
            args.header_text_right_start,
        )
        boxed_target_text, boxed_target_text_score, boxed_target_items = recognize_header_target_text_by_boxes(
            bgr,
            rgb_image,
            rec_model,
            rec_post,
            args.header_ratio,
            args.header_text_right_start,
            args.max_header_text_targets,
        )
        if (
            2 <= len(boxed_target_text) <= args.max_header_text_targets
            and boxed_target_text_score >= 0.35
        ):
            target_text = boxed_target_text
            target_text_score = max(target_text_score, boxed_target_text_score)
        if 2 <= len(target_text) <= args.max_header_text_targets:
            char_body_score, resolved_target_text, body_phrase_reason = analyze_explicit_char_hypothesis(
                target_text,
                detections,
                bgr,
                rec_model,
                rec_post,
                args.header_ratio,
                Path(args.semantic_phrases),
                [Path(path) for path in args.semantic_extra],
                not args.no_jieba_semantic,
                instruction_text,
            )
            if (
                target_text in instruction_text
                and instruction_score >= 0.85
                and target_text_score >= 0.80
            ):
                char_body_score = max(char_body_score, 0.30)
                resolved_target_text = target_text
                if not body_phrase_reason:
                    body_phrase_reason = "instruction-target-consistency"

    target_items: list[dict[str, Any]] = []
    if args.target_source in {"auto", "header-icons"}:
        if has_header:
            target_items = classify_header_icons(
                rgb_image,
                icon_model,
                icon_labels,
                header_ratio=args.header_ratio,
                right_start_ratio=args.header_right_start,
                min_score=args.header_icon_score,
            )
    char_hypothesis_score = target_text_score * char_body_score
    icon_hypothesis_score = min(1.0, len(target_items) / 3.0) if has_header else 0.0
    common_evidence = {
        "instruction_text": instruction_text,
        "instruction_score": instruction_score,
        "header_target_text": target_text,
        "resolved_header_target_text": resolved_target_text,
        "header_target_text_score": target_text_score,
        "boxed_header_target_text": boxed_target_text,
        "boxed_header_target_text_score": boxed_target_text_score,
        "boxed_header_target_items": boxed_target_items,
        "char_body_compatibility": char_body_score,
        "body_phrase_reason": body_phrase_reason,
        "header_icon_count": len(target_items),
        "char_hypothesis_score": char_hypothesis_score,
        "icon_hypothesis_score": icon_hypothesis_score,
    }
    if (
        args.target_source == "auto"
        and is_semantic_order_instruction(
            instruction_text,
            instruction_score,
            has_prompt_targets=len(target_items) >= args.min_header_targets,
        )
    ):
        return TaskSpec(
            action="semantic_order",
            modality="char",
            target_source="body_chars",
            confidence=instruction_score,
            evidence=common_evidence,
        )
    instruction_markers = ("请", "下图", "依次", "点击", "顺序")
    instruction_marker_count = sum(
        marker in instruction_text for marker in instruction_markers
    )
    chinese_click_instruction = (
        instruction_score >= 0.72 and instruction_marker_count >= 2
    )
    prompt_target_overlap = ordered_char_overlap_ratio(
        instruction_text,
        resolved_target_text or target_text,
    )
    strong_header_text = (
        bool(resolved_target_text)
        and chinese_click_instruction
        and target_text_score >= max(args.header_text_score, 0.92)
        and (prompt_target_overlap >= 0.60 or char_body_score >= 0.25)
    )
    common_evidence["instruction_marker_count"] = instruction_marker_count
    common_evidence["chinese_click_instruction"] = chinese_click_instruction
    common_evidence["strong_header_text"] = strong_header_text
    icon_prompt_conflict = (
        len(target_items) >= args.min_header_targets
        and not strong_header_text
        and char_body_score < 0.24
        and prompt_target_overlap < 0.60
        and not (
            instruction_score >= 0.85
            and target_text_score >= 0.85
            and prompt_target_overlap >= 0.60
        )
        and not (
            instruction_score >= 0.85
            and char_body_score >= 0.12
            and str(body_phrase_reason).startswith("header-body-joint:")
        )
    )
    common_evidence["prompt_target_overlap"] = prompt_target_overlap
    common_evidence["icon_prompt_conflict"] = icon_prompt_conflict

    if (
        resolved_target_text
        and (
            chinese_click_instruction
            or (
                instruction_score >= 0.82
                and prompt_target_overlap >= 0.67
                and char_body_score >= 0.25
            )
        )
        and not icon_prompt_conflict
        and (
            target_text_score >= args.header_text_score
            or char_body_score >= 0.25
            or (
                char_body_score >= 0.12
                and any(marker in instruction_text for marker in ("请", "点击", "依次"))
            )
        )
        and (strong_header_text or char_body_score >= 0.08)
    ):
        return TaskSpec(
            action="explicit_order",
            modality="char",
            target_source="header_text",
            target_order=",".join(resolved_target_text),
            confidence=max(
                char_hypothesis_score,
                target_text_score * 0.85 if strong_header_text else 0.0,
            ),
            evidence=common_evidence,
        )

    if len(target_items) >= args.min_header_targets:
        return TaskSpec(
            action="explicit_order",
            modality="icon",
            target_source="header_icons",
            target_order=",".join(item["label"] for item in target_items),
            target_items=target_items,
            confidence=icon_hypothesis_score,
            evidence=common_evidence,
        )

    if args.target_source in {"auto"} and has_header:
        header_text, header_text_score = recognize_header_instruction_text(
            bgr,
            rec_model,
            rec_post,
            args.header_ratio,
        )
        targets = parse_header_text_targets(header_text)
        if targets:
            return TaskSpec(
                action="explicit_order",
                modality="char",
                target_source="header_text",
                target_order=",".join(targets),
                confidence=header_text_score,
                evidence=common_evidence,
            )

    return TaskSpec(
        action="detect_only",
        modality="mixed",
        target_source="none",
        evidence=common_evidence,
    )


def draw_results(image_path: Path, items: list[dict[str, Any]], output_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("simhei.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    colors = {"char": (255, 0, 180), "icon": (255, 150, 60)}
    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        color = colors.get(item["kind"], (0, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{item['kind']}:{item.get('text') or item.get('label')} {item.get('final_score', 0):.2f}"
        draw.text((x1, max(0, y1 - 22)), text, fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def predict_header_intent_shadow(
    payload: dict[str, Any],
    model_path: Path,
    *,
    threshold: float,
    min_margin: float,
) -> dict[str, Any]:
    """Run the optional lightweight header-intent model in shadow mode.

    This deliberately does not mutate the main inference result. It only records
    whether the learned intent model would recommend a task/order override.
    """

    tools_dir = GSXT_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    try:
        from train_header_intent_model import extract_features, matrix_from, softmax
    except Exception as exc:  # pragma: no cover - optional diagnostic path
        return {"enabled": False, "error": f"cannot import header intent tools: {exc}"}

    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        feature_names = list(model["feature_names"])
        features = extract_features(payload)
        x = matrix_from([features], feature_names)
        mean = np.asarray(model["mean"], dtype=np.float64)
        std = np.asarray(model["std"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        bias = np.asarray(model["bias"], dtype=np.float64)
        probs = softmax(((x - mean) / std) @ weights + bias)[0]
        classes = list(model["classes"])
    except Exception as exc:  # pragma: no cover - optional diagnostic path
        return {"enabled": False, "error": f"cannot run header intent model: {exc}"}

    order = np.argsort(-probs)
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best
    label = classes[best]
    if label.endswith("_given_order"):
        predicted_task_type = label[: -len("_given_order")]
        predicted_order_mode = "given_order"
    elif label.endswith("_semantic_order"):
        predicted_task_type = label[: -len("_semantic_order")]
        predicted_order_mode = "semantic_order"
    else:
        predicted_task_type = label
        predicted_order_mode = "unknown"

    task_spec = payload.get("task_spec") or {}
    rule_task_type = str(payload.get("task_type") or task_spec.get("modality") or "")
    rule_order_mode = (
        "semantic_order" if task_spec.get("action") == "semantic_order" else "given_order"
    )
    confidence = float(probs[best])
    margin = float(probs[best] - probs[second]) if len(order) > 1 else confidence
    disagrees = (
        predicted_task_type != rule_task_type
        or predicted_order_mode != rule_order_mode
    )
    # The rule path tends to over-trust header icon classification on real
    # Geetest crops: distorted black Chinese title glyphs can look like icons,
    # while the header OCR may fail. In that specific conflict direction, let a
    # moderately confident learned "char" intent veto the icon rule. Keep the
    # original stricter threshold for icon-over-char and same-modality changes.
    effective_threshold = threshold
    effective_margin = min_margin
    if rule_task_type == "icon" and predicted_task_type == "char":
        effective_threshold = min(effective_threshold, 0.55)
        effective_margin = min(effective_margin, 0.10)
    override_recommended = (
        confidence >= effective_threshold
        and margin >= effective_margin
        and disagrees
    )

    return {
        "enabled": True,
        "model": str(model_path),
        "label": label,
        "task_type": predicted_task_type,
        "order_mode": predicted_order_mode,
        "confidence": confidence,
        "margin": margin,
        "probabilities": {classes[idx]: float(probs[idx]) for idx in order},
        "rule_task_type": rule_task_type,
        "rule_order_mode": rule_order_mode,
        "disagrees_with_rule": disagrees,
        "override_threshold": threshold,
        "override_margin": min_margin,
        "effective_override_threshold": effective_threshold,
        "effective_override_margin": effective_margin,
        "override_recommended": override_recommended,
    }


def apply_header_intent_override(
    *,
    intent: dict[str, Any],
    task_spec: TaskSpec,
    raw_results: list[dict[str, Any]],
    merged_results: list[dict[str, Any]],
    rgb_image: Image.Image,
    icon_model,
    target_items: list[dict[str, Any]],
    phrase_path: Path,
    extra_phrase_paths: list[Path],
    use_jieba: bool,
) -> tuple[TaskSpec, list[dict[str, Any]], list[dict[str, Any]], str, str]:
    if not intent.get("override_recommended"):
        return task_spec, [], target_items, task_spec.target_order, task_spec.target_source

    predicted_task_type = str(intent.get("task_type") or "")
    predicted_order_mode = str(intent.get("order_mode") or "")
    evidence = dict(task_spec.evidence or {})
    evidence["header_intent_override"] = {
        "label": intent.get("label"),
        "confidence": intent.get("confidence"),
        "margin": intent.get("margin"),
        "previous_modality": task_spec.modality,
        "previous_action": task_spec.action,
        "previous_target_source": task_spec.target_source,
    }

    if predicted_task_type == "char" and predicted_order_mode == "semantic_order":
        char_candidates = [item for item in raw_results if item.get("kind") == "char"]
        candidate_results = char_candidates if len(char_candidates) >= 2 else merged_results
        semantic_order, semantic_reason = infer_semantic_target_order(
            candidate_results,
            phrase_path,
            extra_phrase_paths=extra_phrase_paths,
            use_jieba=use_jieba,
        )
        final_results = order_items_by_target(
            candidate_results,
            semantic_order,
            normalize_aliases=False,
            keep_unmatched=False,
        )
        if len(final_results) > 3:
            final_results = final_results[:3]
        if len(final_results) < 3:
            return task_spec, [], target_items, task_spec.target_order, task_spec.target_source
        target_order = ",".join(str(item.get("text") or "") for item in final_results)
        new_spec = TaskSpec(
            action="semantic_order",
            modality="char",
            target_source=semantic_reason or "header_intent_model",
            target_order=target_order,
            confidence=float(intent.get("confidence") or 0.0),
            evidence=evidence,
        )
        return new_spec, final_results, [], target_order, new_spec.target_source

    if predicted_task_type == "char" and predicted_order_mode == "given_order":
        target_text = str(
            evidence.get("resolved_header_target_text")
            or evidence.get("header_target_text")
            or ""
        )
        target_text = "".join(ch for ch in target_text if "\u4e00" <= ch <= "\u9fff")
        if len(target_text) >= 2:
            char_candidates = [item for item in raw_results if item.get("kind") == "char"]
            candidate_results = (
                char_candidates
                if len(char_candidates) >= min(len(target_text), 2)
                else merged_results
            )
            final_results = order_items_by_target(
                candidate_results,
                ",".join(target_text),
                normalize_aliases=False,
                keep_unmatched=False,
            )
            if len(final_results) < min(len(target_text), 3):
                return task_spec, [], target_items, task_spec.target_order, task_spec.target_source
            new_spec = TaskSpec(
                action="explicit_order",
                modality="char",
                target_source="header_intent_model",
                target_order=",".join(target_text),
                confidence=float(intent.get("confidence") or 0.0),
                evidence=evidence,
            )
            return new_spec, final_results, [], new_spec.target_order, new_spec.target_source

    if (
        predicted_task_type == "icon"
        and predicted_order_mode == "given_order"
        and target_items
        and not (task_spec.modality == "char" and task_spec.action == "semantic_order")
    ):
        header_mode_results = restore_icon_candidates_for_header_mode(merged_results)
        header_mode_results = promote_char_candidates_for_header_mode(
            header_mode_results,
            rgb_image,
            icon_model,
            target_items,
        )
        final_results, new_target_items = order_items_by_header_shape(
            header_mode_results,
            target_items,
            rgb_image,
            icon_model,
        )
        matched_order = [str(item.get("label") or "") for item in final_results]
        new_spec = TaskSpec(
            action="explicit_order",
            modality="icon",
            target_source="header_intent_model",
            target_order=",".join(matched_order),
            target_items=new_target_items,
            confidence=float(intent.get("confidence") or 0.0),
            evidence=evidence,
        )
        return new_spec, final_results, new_target_items, new_spec.target_order, new_spec.target_source

    return task_spec, [], target_items, task_spec.target_order, task_spec.target_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic mixed detector/OCR/icon inference.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--det-config", default=str(DEFAULT_DET_CONFIG))
    parser.add_argument("--det-weights", default=str(DEFAULT_DET_WEIGHTS))
    parser.add_argument("--det-dataset", default=str(DEFAULT_DET_DATASET))
    parser.add_argument("--rec-config", default=str(DEFAULT_REC_CONFIG))
    parser.add_argument("--rec-weights", default=str(DEFAULT_REC_WEIGHTS))
    parser.add_argument("--icon-weights", default=str(DEFAULT_ICON_WEIGHTS))
    parser.add_argument("--icon-labels", default=str(DEFAULT_ICON_LABELS))
    parser.add_argument("--icon-model", default="mobilenet_v3_large")
    parser.add_argument("--semantic-phrases", default=str(DEFAULT_SEMANTIC_PHRASES))
    parser.add_argument(
        "--semantic-extra",
        action="append",
        default=[],
        help="Extra semantic phrase file/dir. Supports THUOCL/jieba userdict style: word freq tag.",
    )
    parser.add_argument("--no-jieba-semantic", action="store_true", help="Disable optional jieba dictionary lookup.")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--overlap-iou", type=float, default=0.45)
    parser.add_argument("--char-rec-threshold", type=float, default=0.90)
    parser.add_argument(
        "--max-body-box-area-ratio",
        type=float,
        default=0.28,
        help="Filter implausibly large body detections before OCR/classification.",
    )
    parser.add_argument("--target-order", default="", help="Optional final order, e.g. 古罗马 or 伞,路,暂停")
    parser.add_argument(
        "--target-source",
        choices=["auto", "manual", "header-icons", "none"],
        default="auto",
        help="auto uses --target-order, then header icons, then header text when a dark prompt header exists",
    )
    parser.add_argument("--header-ratio", type=float, default=0.20)
    parser.add_argument("--header-right-start", type=float, default=0.45)
    parser.add_argument("--header-text-right-start", type=float, default=0.62)
    parser.add_argument(
        "--header-text-score",
        type=float,
        default=0.88,
        help="Minimum OCR confidence for treating the right header as a Chinese target word.",
    )
    parser.add_argument("--max-header-text-targets", type=int, default=8)
    parser.add_argument("--header-icon-score", type=float, default=0.02)
    parser.add_argument("--min-header-targets", type=int, default=2)
    parser.add_argument(
        "--header-intent-model",
        default="",
        help="Optional lightweight header-intent model JSON.",
    )
    parser.add_argument(
        "--header-intent-apply",
        action="store_true",
        help="Apply high-confidence lightweight header-intent overrides.",
    )
    parser.add_argument("--header-intent-threshold", type=float, default=0.75)
    parser.add_argument("--header-intent-margin", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    use_gpu = not args.cpu
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detections = run_paddledet(
        image=image_path,
        output_dir=output_dir,
        config=Path(args.det_config),
        weights=Path(args.det_weights),
        dataset_dir=Path(args.det_dataset),
        threshold=args.threshold,
        use_gpu=use_gpu,
    )

    bgr = cv2.imread(as_posix(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    height, width = bgr.shape[:2]
    rgb_image = Image.open(image_path).convert("RGB")
    prompt_header_present = has_prompt_header(rgb_image, args.header_ratio, args.header_right_start)

    rec_model, rec_post = load_ocr_model(Path(args.rec_config), Path(args.rec_weights), use_gpu)
    icon_model, icon_labels = load_icon_model(Path(args.icon_weights), Path(args.icon_labels), args.icon_model, use_gpu)
    task_spec = resolve_task_spec(
        args,
        rgb_image,
        icon_model,
        icon_labels,
        bgr,
        rec_model,
        rec_post,
        detections,
    )
    resolved_target_order = task_spec.target_order
    target_items = task_spec.target_items
    target_source_resolved = task_spec.target_source
    text_target_task = task_spec.action == "explicit_order" and task_spec.modality == "char"
    semantic_instruction_task = task_spec.action == "semantic_order"
    char_task = task_spec.modality == "char"
    target_chars = parse_target_order(resolved_target_order) if text_target_task else []

    results: list[dict[str, Any]] = []
    for index, det in enumerate(detections, start=1):
        box = clamp_box(det["bbox"], width, height)
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        if box_area_ratio(box, width, height) > args.max_body_box_area_ratio:
            continue
        if (prompt_header_present or semantic_instruction_task) and is_header_box(
            box, height, args.header_ratio
        ):
            continue
        item = {
            "index": index,
            "kind": "char" if char_task else det["kind"],
            "detected_kind": det["kind"],
            "det_score": det["score"],
            "bbox": box,
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
        }
        if char_task or det["kind"] == "char":
            text, score, target_scores, candidate_scores = recognize_char_with_targets(
                rec_model,
                rec_post,
                bgr[y1:y2, x1:x2],
                target_chars=target_chars,
            )
            if semantic_instruction_task:
                target_scores = candidate_scores
            item.update(
                {
                    "text": text,
                    "rec_score": score,
                    "target_scores": target_scores,
                    "candidate_scores": candidate_scores,
                    "final_score": det["score"] * score,
                }
            )
        else:
            label, score = classify_icon(icon_model, icon_labels, rgb_image.crop((x1, y1, x2, y2)))
            item.update({"label": label, "cls_score": score, "final_score": det["score"] * score})
        results.append(item)

    if char_task and len(results) >= 4:
        areas = sorted(
            max(1, int(item["bbox"][2] - item["bbox"][0])) * max(1, int(item["bbox"][3] - item["bbox"][1]))
            for item in results
        )
        median_area = float(np.median(areas))
        results = [
            item
            for item in results
            if (
                max(1, int(item["bbox"][2] - item["bbox"][0]))
                * max(1, int(item["bbox"][3] - item["bbox"][1]))
            )
            <= median_area * 2.5
        ]

    results = suppress_bridge_boxes(results)
    prefer_kind = "char" if char_task else None
    merged_results = merge_overlapping_items(
        results,
        overlap_iou=args.overlap_iou,
        char_rec_threshold=args.char_rec_threshold,
        prefer_kind=prefer_kind,
    )
    if (
        task_spec.target_source == "none"
        and task_spec.modality == "mixed"
        and is_semantic_order_instruction(
            str((task_spec.evidence or {}).get("instruction_text") or ""),
            float((task_spec.evidence or {}).get("instruction_score") or 0.0),
            has_prompt_targets=False,
        )
    ):
        char_only_results = [
            item
            for item in merged_results
            if item.get("kind") == "char" and is_single_cjk(str(item.get("text", "")))
        ]
        if len(char_only_results) == 3 and not target_items:
            semantic_order, semantic_reason = infer_semantic_target_order(
                char_only_results,
                Path(args.semantic_phrases),
                extra_phrase_paths=[Path(path) for path in args.semantic_extra],
                use_jieba=not args.no_jieba_semantic,
            )
            if semantic_order and semantic_reason.startswith("semantic-joint:"):
                task_spec = TaskSpec(
                    action="semantic_order",
                    modality="char",
                    target_source="semantic_fallback_no_header_targets",
                    target_order=semantic_order,
                    confidence=max(float(task_spec.confidence or 0.0), 0.50),
                    evidence={
                        **(task_spec.evidence or {}),
                        "semantic_fallback_no_header_targets": semantic_reason,
                    },
                )
                merged_results = char_only_results
                resolved_target_order = semantic_order
                target_source_resolved = task_spec.target_source
                char_task = True
                text_target_task = False
                semantic_instruction_task = True
    if task_spec.modality == "icon" and not any(item["kind"] == "icon" for item in merged_results):
        resolved_target_order = ""
        target_source_resolved = "header_icons_ignored_no_body_icons"
        target_items = []
    if task_spec.action == "semantic_order" and not resolved_target_order:
        semantic_order, semantic_reason = infer_semantic_target_order(
            merged_results,
            Path(args.semantic_phrases),
            extra_phrase_paths=[Path(path) for path in args.semantic_extra],
            use_jieba=not args.no_jieba_semantic,
        )
        if semantic_order:
            resolved_target_order = semantic_order
            target_source_resolved = semantic_reason
            task_spec.target_order = semantic_order
    char_body_fallback = infer_char_body_fallback_for_icon_rule(
        task_spec=task_spec,
        raw_results=results,
        overlap_iou=args.overlap_iou,
        char_rec_threshold=args.char_rec_threshold,
    )
    if char_body_fallback is not None:
        task_spec, merged_results, resolved_target_order, target_source_resolved = char_body_fallback
        target_items = []
        char_task = True
        text_target_task = True
        semantic_instruction_task = False
    if task_spec.modality == "icon":
        header_mode_results = restore_icon_candidates_for_header_mode(merged_results)
        header_mode_results = promote_char_candidates_for_header_mode(
            header_mode_results,
            rgb_image,
            icon_model,
            icon_labels,
        )
        final_results, target_items = order_items_by_header_shape(
            header_mode_results,
            target_items,
            rgb_image,
            icon_model,
        )
        matched_order = [item.get("matched_label", "") for item in target_items if item.get("matched_label")]
        if matched_order:
            resolved_target_order = ",".join(matched_order)
        reinterpreted_char = reinterpret_icon_results_as_char_if_cjk_evidence(
            task_spec=task_spec,
            final_results=final_results,
            target_source_resolved=target_source_resolved,
        )
        if reinterpreted_char is not None:
            task_spec, final_results, resolved_target_order, target_source_resolved = reinterpreted_char
            target_items = []
            char_task = True
            text_target_task = True
            semantic_instruction_task = False
    else:
        final_results = order_items_by_target(
            merged_results,
            resolved_target_order,
            normalize_aliases=not char_task,
            keep_unmatched=not char_task,
        )

    if task_spec.action == "semantic_order" and task_spec.modality == "char":
        semantic_order, semantic_reason = infer_semantic_target_order(
            final_results,
            Path(args.semantic_phrases),
            extra_phrase_paths=[Path(path) for path in args.semantic_extra],
            use_jieba=not args.no_jieba_semantic,
        )
        if semantic_order and semantic_order != resolved_target_order.replace(",", ""):
            resolved_target_order = semantic_order
            target_source_resolved = semantic_reason
            task_spec.target_order = semantic_order
            task_spec.target_source = semantic_reason
            final_results = order_items_by_target(
                final_results,
                resolved_target_order,
                normalize_aliases=False,
                keep_unmatched=False,
            )

    if semantic_instruction_task and len(final_results) > 3:
        final_results = final_results[:3]
        resolved_target_order = ",".join(
            str(item.get("text") or item.get("label") or "") for item in final_results
        )

    task_type = task_spec.modality

    payload = {
        "image": as_posix(image_path),
        "use_gpu": use_gpu,
        "task_type": task_type,
        "task_spec": asdict(task_spec),
        "merge_settings": {
            "overlap_iou": args.overlap_iou,
            "char_rec_threshold": args.char_rec_threshold,
            "target_order": args.target_order,
            "target_source": args.target_source,
            "target_source_resolved": target_source_resolved,
            "resolved_target_order": resolved_target_order,
        },
        "target_items": target_items,
        "raw_items": results,
        "merged_items": merged_results,
        "items": final_results,
    }
    if args.header_intent_model:
        payload["header_intent_model"] = predict_header_intent_shadow(
            payload,
            Path(args.header_intent_model),
            threshold=args.header_intent_threshold,
            min_margin=args.header_intent_margin,
        )
        if args.header_intent_apply and payload["header_intent_model"].get("override_recommended"):
            (
                task_spec,
                override_results,
                target_items,
                resolved_target_order,
                target_source_resolved,
            ) = apply_header_intent_override(
                intent=payload["header_intent_model"],
                task_spec=task_spec,
                raw_results=results,
                merged_results=merged_results,
                rgb_image=rgb_image,
                icon_model=icon_model,
                target_items=target_items,
                phrase_path=Path(args.semantic_phrases),
                extra_phrase_paths=[Path(path) for path in args.semantic_extra],
                use_jieba=not args.no_jieba_semantic,
            )
            if override_results:
                final_results = override_results
                if task_spec.action == "semantic_order" and task_spec.modality == "char":
                    semantic_order, semantic_reason = infer_semantic_target_order(
                        final_results,
                        Path(args.semantic_phrases),
                        extra_phrase_paths=[Path(path) for path in args.semantic_extra],
                        use_jieba=not args.no_jieba_semantic,
                    )
                    if semantic_order and semantic_order != resolved_target_order.replace(",", ""):
                        resolved_target_order = semantic_order
                        target_source_resolved = semantic_reason
                        task_spec.target_order = semantic_order
                        task_spec.target_source = semantic_reason
                        final_results = order_items_by_target(
                            final_results,
                            resolved_target_order,
                            normalize_aliases=False,
                            keep_unmatched=False,
                        )
                task_type = task_spec.modality
                payload["task_type"] = task_type
                payload["task_spec"] = asdict(task_spec)
                payload["merge_settings"]["target_source_resolved"] = target_source_resolved
                payload["merge_settings"]["resolved_target_order"] = resolved_target_order
                payload["target_items"] = target_items
                payload["items"] = final_results
                payload["header_intent_model"]["override_applied"] = True
            else:
                payload["header_intent_model"]["override_applied"] = False
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    visual_path = output_dir / "visual" / image_path.name
    draw_results(image_path, final_results, visual_path)

    print(f"raw_items={len(results)} merged_items={len(merged_results)} final_items={len(final_results)}")
    print(f"task_type={task_type}")
    print(f"target_source={args.target_source} resolved={target_source_resolved}")
    if resolved_target_order:
        print(f"resolved_target_order={resolved_target_order}")
    if target_items:
        print(
            "recognized_prompt_order="
            + " -> ".join(str(item.get("matched_label") or item.get("label")) for item in target_items)
        )
        print(
            "target_items="
            + ", ".join(
                f"{item.get('label')}({item.get('cls_score', 0):.3f})"
                + (
                    f"=>{item.get('matched_label')}@{item.get('shape_score', 0):.3f}"
                    if item.get("matched_label")
                    else ""
                )
                for item in target_items
            )
        )
    elif target_source_resolved != "none":
        print("recognized_prompt_order=<none>")
    for item in final_results:
        name = item.get("text") if item["kind"] == "char" else item.get("label")
        print(
            f"{item['index']:>2}. {item['kind']} center={item['center']} "
            f"name={name} det={item['det_score']:.3f} final={item.get('final_score', 0):.3f} "
            f"merge={item.get('merge_reason')}"
        )
    print(f"Saved JSON: {result_path}")
    print(f"Saved visual: {visual_path}")


if __name__ == "__main__":
    main()
