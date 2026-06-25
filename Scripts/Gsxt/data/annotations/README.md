# GSXT annotation data

This directory contains the simplified annotation set for the bundled 200-image GSXT
evaluation corpus.

## Files

- `gsxt_200_simple_annotation.xlsx`: editable spreadsheet version.
- `gsxt_200_simple_annotation_filled.csv`: CSV export of the same annotations.
- `gsxt_200_simple_annotation.json`: validated JSON used by tooling.
- `ANNOTATION_GUIDE.md`: short annotation guide.

The corresponding images are stored in `Scripts/Gsxt/data/images`.

## Schema

Each row describes one image:

- `image`: image filename.
- `usable`: whether the image should be used in evaluation/training experiments.
- `task_type`: `char` or `icon`.
- `instruction_text`: one of the known prompt templates.
- `order_mode`: `given_order` for prompt/header order, or `semantic_order` for semantic
  text order.
- `target1`, `target2`, `target3`: the expected ordered targets.
- `note`: optional clarification for ambiguous icons or labels.

The target columns are already ordered: `target1 -> target2 -> target3`.
