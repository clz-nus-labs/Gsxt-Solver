# GSXT Solver

GSXT Solver is a PaddlePaddle-based image understanding pipeline for structured
character and icon ordering tasks. It detects candidate regions, recognizes Chinese
characters or icons, interprets the instruction area, and returns the targets in the
requested order.

The package provides:

- automatic task-type inference for character and icon tasks;
- lightweight header-intent arbitration for noisy prompt OCR;
- character recognition with semantic lexicon decoding;
- icon classification plus prompt-to-body shape and embedding matching;
- a Python API and command-line interface;
- automatic model download, verification, and directory assembly;
- 50 bundled evaluation images for regression testing.

## Supported use cases

GSXT Solver is designed for images containing:

- a text instruction or visual prompt near the top of the image;
- Chinese-character or icon candidates in the main image area;
- an explicit target order or a semantic ordering instruction.

It is a domain-specific pipeline rather than a general-purpose OCR or object-detection
library. Performance is best on layouts and visual styles similar to the supplied
evaluation images.

## Requirements

- Windows
- Python 3.10
- PaddlePaddle 3.2
- Git
- GitHub CLI (`gh`) only if you want to contribute through GitHub workflows

For GPU inference, install a PaddlePaddle build compatible with the host's CUDA and
CUDNN versions.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/clz-nus-labs/Gsxt-Solver.git
cd Gsxt-Solver
```

### 2. Create an environment

CPU example:

```powershell
conda create -n gsxt_solver python=3.10 -y
conda activate gsxt_solver

python -m pip install paddlepaddle==3.2.0 `
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -e ".[inference]"
```

For GPU inference, replace the CPU PaddlePaddle package with the build matching the
host's CUDA/CUDNN environment.

The inference extra installs `numpy<2` and `opencv-python<=4.6.0`, which are required by
the pinned PaddleDetection/imgaug runtime.
The source dependency setup scripts also apply `Scripts/Gsxt/runtime-constraints.txt`
so PaddleOCR cannot replace the compatible NumPy/OpenCV runtime with NumPy 2.x wheels.

### 3. Install runtime source dependencies

```powershell
powershell -ExecutionPolicy Bypass `
  -File .\Scripts\Gsxt\training\setup_paddledetection_repo.ps1 `
  -EnvName gsxt_solver `
  -DirectPython

powershell -ExecutionPolicy Bypass `
  -File .\Scripts\Gsxt\training\setup_paddleocr_repo.ps1 `
  -EnvName gsxt_solver `
  -DirectPython
```

## Download and assemble the models

The model release contains the detector, character recognizer, icon classifier,
recognition configuration, and icon label list.

```powershell
gsxt-models `
  --destination .\models\gsxt-models-v0.1.0 `
  --release-base-url https://github.com/clz-nus-labs/Gsxt-Solver/releases/download/models-v0.1.0
```

`gsxt-models` automatically:

1. downloads every required model component;
2. verifies each file using SHA-256;
3. creates the detector, recognizer, and icon-classifier directories;
4. generates the minimal detector metadata required at runtime.

No manual weight concatenation is required.

The public model release does not require a GitHub token.

## Command-line usage

The CLI has two output modes:

- `standard` (default): stable probability-free business output;
- `debug`: complete detector, OCR, classifier, candidate, score, and merge diagnostics.

### Standard mode

```powershell
gsxt-solve .\tests\fixtures\test10.png `
  --project-root . `
  --model-dir .\models\gsxt-models-v0.1.0 `
  --mode standard `
  --cpu
```

Example output:

```json
{
  "schema_version": "1.0",
  "success": true,
  "image": "test10.png",
  "task": {
    "action": "explicit_order",
    "type": "icon"
  },
  "result": {
    "count": 3,
    "sequence": ["flamingo", "grasshopper", "air horn"],
    "points": [
      {"x": 285, "y": 303},
      {"x": 239, "y": 212},
      {"x": 363, "y": 156}
    ],
    "items": [
      {
        "index": 1,
        "type": "icon",
        "value": "flamingo",
        "center": {"x": 285, "y": 303},
        "bbox": {"left": 244, "top": 262, "right": 327, "bottom": 344}
      }
    ]
  }
}
```

Standard mode does not expose model probabilities or intermediate candidates and does
not save files by default. Add `--save-result`, `--save-visual`, and optionally
`--output-dir .\runs\test10` when files are required. If `--output-dir` is omitted
while saving is enabled, files are written under `runs/<image-name>`.
If inference fails, standard mode returns `success: false` with a short `error` object
instead of exposing backend logs or a diagnostic traceback.

### Debug mode

```powershell
gsxt-solve .\tests\fixtures\test10.png `
  --project-root . `
  --model-dir .\models\gsxt-models-v0.1.0 `
  --output-dir .\runs\test10-debug `
  --mode debug `
  --cpu
```

Debug mode returns the complete backend payload, including task evidence, raw and merged
candidates, detection/OCR/classification scores, alternative candidates, suppression
details, and runtime logs. It saves `result.json` and the annotated image by default.
Use `--no-save-result` or `--no-save-visual` to disable either file.

Remove `--cpu` to use the configured GPU environment.

The Python API and CLI enable the bundled lightweight header-intent model by
default. It only applies high-confidence task/order overrides and can be disabled
with `--no-header-intent` when comparing against the rule-only pipeline.

When saving is enabled, the output directory contains:

- `result.json`: standard or debug JSON according to the selected mode;
- `visual/<image-name>`: visualization of the final selected items.

You can provide a known order manually:

```powershell
gsxt-solve .\example.png `
  --project-root . `
  --model-dir .\models\gsxt-models-v0.1.0 `
  --target-order "古,罗,马" `
  --mode standard `
  --output-dir .\runs\manual-order `
  --save-result `
  --save-visual
```

## Python API

```python
from pathlib import Path

from gsxt_solver import Solver

project_root = Path(r"D:\path\to\Gsxt-Solver")
model_dir = project_root / "models" / "gsxt-models-v0.1.0"

solver = Solver.from_bundle(
    project_root,
    model_dir,
    use_gpu=False,
)

# Standard mode: stable, probability-free output.
result = solver.predict(
    project_root / "tests" / "fixtures" / "test10.png",
)

print(result["task"])
print(result["result"]["sequence"])
print(result["result"]["points"])

for item in result["result"]["items"]:
    print(item["center"], item["value"])
```

Standard mode only returns the result by default. Enable optional files explicitly:

```python
result = solver.predict(
    project_root / "tests" / "fixtures" / "test10.png",
    output_dir=project_root / "runs" / "api-test10",
    save_result=True,
    save_visual=True,
)
```

`output_dir` is optional. When omitted with saving enabled, the default directory is
`runs/<image-name>`:

```python
result = solver.predict(
    project_root / "tests" / "fixtures" / "test10.png",
    save_result=True,
    save_visual=True,
)
```

Debug mode uses the same solver instance:

```python
debug_result = solver.debug(
    project_root / "tests" / "fixtures" / "test10.png",
    output_dir=project_root / "runs" / "api-test10-debug",
)

print(debug_result["task_spec"])
print(debug_result["raw_items"])
print(debug_result["merged_items"])
```

Debug mode saves both files by default. It can also run without writing anything:

```python
debug_result = solver.debug(
    project_root / "tests" / "fixtures" / "test10.png",
    save_result=False,
    save_visual=False,
)
```

`solve()` is also available as an explicit mode dispatcher:

```python
standard_result = solver.solve("example.png", mode="standard")
debug_result = solver.solve("example.png", mode="debug")
```

Standard result fields:

| Field | Description |
| --- | --- |
| `schema_version` | Public result schema version |
| `success` | Whether inference completed successfully |
| `image` | Input image filename |
| `task.action` | `explicit_order`, `semantic_order`, or `detect_only` |
| `task.type` | `char`, `icon`, or `mixed` |
| `result.sequence` | Final ordered character/icon values |
| `result.points` | Ordered center points |
| `result.items` | Ordered values, centers, and bounding boxes |

## Run the 50-image evaluation set

```powershell
gsxt-test-suite `
  --project-root . `
  --model-dir .\models\gsxt-models-v0.1.0 `
  --fixtures .\tests\fixtures `
  --output-dir .\runs\test-suite `
  --cpu
```

The command creates one result directory per image and writes the combined report to:

```text
runs/test-suite/summary.json
```

## Current limitations

- Visually similar characters and icons may produce ambiguous top candidates.
- Unusual prompt layouts can affect task-type and target-order inference.
- Small, overlapping, or heavily distorted regions can cause missed or duplicate detections.
- Semantic ordering quality depends on recognition candidates and lexicon coverage.
- The 50 bundled images are regression inputs; authoritative ground-truth annotations are
  not yet included.

See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for the planned evaluation, training, and
decoding improvements.

## License

Source code is licensed under Apache License 2.0. Model and third-party information is
documented in [MODEL_CARD.md](MODEL_CARD.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
