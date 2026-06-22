# GSXT Solver

GSXT Solver is a PaddlePaddle-based image understanding pipeline for structured
character and icon ordering tasks. It detects candidate regions, recognizes Chinese
characters or icons, interprets the instruction area, and returns the targets in the
requested order.

The package provides:

- automatic task-type inference for character and icon tasks;
- character recognition with semantic lexicon decoding;
- icon classification and prompt-to-body matching;
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
- GitHub CLI (`gh`) when accessing the private repository and model release

For GPU inference, install a PaddlePaddle build compatible with the host's CUDA and
CUDNN versions.

## Installation

### 1. Clone the repository

For the current private repository, first authenticate an account with access:

```powershell
gh auth login
gh auth setup-git
gh repo clone clz-nus/Gsxt-Solver
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

For the private repository:

```powershell
$env:GH_TOKEN = gh auth token

gsxt-models `
  --destination .\models\gsxt-models-v0.1.0 `
  --release-base-url https://github.com/clz-nus/Gsxt-Solver/releases/download/models-v0.1.0
```

`gsxt-models` automatically:

1. downloads every required model component;
2. verifies each file using SHA-256;
3. creates the detector, recognizer, and icon-classifier directories;
4. generates the minimal detector metadata required at runtime.

No manual weight concatenation is required.

If the repository is made public later, the same command works without `GH_TOKEN`.

## Command-line usage

Run one image:

```powershell
gsxt-solve .\tests\fixtures\test10.png `
  --project-root . `
  --model-dir .\models\gsxt-models-v0.1.0 `
  --output-dir .\runs\test10 `
  --cpu
```

Remove `--cpu` to use the configured GPU environment.

The output directory contains:

- `result.json`: task interpretation, detected candidates, resolved order, and final items;
- `visual/<image-name>`: visualization of the final selected items.

You can provide a known order manually:

```powershell
gsxt-solve .\example.png `
  --project-root . `
  --model-dir .\models\gsxt-models-v0.1.0 `
  --target-order "古,罗,马" `
  --output-dir .\runs\manual-order
```

## Python API

```python
from pathlib import Path

from gsxt_solver import ModelPaths, Solver

project_root = Path(r"D:\path\to\Gsxt-Solver")
model_dir = project_root / "models" / "gsxt-models-v0.1.0"

models = ModelPaths.from_bundle(
    model_dir,
    project_root=project_root,
)

solver = Solver.from_project(
    project_root,
    models=models,
    use_gpu=False,
)

result = solver.solve(
    project_root / "tests" / "fixtures" / "test10.png",
    output_dir=project_root / "runs" / "api-test10",
)

print(result["task_spec"])
print(result["merge_settings"]["resolved_target_order"])

for item in result["items"]:
    print(item["center"], item.get("text") or item.get("label"))
```

Important result fields:

| Field | Description |
| --- | --- |
| `task_type` | Resolved modality: `char`, `icon`, or `mixed` |
| `task_spec` | Interpreted action, target source, confidence, and evidence |
| `merge_settings.resolved_target_order` | Final target sequence used by the solver |
| `raw_items` | Candidates before overlap merging |
| `merged_items` | Candidates after overlap and duplicate handling |
| `items` | Final ordered output |

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
