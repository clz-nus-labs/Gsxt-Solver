# GSXT Solver

`gsxt-solver` packages the latest trained character/icon ordering pipeline behind a stable Python API and CLI.

> Use only on images and systems you own or are authorized to test.

The repository intentionally contains only the current inference implementation and the
training utilities needed to reproduce it. Model weights, datasets, browser profiles,
generated images, historical experiments, and the unrelated `GJQYXYGS` automation project
are not stored in Git.

## Install from a checkout

```powershell
python -m pip install -e .
```

Install PaddlePaddle separately for the target CPU/GPU environment, then install the optional runtime dependencies:

```powershell
python -m pip install -e ".[inference]"
```

The current backend requires pinned PaddleDetection and PaddleOCR source trees:

```powershell
powershell -ExecutionPolicy Bypass -File .\Scripts\Gsxt\training\setup_paddledetection_repo.ps1
powershell -ExecutionPolicy Bypass -File .\Scripts\Gsxt\training\setup_paddleocr_repo.ps1
```

PaddlePaddle must be installed separately for the target CPU/GPU and CUDA environment.

## Python API

```python
from gsxt_solver import Solver

solver = Solver.from_project(
    project_root=r"D:\path\to\repository",
    python_executable=r"C:\path\to\paddlex_cv\python.exe",
)

result = solver.solve("example.png")
print(result["task_spec"])
print(result["items"])
```

You can point to a downloaded model bundle:

```python
from gsxt_solver import ModelPaths, Solver

solver = Solver.from_project(
    project_root=r"D:\path\to\repository",
    models=ModelPaths.from_bundle(
        r"D:\models\gsxt-models-v0.1.0",
        project_root=r"D:\path\to\repository",
    ),
)
```

## CLI

```powershell
gsxt-solve .\example.png --project-root . --output-dir .\runs\example
```

With a downloaded model bundle:

```powershell
gsxt-solve .\example.png `
  --project-root D:\path\to\repository `
  --model-dir D:\models\gsxt-models-v0.1.0
```

## Model distribution

Model weights are intentionally excluded from Git. Build the release bundle:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_model_bundle.ps1
```

Upload the generated archive under `dist/models/` to a GitHub Release. The bundle includes SHA-256 metadata.

For this private repository, an authenticated user can download a release with:

```powershell
gh release download models-v0.1.0 `
  --repo clz-nus/Gsxt-Solver `
  --pattern "gsxt-models-v0.1.0.zip" `
  --dir .\models
```

Extract the archive before passing its directory to `ModelPaths.from_bundle`.

The component downloader also supports private releases when `GH_TOKEN` or
`GITHUB_TOKEN` is available:

```powershell
$env:GH_TOKEN = gh auth token
gsxt-models `
  --destination .\models\gsxt-models-v0.1.0 `
  --release-base-url https://github.com/clz-nus/Gsxt-Solver/releases/download/models-v0.1.0
```

## Scope

- `src/gsxt_solver`: importable API, CLI, model manifest and downloader
- `Scripts/Gsxt/demos/dynamic_mixed_infer.py`: current inference backend
- `Scripts/Gsxt/training`: current detector, recognizer and icon-classifier training entry points
- `Scripts/Gsxt/tools`: dataset conversion and semantic lexicon tools used by the current workflow
- `Scripts/Gsxt/synthetic/generate_mixed_scene.py`: synthetic training data generator

The model weights are distributed separately because their training-data provenance and
usage terms must be reviewed independently from the Apache-2.0 source-code license.

## Repository publishing

See [GITHUB_RELEASE.md](GITHUB_RELEASE.md) for the recommended GitHub and release workflow.
