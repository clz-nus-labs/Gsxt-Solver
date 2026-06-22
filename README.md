# GSXT Solver

`gsxt-solver` packages the latest trained character/icon ordering pipeline behind a stable Python API and CLI.



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

## Install on another Windows host

The repository can remain private. The GitHub account used on the other host must be an
owner, organization member or collaborator with access to `clz-nus/Gsxt-Solver`.

```powershell
gh auth login
gh auth setup-git
gh repo clone clz-nus/Gsxt-Solver
cd Gsxt-Solver

conda create -n gsxt_solver python=3.10 -y
conda activate gsxt_solver

# CPU example. For GPU, install the PaddlePaddle build matching CUDA/CUDNN instead.
python -m pip install paddlepaddle==3.2.0 `
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -e ".[inference]"

powershell -ExecutionPolicy Bypass `
  -File .\Scripts\Gsxt\training\setup_paddledetection_repo.ps1 `
  -EnvName gsxt_solver -DirectPython
powershell -ExecutionPolicy Bypass `
  -File .\Scripts\Gsxt\training\setup_paddleocr_repo.ps1 `
  -EnvName gsxt_solver `
  -DirectPython
```

The existing development machine reported a CUDNN mismatch: Paddle was compiled with
CUDNN 9.9 while the machine provided CUDNN 9.5. A new GPU host should install a compatible
PaddlePaddle/CUDA/CUDNN combination. CPU installation avoids that GPU compatibility issue.

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

This command downloads the individual detector, recognizer and icon-classifier assets,
verifies every SHA-256 hash, and assembles the directory expected by `ModelPaths.from_bundle`.
No manual weight concatenation is required.

Run one bundled fixture:

```powershell
gsxt-solve .\tests\fixtures\test10.png `
  --project-root . `
  --model-dir .\models\gsxt-models-v0.1.0 `
  --output-dir .\runs\test10 `
  --cpu
```

Run all 50 fixtures:

```powershell
gsxt-test-suite `
  --project-root . `
  --model-dir .\models\gsxt-models-v0.1.0 `
  --fixtures .\tests\fixtures `
  --output-dir .\runs\test-suite `
  --cpu
```

The combined report is written to `runs/test-suite/summary.json`.

## Scope

- `src/gsxt_solver`: importable API, CLI, model manifest and downloader
- `Scripts/Gsxt/demos/dynamic_mixed_infer.py`: current inference backend
- `Scripts/Gsxt/training`: current detector, recognizer and icon-classifier training entry points
- `Scripts/Gsxt/tools`: dataset conversion and semantic lexicon tools used by the current workflow
- `Scripts/Gsxt/synthetic/generate_mixed_scene.py`: synthetic training data generator
- `tests/fixtures`: the 50 development evaluation images

The model weights are distributed separately because their training-data provenance and
usage terms must be reviewed independently from the Apache-2.0 source-code license.

## Repository publishing

See [GITHUB_RELEASE.md](GITHUB_RELEASE.md) for the recommended GitHub and release workflow.
