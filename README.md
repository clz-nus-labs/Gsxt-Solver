# GSXT Solver

**Version 0.3.0**

GSXT Solver is a PaddlePaddle-based structured image-ordering pipeline for
Chinese-character and icon click tasks. Given one captcha-like image, it detects
candidate targets, understands the prompt/header, and returns the three click
points in the required order.

The project can be used in two ways:

1. a standalone Python package/CLI for solving a saved image;
2. a local Chrome/Edge extension bridge that captures the visible challenge,
   calls the Python solver service, and clicks the returned points.

> Authorized-use notice: this project is intended for research, regression
> testing, and automation in environments where you have permission to operate.
> Respect the rules and terms of any website or system you interact with.

## What is new in 0.3.0

- Bundled a newer lightweight header-intent model trained on 200 annotated
  fixtures plus recent real challenge captures.
- Improved arbitration among:
  - `char_given_order`
  - `char_semantic_order`
  - `icon_given_order`
- Fixed semantic ordering for common Chinese phrases such as `台北市`.
- Added a browser-extension bridge under `Scripts/GJQYXYGS`.
- Kept standard API output compact: probabilities and debug internals are hidden
  unless debug mode is requested.

## Supported task scope

GSXT Solver is designed for structured click-order images with:

- a prompt/header area near the top;
- three body targets;
- Chinese-character targets or icon-like targets;
- either a given-order instruction or a semantic-order instruction.

It is not a general OCR, object-detection, or universal captcha framework. It
works best on layouts similar to the bundled fixtures and the supported Geetest
click challenge layout.

Example fixture:

![Example fixture](tests/fixtures/test10.png)

High-level flow:

```text
input image
  -> detect candidate char/icon regions
  -> recognize body characters and icons
  -> parse prompt/header intent
  -> optionally apply header-intent arbitration
  -> order the three targets
  -> return points
```

## Repository layout

```text
src/gsxt_solver/              Python package, CLI, model downloader, public API
Scripts/Gsxt/                 Paddle inference backend, evaluation/training tools
Scripts/GJQYXYGS/             local browser-extension bridge and Flask service
tests/fixtures/               small packaged image fixtures
```

Generated outputs, logs, browser profiles, model weights, Paddle source clones,
and local environments are intentionally not part of the repository.

## Requirements

- Windows
- Python 3.10
- Git
- PaddlePaddle 3.2.x
- Chrome or Edge, only for the extension workflow

GPU inference requires a PaddlePaddle build compatible with your local
CUDA/CUDNN runtime. CPU inference is slower but simpler and is a good first
cross-machine check.

## Installation from GitHub

```powershell
git clone https://github.com/clz-nus-labs/Gsxt-Solver.git
cd Gsxt-Solver
```

Create an environment:

```powershell
conda create -n gsxt_solver python=3.10 -y
conda activate gsxt_solver
```

Install PaddlePaddle. CPU example:

```powershell
python -m pip install paddlepaddle==3.2.0 `
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

Install the package and inference dependencies:

```powershell
python -m pip install -e ".[inference]"
```

Install the required PaddleDetection and PaddleOCR source dependencies:

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

If the `gsxt_solver` environment is not currently activated, omit
`-DirectPython`; the scripts will use `conda run -n gsxt_solver`.

## Download model weights

The repository does not store Paddle model weights. Download the release assets
and assemble the runtime model directory with:

```powershell
gsxt-models `
  --destination .\dist\models\gsxt-models-v0.1.0 `
  --release-base-url https://github.com/clz-nus-labs/Gsxt-Solver/releases/download/models-v0.1.0
```

Expected output directory:

```text
dist/models/gsxt-models-v0.1.0/
  det/
    best_model.pdparams
    dataset/
  rec/
    best_accuracy.pdparams
    config.yml
  icon/
    best_accuracy.pdparams
    label_list.txt
```

For public releases, no token is required. For a private release, set `GH_TOKEN`
or `GITHUB_TOKEN` before running `gsxt-models`.

## Standalone Python API

Use `Solver.from_bundle()` after downloading the model bundle:

```python
from gsxt_solver import Solver

solver = Solver.from_bundle(
    project_root=".",
    model_dir="dist/models/gsxt-models-v0.1.0",
    use_gpu=False,
)

result = solver.solve("tests/fixtures/test10.png", mode="standard")
print(result)
```

Standard mode returns a stable public schema and does not save files by default:

```json
{
  "schema_version": "1.0",
  "success": true,
  "image": "captcha.png",
  "task": {
    "action": "semantic_order",
    "type": "char"
  },
  "result": {
    "count": 3,
    "sequence": ["台", "北", "市"],
    "points": [
      {"x": 306, "y": 334},
      {"x": 315, "y": 154},
      {"x": 131, "y": 322}
    ],
    "items": [
      {
        "index": 1,
        "type": "char",
        "value": "台",
        "center": {"x": 306, "y": 334},
        "bbox": {"left": 264, "top": 288, "right": 348, "bottom": 380}
      }
    ]
  }
}
```

Save standard-mode files only when needed:

```python
result = solver.solve(
    "tests/fixtures/test10.png",
    mode="standard",
    output_dir="runs/example",
    save_result=True,
    save_visual=True,
)
```

Debug mode returns the full backend payload and saves `result.json` plus a
visualization by default:

```python
debug_result = solver.solve(
    "tests/fixtures/test10.png",
    mode="debug",
    output_dir="runs/example-debug",
)
```

The bundled header-intent model is enabled by default. Disable it only for
rule-only comparisons:

```python
solver = Solver.from_bundle(
    project_root=".",
    model_dir="dist/models/gsxt-models-v0.1.0",
    use_gpu=False,
    use_header_intent=False,
)
```

## Command-line usage

Standard mode:

```powershell
gsxt-solve .\tests\fixtures\test10.png `
  --project-root . `
  --model-dir .\dist\models\gsxt-models-v0.1.0 `
  --mode standard `
  --cpu
```

Debug mode:

```powershell
gsxt-solve .\tests\fixtures\test10.png `
  --project-root . `
  --model-dir .\dist\models\gsxt-models-v0.1.0 `
  --mode debug `
  --output-dir .\runs\test10-debug `
  --cpu
```

Useful options:

```text
--cpu / no --cpu       CPU or GPU inference
--mode standard        compact public output
--mode debug           full diagnostic output
--target-order TEXT    manually provide a target order
--no-header-intent     disable header-intent arbitration
--save-result          save result.json in standard mode
--save-visual          save annotated image in standard mode
```

## Browser extension workflow

The extension is a local companion for Chrome/Edge. It captures the currently
visible click challenge, sends it to the local Python service, receives ordered
points, clicks the points in the page, and then clicks confirm.

```text
Chrome/Edge page
  -> unpacked extension captures challenge image
  -> POST http://127.0.0.1:7755/solve
  -> local gsxt_solver service
  -> ordered points
  -> extension clicks points and confirm
```

### 1. Start the local service

From the repository root:

```powershell
python .\Scripts\GJQYXYGS\server.py
```

Expected output:

```text
GSXT Solver local service started
Listening: http://127.0.0.1:7755
Endpoint: POST /solve with image_base64
```

### 2. Load the extension

Open:

```text
chrome://extensions/
```

or:

```text
edge://extensions/
```

Then:

1. enable Developer mode;
2. click **Load unpacked**;
3. select `Scripts/GJQYXYGS/edge_extension`;
4. reload the target page.

### 3. Use it

1. Keep `server.py` running.
2. Open the target page in Chrome or Edge.
3. Use the **GSXT assistant** panel inserted by the extension.
4. When a click challenge appears, the extension captures the challenge-only
   image and calls the local service.
5. If the service returns exactly three valid points, the extension clicks them
   and then clicks confirm.
6. If recognition fails, finish the challenge manually; the workflow is designed
   to continue afterwards.

See the extension-specific guide:

[Scripts/GJQYXYGS/README.md](Scripts/GJQYXYGS/README.md)

## Testing on another machine

After cloning, installing dependencies, and downloading the model bundle, run:

```powershell
python -m pip install -e ".[inference]"

gsxt-solve .\tests\fixtures\test10.png `
  --project-root . `
  --model-dir .\dist\models\gsxt-models-v0.1.0 `
  --mode standard `
  --cpu
```

Then verify the browser service starts:

```powershell
python .\Scripts\GJQYXYGS\server.py
```

If both commands work, the Python API and extension service are wired correctly.

## Evaluation fixtures

Run the packaged fixture suite:

```powershell
gsxt-test-suite `
  --project-root . `
  --model-dir .\dist\models\gsxt-models-v0.1.0 `
  --fixtures .\tests\fixtures `
  --output-dir .\runs\test-suite `
  --cpu
```

The fixture images are regression inputs, not a broad benchmark.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `ModuleNotFoundError: paddle` | Install PaddlePaddle in the active environment. |
| `PaddleDetection` or `PaddleOCR` missing | Run the two setup scripts under `Scripts/Gsxt/training`. |
| model file not found | Run `gsxt-models` and check `dist/models/gsxt-models-v0.1.0`. |
| extension cannot reach service | Confirm `python .\Scripts\GJQYXYGS\server.py` is running. |
| extension behavior did not change after update | Reload the unpacked extension and restart `server.py`. |
| GPU/CUDNN warning | Use CPU first or install a Paddle build compatible with your CUDA/CUDNN runtime. |

## License

This project is licensed under the terms in [LICENSE](LICENSE).
