# GSXT Browser Extension Bridge

This directory contains the local Chrome/Edge extension bridge for GSXT Solver.

The extension itself does not contain Paddle models. It only captures the
visible challenge area in the browser and calls the local Python service.

```text
Chrome/Edge page
  -> unpacked extension captures the click challenge
  -> local Flask service receives image_base64
  -> gsxt_solver runs image inference
  -> service returns ordered points
  -> extension clicks the points and confirms
```

## Directory layout

```text
Scripts/GJQYXYGS/
  server.py                    Local Flask service
  gsxt_solver_bridge.py        Base64 image bridge to gsxt_solver
  config.example.json          Example task configuration
  requirements.txt             Small service dependencies
  edge_extension/              Chrome/Edge unpacked extension
    manifest.json
    background.js
    content.js
    panel.css
```

Runtime logs are written under `Scripts/GJQYXYGS/logs/`; this directory is local
output and is not meant to be committed.

## Prerequisites

From the repository root:

```powershell
python -m pip install -e ".[inference]"
python -m pip install -r .\Scripts\GJQYXYGS\requirements.txt
```

Download the model bundle:

```powershell
gsxt-models `
  --destination .\dist\models\gsxt-models-v0.1.0 `
  --release-base-url https://github.com/clz-nus-labs/Gsxt-Solver/releases/download/models-v0.1.0
```

Expected model directory:

```text
dist/models/gsxt-models-v0.1.0
```

## Start the local service

From the repository root:

```powershell
python .\Scripts\GJQYXYGS\server.py
```

Expected console output:

```text
GSXT Solver local service started
Listening: http://127.0.0.1:7755
Endpoint: POST /solve with image_base64
```

The service defaults to CPU mode for browser integration. If you have a stable
Paddle/CUDA/CUDNN GPU environment, you can change `USE_GPU` in `server.py`.

## Load the extension

Chrome:

```text
chrome://extensions/
```

Edge:

```text
edge://extensions/
```

Then:

1. enable Developer mode;
2. click **Load unpacked**;
3. select `Scripts/GJQYXYGS/edge_extension`;
4. reload the target page.

Reload the extension after editing any file under `edge_extension/`.

## Use the extension

1. Start `server.py`.
2. Open the target page normally in Chrome or Edge.
3. Use the **GSXT assistant** panel added by the extension.
4. Trigger the current page action from the panel.
5. When a click challenge appears, the extension captures the challenge-only
   image and sends it to the local service.
6. If the service returns exactly three valid points, the extension clicks them
   in order and then clicks confirm.
7. If recognition fails, manually complete the challenge; the workflow can
   continue afterwards.

## Local service API

Endpoint:

```text
POST http://127.0.0.1:7755/solve
```

Request shape:

```json
{
  "image_base64": "data:image/png;base64,...",
  "crop_info": {
    "solverCropMode": "challenge_only"
  }
}
```

Example successful response:

```json
{
  "success": true,
  "task": {
    "action": "semantic_order",
    "type": "char"
  },
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
      "center": {"x": 306, "y": 334}
    }
  ]
}
```

For icon-like tasks, labels are primarily diagnostic. The browser workflow
should trust the returned point order more than the textual label names.

## Manual API test without the extension

Use a saved PNG:

```powershell
gsxt-solve .\tests\fixtures\test10.png `
  --project-root . `
  --model-dir .\dist\models\gsxt-models-v0.1.0 `
  --mode standard `
  --cpu
```

Or call Python directly:

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

## Debug captures

The service saves debug captures and JSON responses under:

```text
Scripts/GJQYXYGS/logs/captcha_debug
```

Typical files:

```text
captcha_YYYYMMDDHHMMSS_xxxxx.png
captcha_YYYYMMDDHHMMSS_xxxxx.json
captcha_YYYYMMDDHHMMSS_xxxxx.capture.json
```

Use them to locate failures:

| Symptom | Inspect | Likely cause |
| --- | --- | --- |
| PNG is empty or shifted | `.png` and `.capture.json` | browser capture/cropping issue |
| PNG is correct but task type is wrong | `.json` task fields | prompt/header intent issue |
| task type is correct but points are wrong | `.json` items and points | detection/OCR/icon recognition issue |
| fewer or more than 3 points | `.json` postprocess field | non-target controls or crop filtering |

The service terminal also prints concise logs for the task type, sequence,
points, crop metadata, and saved debug image path.

## Common issues

### Extension says the local service is unavailable

Start the service:

```powershell
python .\Scripts\GJQYXYGS\server.py
```

Then reload the page and try again.

### Extension was updated but behavior did not change

Reload the unpacked extension from `chrome://extensions/` or
`edge://extensions/`, then refresh the target page. Restart `server.py` if the
Python code changed.

### Captured image includes the confirm button or footer controls

Version 0.3.0 sends a challenge-only crop to the solver and keeps confirm-click
logic in the extension. If old behavior persists, reload the extension and make
sure the newest `content.js` is loaded.

### Solver returns fewer or more than three points

The service marks the response as failed and leaves the page for manual
completion. Check the saved debug PNG and JSON files to decide whether the
problem is cropping, task intent, detection, or recognition.

### GPU runtime warnings

Use CPU mode first. Only enable GPU in `server.py` after confirming that your
PaddlePaddle, CUDA, and CUDNN versions are compatible.
