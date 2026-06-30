# GitHub publication guide

Current package release: `v0.3.0`

Model asset release: `models-v0.1.0`

## 1. What belongs in Git

Commit:

- `src/gsxt_solver`
- the current `Scripts/Gsxt/demos/dynamic_mixed_infer.py` backend
- the current detector, recognizer and icon-classifier training entry points
- dataset conversion, synthetic generation and semantic lexicon utilities
- semantic lexicons that permit redistribution
- package metadata, documentation, and tests
- the local browser-extension bridge under `Scripts/GJQYXYGS`

Do not commit:

- trained weights and optimizer state
- generated outputs
- large training datasets
- local virtual environments
- cloned Paddle repositories as ordinary copied directories
- browser profiles, logs, downloaded reports, and generated debug captures

## 2. Third-party repositories

The current inference backend imports PaddleDetection and PaddleOCR from source. They stay
out of Git and are installed at pinned commits by the setup scripts under
`Scripts/Gsxt/training`.

## 3. Create the repository

```powershell
git init
git branch -M main
git add .
git commit -m "Package GSXT inference pipeline"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Review `git status` and the staged file list before committing. In particular, make sure no
training images, private data, or model checkpoints are staged.

## 4. Publish model assets

Build the local bundle:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_model_bundle.ps1 -Version 0.1.0
```

Create a release named `models-v0.1.0`, then upload these files from the bundle:

- `gsxt-models-v0.1.0.zip`
- `det-best_model.pdparams`
- `rec-best_accuracy.pdparams`
- `rec-config.yml`
- `icon-best_accuracy.pdparams`
- `icon-label_list.txt`

The names must match `src/gsxt_solver/assets/models.json`.

Users download them with:

```powershell
gsxt-models `
  --destination .\dist\models\gsxt-models-v0.1.0 `
  --release-base-url https://github.com/OWNER/REPOSITORY/releases/download/models-v0.1.0
```

## 5. Package release

Create or update the code release tag:

```powershell
git tag v0.3.0
git push origin v0.3.0
```

Then create a GitHub Release named `GSXT Solver v0.3.0`. The code release does
not need to include model weights; users download weights from `models-v0.1.0`
with `gsxt-models`.

```powershell
python -m pip install build
python -m build
```

The wheel and source archive are written under `dist/`. Test the wheel in a fresh environment
before publishing it to PyPI.

## 6. Before making the repository public

- Verify the license of every dataset, icon source, font, and lexicon.
- Remove absolute local paths from distributable configuration files.
- Document the exact PaddlePaddle/CUDA/CUDNN compatibility matrix.
- Keep the authorized-use notice in the README.
