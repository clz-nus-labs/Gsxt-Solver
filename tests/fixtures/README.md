# Evaluation fixtures

This directory contains the 50 image fixtures used during development:

- `test1.png` corresponds to the original local `test.png`;
- `test2.png` through `test50.png` retain their original numbering.

They are evaluation inputs, not training data, and currently do not include authoritative
ground-truth labels. Keep the repository private until the provenance and redistribution
terms of every image have been reviewed.

Run all fixtures after installing the package and downloading the model bundle:

```powershell
gsxt-test-suite `
  --project-root . `
  --model-dir .\models\gsxt-models-v0.1.0 `
  --fixtures .\tests\fixtures `
  --output-dir .\runs\test-suite
```
