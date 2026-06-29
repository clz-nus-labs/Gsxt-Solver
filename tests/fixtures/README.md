# Evaluation fixtures

This directory contains the 50 image fixtures used during development:

They are evaluation inputs, not training data, and currently do not include authoritative
ground-truth labels.

Run all fixtures after installing the package and downloading the model bundle:

```powershell
gsxt-test-suite `
  --project-root . `
  --model-dir .\dist\models\gsxt-models-v0.1.0 `
  --fixtures .\tests\fixtures `
  --output-dir .\runs\test-suite
```
