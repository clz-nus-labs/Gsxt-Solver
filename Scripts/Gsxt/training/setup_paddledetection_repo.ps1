param(
  [string]$EnvName = "paddlex_cv",
  [switch]$DirectPython
)

$ErrorActionPreference = "Stop"
$repoDir = "Scripts\Gsxt\third_party\PaddleDetection"
$repoUrl = "https://github.com/PaddlePaddle/PaddleDetection.git"
$repoCommit = "b25522a0f4bde8c80603f3ba5e3472059972e3b5"
$constraints = "Scripts\Gsxt\runtime-constraints.txt"

if (
  $DirectPython -and (
    -not $env:CONDA_PREFIX -or
    $env:CONDA_DEFAULT_ENV -ne $EnvName
  )
) {
  throw "Direct Python requires the activated conda env: $EnvName"
}

if (-not (Test-Path $repoDir)) {
  Write-Host "Cloning PaddleDetection into $repoDir ..."
  git clone --no-checkout $repoUrl $repoDir
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $repoDir)) {
    throw "PaddleDetection clone failed. Please download the repo manually into $repoDir, then rerun this script."
  }
}
else {
  Write-Host "PaddleDetection repo already exists: $repoDir"
}

git -C $repoDir checkout $repoCommit
if ($LASTEXITCODE -ne 0) {
  throw "Failed to check out PaddleDetection commit: $repoCommit"
}

$requirements = Join-Path $repoDir "requirements.txt"
if (-not (Test-Path $requirements)) {
  throw "PaddleDetection requirements.txt not found: $requirements"
}
if (-not (Test-Path $constraints)) {
  throw "Runtime constraints not found: $constraints"
}

if ($DirectPython -or $env:CONDA_DEFAULT_ENV -eq $EnvName) {
  $python = Join-Path $env:CONDA_PREFIX "python.exe"
  if (-not (Test-Path $python)) {
    throw "Python not found under CONDA_PREFIX: $python"
  }

  Write-Host "Using activated env python: $python"
  & $python -m pip install -U pip "setuptools<81" wheel
  if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip/setuptools/wheel." }

  & $python -m pip install --no-build-isolation -c $constraints -r $requirements
  if ($LASTEXITCODE -ne 0) { throw "Failed to install PaddleDetection requirements." }

  & $python -m pip install -c $constraints imgaug
  if ($LASTEXITCODE -ne 0) { throw "Failed to install imgaug." }
}
else {
  Write-Host "Using conda run env: $EnvName"
  conda run -n $EnvName python -m pip install -U pip "setuptools<81" wheel
  if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip/setuptools/wheel." }

  conda run -n $EnvName python -m pip install --no-build-isolation -c $constraints -r $requirements
  if ($LASTEXITCODE -ne 0) { throw "Failed to install PaddleDetection requirements." }

  conda run -n $EnvName python -m pip install -c $constraints imgaug
  if ($LASTEXITCODE -ne 0) { throw "Failed to install imgaug." }
}

Write-Host "Done."
