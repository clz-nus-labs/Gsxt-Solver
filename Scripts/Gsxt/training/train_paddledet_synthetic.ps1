param(
  [switch]$UseGpu,
  [switch]$DirectPython,
  [switch]$Resume,
  [int]$EpochNum = 36,
  [int]$TrainBatchSize = 8,
  [int]$EvalBatchSize = 1,
  [double]$LearningRate = 0.003,
  [string]$DatasetDir = "Scripts\Gsxt\data\datasets\synthetic_mixed_paddledet",
  [string]$ImageDir = "Scripts\Gsxt\synthetic\output_v4\images",
  [string]$SaveDir = "Scripts\Gsxt\output\training\paddledet_synthetic"
)

$ErrorActionPreference = "Stop"

$envName = "paddlex_cv"
$repoDir = "Scripts\Gsxt\third_party\PaddleDetection"
$datasetDir = $DatasetDir
$imageDir = $ImageDir
$saveDir = $SaveDir

$configCandidates = @(
    "configs\picodet\picodet_s_320_coco_lcnet.yml",
    "configs\picodet\picodet_s_416_coco_lcnet.yml",
    "configs\ppyoloe\ppyoloe_crn_s_300e_coco.yml"
)

if (-not (Test-Path "$repoDir\tools\train.py")) {
    Write-Host "Missing PaddleDetection train.py. Run setup_paddledetection_repo.ps1 first."
    exit 1
}
if (-not (Test-Path "$datasetDir\train.json") -or -not (Test-Path "$datasetDir\val.json")) {
    Write-Host "Missing PaddleDetection dataset. Run synthetic_to_paddledet.py first."
    exit 1
}

$config = $null
foreach ($candidate in $configCandidates) {
    $path = Join-Path $repoDir $candidate
    if (Test-Path $path) {
        $config = $path
        break
    }
}
if ($null -eq $config) {
    Write-Host "No supported PaddleDetection config found."
    $configCandidates | ForEach-Object { Write-Host $_ }
    exit 1
}

New-Item -ItemType Directory -Force -Path $saveDir | Out-Null

$repoAbs = (Resolve-Path $repoDir).Path
$trainPy = Join-Path $repoAbs "tools\train.py"
$config = (Resolve-Path $config).Path
$datasetDir = (Resolve-Path $datasetDir).Path
$imageDir = (Resolve-Path $imageDir).Path
$saveDir = (Resolve-Path $saveDir).Path
$useGpuValue = if ($UseGpu) { "true" } else { "false" }

function Get-LatestCheckpointPrefix {
    param([string]$Dir)

    $bestModel = Join-Path $Dir "best_model"
    if (Test-Path "$bestModel.pdparams") {
        return $bestModel
    }

    $modelFinal = Join-Path $Dir "model_final"
    if (Test-Path "$modelFinal.pdparams") {
        return $modelFinal
    }

    $latestEpoch = Get-ChildItem -Path $Dir -Filter "*.pdparams" -File |
        Where-Object { $_.BaseName -match "^\d+$" } |
        Sort-Object { [int]$_.BaseName } -Descending |
        Select-Object -First 1
    if ($latestEpoch) {
        return (Join-Path $Dir $latestEpoch.BaseName)
    }

    return $null
}

$resumeCheckpoint = $null
if ($Resume) {
    $resumeCheckpoint = Get-LatestCheckpointPrefix -Dir $saveDir
    if (-not $resumeCheckpoint) {
        Write-Host "No checkpoint found under: $saveDir"
        exit 1
    }
}

Write-Host "Using PaddleDetection config: $config"
Write-Host "Training output: $saveDir"
Write-Host "Use GPU: $useGpuValue"
if ($Resume) {
    Write-Host "Resume checkpoint: $resumeCheckpoint"
}

$oldPythonPath = $env:PYTHONPATH
if ($oldPythonPath) {
    $env:PYTHONPATH = "$repoAbs;$repoAbs\tools;$oldPythonPath"
} else {
    $env:PYTHONPATH = "$repoAbs;$repoAbs\tools"
}

$overrideArgs = @(
    "use_gpu=$useGpuValue",
    "epoch=$EpochNum",
    "save_dir=$saveDir",
    "snapshot_epoch=1",
    "num_classes=2",
    "TrainDataset.dataset_dir=$datasetDir",
    "TrainDataset.image_dir=$imageDir",
    "TrainDataset.anno_path=train.json",
    "EvalDataset.dataset_dir=$datasetDir",
    "EvalDataset.image_dir=$imageDir",
    "EvalDataset.anno_path=val.json",
  "TrainReader.batch_size=$TrainBatchSize",
  "EvalReader.batch_size=$EvalBatchSize",
  "LearningRate.base_lr=$LearningRate"
)

if (-not $Resume) {
    $overrideArgs += "pretrain_weights="
}

$trainArgs = @($trainPy, "-c", $config, "--eval")
if ($Resume) {
    $trainArgs += @("-r", $resumeCheckpoint)
}
$trainArgs += "-o"
$trainArgs += $overrideArgs

if ($DirectPython) {
    if ($env:CONDA_DEFAULT_ENV -ne $envName -or -not $env:CONDA_PREFIX) {
        Write-Host "DirectPython requires activated conda env: $envName"
        Write-Host "Current CONDA_DEFAULT_ENV: $env:CONDA_DEFAULT_ENV"
        exit 1
    }
    $pythonExe = Join-Path $env:CONDA_PREFIX "python.exe"
    Push-Location $repoAbs
    try {
        & $pythonExe @trainArgs
    } finally {
        Pop-Location
    }
} else {
    Push-Location $repoAbs
    try {
        $condaArgs = @("run", "-n", $envName, "python") + $trainArgs
        & conda @condaArgs
    } finally {
        Pop-Location
    }
}

if ($oldPythonPath) {
    $env:PYTHONPATH = $oldPythonPath
} else {
    Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "PaddleDetection synthetic training failed."
    exit 1
}
