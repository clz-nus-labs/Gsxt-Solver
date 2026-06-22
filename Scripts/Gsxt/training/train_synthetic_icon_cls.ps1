param(
    [switch]$UseGpu,
    [switch]$Resume,
    [int]$EpochNum = 80,
    [int]$BatchSize = 64,
    [int]$Patience = 12,
    [double]$LearningRate = 0.0006,
    [string]$Model = "small_cnn",
    [switch]$Pretrained,
    [string]$DatasetDir = "Scripts\Gsxt\data\datasets\synthetic_icon_cls",
    [string]$SaveDir = ""
)

$ErrorActionPreference = "Stop"

$envName = "paddlex_cv"
$trainPy = "Scripts\Gsxt\training\train_synthetic_icon_cls.py"
$datasetDir = $DatasetDir
$saveDir = if ($SaveDir) {
    $SaveDir
} elseif ($Model -eq "small_cnn") {
    "Scripts\Gsxt\output\training\synthetic_icon_cls"
} else {
    "Scripts\Gsxt\output\training\synthetic_icon_backbone_cls\$Model"
}

if (-not (Test-Path $trainPy)) {
    Write-Host "Missing icon classification train script: $trainPy"
    exit 1
}
if (-not (Test-Path "$datasetDir\train.txt") -or -not (Test-Path "$datasetDir\val.txt")) {
    Write-Host "Missing icon classification dataset. Run synthetic_icon_to_cls.py first."
    exit 1
}
if ($Resume) {
    $missingResumeFiles = @()
    foreach ($suffix in @(".pdparams", ".pdopt", ".json")) {
        $path = Join-Path $saveDir "latest$suffix"
        if (-not (Test-Path $path)) {
            $missingResumeFiles += $path
        }
    }
    if ($missingResumeFiles.Count -gt 0) {
        Write-Host "Missing resume checkpoint files:"
        $missingResumeFiles | ForEach-Object { Write-Host $_ }
        exit 1
    }
}

$device = if ($UseGpu) { "gpu" } else { "cpu" }
$args = @(
    $trainPy,
    "--dataset", $datasetDir,
    "--epochs", "$EpochNum",
    "--batch-size", "$BatchSize",
    "--patience", "$Patience",
    "--lr", "$LearningRate",
    "--device", $device,
    "--model", $Model,
    "--output", $saveDir
)

if ($Pretrained) {
    $args += "--pretrained"
}

if ($Resume) {
    $args += "--resume"
}

if ($env:CONDA_DEFAULT_ENV -eq $envName -and $env:CONDA_PREFIX) {
    $pythonExe = Join-Path $env:CONDA_PREFIX "python.exe"
    & $pythonExe @args
} else {
    conda run -n $envName python @args
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Synthetic icon classification training failed."
    exit 1
}
