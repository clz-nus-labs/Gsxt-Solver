param(
    [string]$EnvName = "paddlex_cv",
    [switch]$DirectPython
)

$ErrorActionPreference = "Stop"

$repoDir = "Scripts\Gsxt\third_party\PaddleOCR"
$repoUrl = "https://github.com/PaddlePaddle/PaddleOCR.git"
$repoCommit = "9f704bc6abf7a09f22593d597f633d62668b2984"
$requirementsPath = Join-Path $repoDir "requirements.txt"

if (-not (Test-Path "Scripts\Gsxt\third_party")) {
    New-Item -ItemType Directory -Path "Scripts\Gsxt\third_party" | Out-Null
}

if (-not (Test-Path $repoDir)) {
    Write-Host "Cloning PaddleOCR into $repoDir ..."
    git clone --no-checkout $repoUrl $repoDir

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $repoDir)) {
        Write-Host ""
        Write-Host "PaddleOCR clone failed. Dependency installation stopped."
        Write-Host "Please check GitHub access, or manually download PaddleOCR to:"
        Write-Host $repoDir
        Write-Host "Repository: $repoUrl"
        exit 1
    }
} else {
    Write-Host "PaddleOCR repo already exists: $repoDir"
}

git -C $repoDir checkout $repoCommit
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to check out PaddleOCR commit: $repoCommit"
    exit 1
}

if (-not (Test-Path $requirementsPath)) {
    Write-Host ""
    Write-Host "Missing requirements file:"
    Write-Host $requirementsPath
    Write-Host "Please make sure PaddleOCR is fully downloaded into:"
    Write-Host $repoDir
    exit 1
}

if ($DirectPython -or $env:CONDA_DEFAULT_ENV -eq $EnvName) {
    if (-not $env:CONDA_PREFIX) {
        throw "CONDA_PREFIX is empty. Please run: conda activate $EnvName"
    }
    $python = Join-Path $env:CONDA_PREFIX "python.exe"
    if (-not (Test-Path $python)) {
        throw "Python not found under CONDA_PREFIX: $python"
    }
    Write-Host "Installing PaddleOCR training requirements with: $python"
    & $python -m pip install -r $requirementsPath
}
else {
    Write-Host "Installing PaddleOCR training requirements into $EnvName ..."
    conda run -n $EnvName python -m pip install -r $requirementsPath
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Dependency installation failed. Please check pip network access or conda env:"
    Write-Host $EnvName
    exit 1
}

Write-Host "Done."
