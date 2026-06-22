$ErrorActionPreference = "Stop"

$repoDir = "Scripts\Gsxt\third_party\PaddleOCR"
$pretrainDir = Join-Path $repoDir "pretrain_models"
$archivePath = Join-Path $pretrainDir "ch_PP-OCRv4_rec_train.tar"
$targetDir = Join-Path $pretrainDir "ch_PP-OCRv4_rec_train"
$modelPath = Join-Path $targetDir "student.pdparams"
$url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar"

if (-not (Test-Path $repoDir)) {
    Write-Host "Missing PaddleOCR repo: $repoDir"
    Write-Host "Run setup_paddleocr_repo.ps1 first."
    exit 1
}

New-Item -ItemType Directory -Force -Path $pretrainDir | Out-Null

if (-not (Test-Path $modelPath)) {
    if (-not (Test-Path $archivePath)) {
        Write-Host "Downloading PP-OCRv4 recognition pretrained model..."
        Write-Host $url
        Invoke-WebRequest -Uri $url -OutFile $archivePath
    } else {
        Write-Host "Archive already exists: $archivePath"
    }

    Write-Host "Extracting: $archivePath"
    tar -xf $archivePath -C $pretrainDir
}

if (-not (Test-Path $modelPath)) {
    Write-Host "Pretrained model was not found after extraction:"
    Write-Host $modelPath
    Write-Host "Please check archive contents."
    exit 1
}

Write-Host "Recognition pretrained model is ready:"
Write-Host (Join-Path $targetDir "student")
