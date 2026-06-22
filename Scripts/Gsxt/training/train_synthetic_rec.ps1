param(
    [switch]$UseGpu,
    [switch]$DirectPython,
    [switch]$Resume,
    [int]$EpochNum = 30,
    [int]$EvalStep = 2000,
    [int]$BatchSize = 32,
    [double]$LearningRate = 0.00008,
    [string]$TrainLabel = "Scripts\Gsxt\data\datasets\synthetic_mixed_paddleocr\rec\train_rec.txt",
    [string]$ValLabel = "Scripts\Gsxt\data\datasets\synthetic_mixed_paddleocr\rec\val_rec.txt",
    [string]$SaveDir = "Scripts\Gsxt\output\training\synthetic_rec",
    [string]$CharacterDictPath = "",
    [int]$Patience = 0,
    [ValidateSet("ppocrv4", "ppocrv4_ctc", "lite_ctc")]
    [string]$ConfigMode = "ppocrv4",
    [string]$ConfigPath = "",
    [switch]$NoPretrained,
    [int]$NumWorkers = 2
)

$ErrorActionPreference = "Stop"

$envName = "paddlex_cv"
$repoDir = "Scripts\Gsxt\third_party\PaddleOCR"
$earlyStopRunner = "Scripts\Gsxt\tools\early_stop_runner.py"
$trainLabel = $TrainLabel
$valLabel = $ValLabel
$saveDir = $SaveDir
$pretrainedModel = "Scripts\Gsxt\third_party\PaddleOCR\pretrain_models\ch_PP-OCRv4_rec_train\student"
$resumeCheckpoint = Join-Path $saveDir "latest"

$configCandidates = if ($ConfigPath) {
    @($ConfigPath)
} elseif ($ConfigMode -eq "lite_ctc") {
    @(
        "configs\rec\ch_ppocr_v2.0\rec_chinese_lite_train_v2.0.yml",
        "configs\rec\ch_PP-OCRv2\ch_PP-OCRv2_rec.yml"
    )
} elseif ($ConfigMode -eq "ppocrv4_ctc") {
    @(
        "Scripts\Gsxt\training\configs\ppocrv4_mobile_rec_ctc_single_char.yml"
    )
} else {
    @(
        "configs\rec\PP-OCRv4\PP-OCRv4_mobile_rec.yml",
        "configs\rec\PP-OCRv3\PP-OCRv3_mobile_rec.yml",
        "configs\rec\ch_PP-OCRv2\ch_PP-OCRv2_rec.yml"
    )
}

$config = $null
foreach ($candidate in $configCandidates) {
    if ([System.IO.Path]::IsPathRooted($candidate)) {
        $path = $candidate
    } elseif (Test-Path $candidate) {
        $path = $candidate
    } else {
        $path = Join-Path $repoDir $candidate
    }
    if (Test-Path $path) {
        $config = $path
        break
    }
}

if ($null -eq $config) {
    Write-Host "No supported recognition config found in PaddleOCR repo."
    exit 1
}
if (-not (Test-Path "$repoDir\tools\train.py")) {
    Write-Host "Missing PaddleOCR train.py. Please run setup_paddleocr_repo.ps1 first."
    exit 1
}
if ($Patience -gt 0 -and -not (Test-Path $earlyStopRunner)) {
    Write-Host "Missing early-stop runner: $earlyStopRunner"
    exit 1
}
if (-not (Test-Path $trainLabel) -or -not (Test-Path $valLabel)) {
    Write-Host "Missing recognition labels."
    exit 1
}
if ($CharacterDictPath -and -not (Test-Path $CharacterDictPath)) {
    Write-Host "Missing character dict: $CharacterDictPath"
    exit 1
}
if ((-not $Resume) -and (-not $NoPretrained) -and (-not (Test-Path "$pretrainedModel.pdparams"))) {
    Write-Host "Missing recognition pretrained model. Run setup_rec_pretrained.ps1 first."
    exit 1
}
if ($Resume) {
    $missingResumeFiles = @()
    foreach ($suffix in @(".pdparams", ".pdopt", ".states")) {
        if (-not (Test-Path "$resumeCheckpoint$suffix")) {
            $missingResumeFiles += "$resumeCheckpoint$suffix"
        }
    }
    if ($missingResumeFiles.Count -gt 0) {
        Write-Host "Missing resume checkpoint files:"
        $missingResumeFiles | ForEach-Object { Write-Host $_ }
        exit 1
    }
}

New-Item -ItemType Directory -Force -Path $saveDir | Out-Null

$repoAbs = (Resolve-Path $repoDir).Path
$trainPy = Join-Path $repoAbs "tools\train.py"
$earlyStopRunner = (Resolve-Path $earlyStopRunner).Path
$config = (Resolve-Path $config).Path
$trainLabel = (Resolve-Path $trainLabel).Path
$valLabel = (Resolve-Path $valLabel).Path
$saveDir = (Resolve-Path $saveDir).Path
$characterDictPathResolved = ""
if ($CharacterDictPath) {
    $characterDictPathResolved = (Resolve-Path $CharacterDictPath).Path
}
$resumeCheckpoint = Join-Path $saveDir "latest"
if ((-not $Resume) -and (-not $NoPretrained)) {
    $pretrainedModel = (Resolve-Path "$pretrainedModel.pdparams").Path
    $pretrainedModel = $pretrainedModel.Substring(0, $pretrainedModel.Length - ".pdparams".Length)
}

$useGpuValue = if ($UseGpu) { "True" } else { "False" }
$learningRateText = $LearningRate.ToString("0.################", [System.Globalization.CultureInfo]::InvariantCulture)
Write-Host "Using synthetic recognition config: $config"
Write-Host "Training output: $saveDir"
Write-Host "Use GPU: $useGpuValue"
if ($Resume) {
    Write-Host "Resume checkpoint: $resumeCheckpoint"
} elseif ($NoPretrained) {
    Write-Host "Pretrained model: disabled"
} else {
    Write-Host "Pretrained model: $pretrainedModel"
}

$overrideArgs = @(
    "Global.use_gpu=$useGpuValue",
    "Global.save_model_dir=$saveDir",
    "Global.epoch_num=$EpochNum",
    "Global.eval_batch_step=[0,$EvalStep]",
    "Optimizer.lr.learning_rate=$learningRateText",
    "Optimizer.lr.warmup_epoch=1",
    "Train.loader.batch_size_per_card=$BatchSize",
    "Train.loader.num_workers=$NumWorkers",
    "Eval.loader.batch_size_per_card=$BatchSize",
    "Eval.loader.num_workers=$NumWorkers",
    "Train.dataset.data_dir=.",
    "Train.dataset.label_file_list=[$trainLabel]",
    "Eval.dataset.data_dir=.",
    "Eval.dataset.label_file_list=[$valLabel]"
)

if ($ConfigMode -eq "ppocrv4" -and -not $ConfigPath) {
    $overrideArgs += "Train.sampler.first_bs=$BatchSize"
    $overrideArgs += "Train.sampler.fix_bs=True"
}

if ($characterDictPathResolved) {
    $overrideArgs += "Global.character_dict_path=$characterDictPathResolved"
}

if ($Resume) {
    $overrideArgs += "Global.checkpoints=$resumeCheckpoint"
    $overrideArgs += "Global.pretrained_model="
} else {
    if ($NoPretrained) {
        $overrideArgs += "Global.pretrained_model="
    } else {
        $overrideArgs += "Global.pretrained_model=$pretrainedModel"
    }
    $overrideArgs += "Global.checkpoints="
}

if ($DirectPython) {
    if ($env:CONDA_DEFAULT_ENV -ne $envName -or -not $env:CONDA_PREFIX) {
        Write-Host "DirectPython requires activated conda env: $envName"
        Write-Host "Current CONDA_DEFAULT_ENV: $env:CONDA_DEFAULT_ENV"
        exit 1
    }
    $pythonExe = Join-Path $env:CONDA_PREFIX "python.exe"
    Push-Location $repoAbs
    try {
        if ($Patience -gt 0) {
            $runnerLog = Join-Path $saveDir "early_stop_runner.log"
            & $pythonExe $earlyStopRunner --patience $Patience --log-file $runnerLog --cwd $repoAbs -- $pythonExe $trainPy -c $config -o @overrideArgs
        } else {
            & $pythonExe $trainPy -c $config -o @overrideArgs
        }
    } finally {
        Pop-Location
    }
} else {
    Push-Location $repoAbs
    try {
        if ($Patience -gt 0) {
            $runnerLog = Join-Path $saveDir "early_stop_runner.log"
            conda run -n $envName python $earlyStopRunner --patience $Patience --log-file $runnerLog --cwd $repoAbs -- python $trainPy -c $config -o @overrideArgs
        } else {
            conda run -n $envName python $trainPy -c $config -o @overrideArgs
        }
    } finally {
        Pop-Location
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Synthetic recognition training failed."
    exit 1
}
