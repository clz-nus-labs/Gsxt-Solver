param(
    [string]$Source = "Scripts\Gsxt\data\semantic_lexicons\raw_corpus",
    [string]$Output = "Scripts\Gsxt\data\semantic_lexicons\generated\large_corpus_phrases.txt",
    [int]$MinLen = 2,
    [int]$MaxLen = 5,
    [int]$MinCount = 2,
    [int]$TopK = 200000,
    [switch]$UseJieba
)

$ErrorActionPreference = "Stop"

if (-not $env:CONDA_PREFIX) {
    throw "Please activate paddlex_cv first: conda activate paddlex_cv"
}

$python = Join-Path $env:CONDA_PREFIX "python.exe"
$argsList = @(
    "Scripts\Gsxt\tools\build_semantic_lexicon.py",
    "--source", $Source,
    "--output", $Output,
    "--min-len", "$MinLen",
    "--max-len", "$MaxLen",
    "--min-count", "$MinCount",
    "--top-k", "$TopK"
)

if ($UseJieba) {
    $argsList += "--use-jieba"
}

& $python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Build semantic lexicon failed."
}

Write-Host "Semantic lexicon ready: $Output"
