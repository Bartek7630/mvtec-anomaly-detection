# EfficientAD on screw, full data, seeds 1 and 2 (seed 42 is in results\efficientad_screw.csv).
# Same settings as the November run: 70,000 steps, 10% of the good training images held out for map normalisation
# Resumable: a seed that already has a row in results_efficientad_seeds.csv is skipped.
#
#   Test (about 2 min, results thrown away):  powershell -ExecutionPolicy Bypass -File scripts\run_efficientad_seeds.ps1 -Quick
#   Real run:                                 powershell -ExecutionPolicy Bypass -File scripts\run_efficientad_seeds.ps1
param([string]$Seeds = "1,2", [switch]$Quick)

$Repo = "C:\Users\barte\Documents\mvtec-anomaly-detection"
$Root = "C:\Users\barte\Documents\screw"
$Out  = "results_efficientad_seeds.csv"
$Log  = "run_efficientad_seeds_log.txt"
Set-Location $Repo

if (-not (Test-Path "$Root\test")) { Write-Host "No folder $Root\test"; exit 1 }
if (-not (Test-Path "datasets\imagenette")) { Write-Host "No datasets\imagenette (EfficientAD needs it)"; exit 1 }
$seedList = @($Seeds -split "," | ForEach-Object { [int]$_.Trim() })
$extra = @()
$scoresDir = "scores"
if ($Quick) { $seedList = @($seedList[0]); $Out = "results_ea_quick.csv"; $scoresDir = "scores_quick"; $extra = @("--iterations", "200") }

function Has-Row($seed) {
    if (-not (Test-Path $Out)) { return $false }
    return (@(Import-Csv $Out | Where-Object { $_.seed -eq "$seed" }).Count -gt 0)
}

$failed = New-Object System.Collections.ArrayList
$start = Get-Date
foreach ($seed in $seedList) {
    if (Has-Row $seed) { Write-Host "seed $seed already done"; continue }
    $t0 = Get-Date
    Write-Host ("[{0}] EfficientAD screw seed {1}" -f $t0.ToString("HH:mm:ss"), $seed)
    & python run_experiment.py --root $Root --category screw --model efficientad --imagenet-dir datasets/imagenette `
        --seeds $seed --save-scores --scores-dir $scoresDir --out $Out @extra 2>&1 | Out-File -FilePath $Log -Append -Encoding utf8
    if (Has-Row $seed) { Write-Host ("   ok, {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes) }
    else { Write-Host "   FAILED (details in $Log)"; [void]$failed.Add("seed $seed") }
    Get-ChildItem runs -Recurse -Filter *.ckpt -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}
if ($Quick) {
    Write-Host "Quick test done. If it says ok, clean up:  Remove-Item results_ea_quick.csv; Remove-Item -Recurse scores_quick"
}
Write-Host ("FINISHED after {0:N1} h" -f ((Get-Date) - $start).TotalHours)
if ($failed.Count -gt 0) { Write-Host "Failed:"; $failed | ForEach-Object { Write-Host "  $_" } } else { Write-Host "No failed runs." }
