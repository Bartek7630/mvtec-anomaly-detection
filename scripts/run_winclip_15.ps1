# Day 2.12: WinCLIP on all 15 MVTec AD categories: zero-shot (1 run) and 10 reference images (3 draws).
# Resumable: runs already present in results_winclip_15.csv are skipped.
# Try one category first:   powershell -ExecutionPolicy Bypass -File scripts\run_winclip_15.ps1 -Only screw
# Everything:               powershell -ExecutionPolicy Bypass -File scripts\run_winclip_15.ps1
param([string]$Only = "")

$Repo = "C:\Users\barte\Documents\mvtec-anomaly-detection"
$Data = "C:\Users\barte\Documents"        # folder with one subfolder per category
$Out  = "results_winclip_15.csv"
$Log  = "run_winclip_15_log.txt"
$Cats = @("bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut",
          "pill","screw","tile","toothbrush","transistor","wood","zipper")
if ($Only -ne "") { $Cats = @($Only) }

# (k_shot, seeds): zero-shot is deterministic, so one run; 10 reference images: 3 random draws
$Configs = @( @{ K = 0; Seeds = @(42) }, @{ K = 10; Seeds = @(42, 1, 2) } )

Set-Location $Repo
$failed  = New-Object System.Collections.ArrayList
$missing = New-Object System.Collections.ArrayList
$start = Get-Date

function Has-Row($cat, $k, $seed) {
    if (-not (Test-Path $Out)) { return $false }
    $rows = Import-Csv $Out | Where-Object { $_.category -eq $cat -and $_.n_train -eq "$k" -and $_.seed -eq "$seed" }
    return (@($rows).Count -gt 0)
}

foreach ($cat in $Cats) {
    $root = Join-Path $Data $cat
    if (-not (Test-Path (Join-Path $root "test"))) {
        Write-Host "SKIP $cat : no folder $root\test"
        [void]$missing.Add($cat)
        continue
    }
    foreach ($cfg in $Configs) {
        foreach ($seed in $cfg.Seeds) {
            if (Has-Row $cat $cfg.K $seed) { Write-Host "done already: $cat k=$($cfg.K) seed $seed"; continue }
            $t0 = Get-Date
            Write-Host ("[{0}] {1} k={2} seed {3}" -f $t0.ToString("HH:mm:ss"), $cat, $cfg.K, $seed)
            $argList = @("run_experiment.py","--root",$root,"--category",$cat,"--model","winclip",
                         "--k-shot","$($cfg.K)","--seeds","$seed","--save-scores","--out",$Out)
            & python @argList 2>&1 | Out-File -FilePath $Log -Append -Encoding utf8
            if ($LASTEXITCODE -ne 0 -or -not (Has-Row $cat $cfg.K $seed)) {
                Write-Host "   FAILED: $cat k=$($cfg.K) seed $seed (details in $Log)"
                [void]$failed.Add("$cat k=$($cfg.K) seed $seed")
            } else {
                Write-Host ("   ok, {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)
            }
	    Get-ChildItem runs -Recurse -Filter *.ckpt -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host ""
Write-Host ("FINISHED after {0:N1} h" -f ((Get-Date) - $start).TotalHours)
if ($missing.Count -gt 0) { Write-Host ("Categories without data: " + ($missing -join ", ")) }
if ($failed.Count -gt 0)  { Write-Host "Failed runs:"; $failed | ForEach-Object { Write-Host "  $_" } } else { Write-Host "No failed runs." }
