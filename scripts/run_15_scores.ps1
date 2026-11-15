$Repo = "C:\Users\barte\Documents\mvtec-anomaly-detection"
$Data = "C:\Users\barte\Documents"
$Out  = "results_15_scores.csv"
$Log  = "run_15_scores_log.txt"
$Seeds = @(42, 1, 2)
$Cats = @("bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut",
          "pill","screw","tile","toothbrush","transistor","wood","zipper")
$Models = @(
    @{ Name = "patchcore"; Extra = @("--model","patchcore","--backbone","wide_resnet50_2","--coreset-ratio","0.01") },
    @{ Name = "padim";     Extra = @("--model","padim","--backbone","wide_resnet50_2","--n-features","200") }
)

Set-Location $Repo
$failed  = New-Object System.Collections.ArrayList
$missing = New-Object System.Collections.ArrayList
$start = Get-Date

function Has-Row($cat, $model, $seed) {
    if (-not (Test-Path $Out)) { return $false }
    $rows = Import-Csv $Out | Where-Object { $_.category -eq $cat -and $_.model -eq $model -and $_.seed -eq "$seed" }
    return (@($rows).Count -gt 0)
}

foreach ($cat in $Cats) {
    $root = Join-Path $Data $cat
    if (-not (Test-Path (Join-Path $root "test"))) {
        Write-Host "SKIP $cat : no folder $root\test"
        [void]$missing.Add($cat)
        continue
    }
    foreach ($m in $Models) {
        foreach ($seed in $Seeds) {
            if (Has-Row $cat $m.Name $seed) { Write-Host "done already: $cat $($m.Name) seed $seed"; continue }
            $t0 = Get-Date
            Write-Host ("[{0}] {1} {2} seed {3}" -f $t0.ToString("HH:mm:ss"), $cat, $m.Name, $seed)
            $argList = @("run_experiment.py","--root",$root,"--category",$cat) + $m.Extra + @("--seeds","$seed","--save-scores","--out",$Out)
            & python @argList 2>&1 | Out-File -FilePath $Log -Append -Encoding utf8
            if ($LASTEXITCODE -ne 0 -or -not (Has-Row $cat $m.Name $seed)) {
                Write-Host "   FAILED: $cat $($m.Name) seed $seed (details in $Log)"
                [void]$failed.Add("$cat $($m.Name) seed $seed")
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
