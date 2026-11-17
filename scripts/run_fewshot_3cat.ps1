param([string]$Only = "", [switch]$Quick)

$Repo = "C:\Users\barte\Documents\mvtec-anomaly-detection"
$Data = "C:\Users\barte\Documents"
$Out  = "results_fewshot_3cat.csv"
$Log  = "run_fewshot_3cat_log.txt"
$Cats = @("metal_nut", "transistor", "carpet")
if ($Only -ne "") { $Cats = @($Only) }
$Seeds = @(42, 1, 2)

$Jobs = @()
foreach ($n in @(1, 5, 10, 50)) {
    $Jobs += @{ Model = "patchcore"; N = $n; Seeds = $Seeds
                Extra = @("--model","patchcore","--backbone","wide_resnet50_2","--coreset-ratio","1.0","--n-train","$n") }
}
$Jobs += @{ Model = "winclip"; N = 0; Seeds = @(42); Extra = @("--model","winclip","--k-shot","0") }
foreach ($k in @(1, 5, 10)) {
    $Jobs += @{ Model = "winclip"; N = $k; Seeds = $Seeds; Extra = @("--model","winclip","--k-shot","$k") }
}
if ($Quick) { $Jobs = @($Jobs[0]); $Seeds = @(42) }   # one PatchCore run (N = 1, seed 42) to test the script

Set-Location $Repo
$failed  = New-Object System.Collections.ArrayList
$missing = New-Object System.Collections.ArrayList
$start = Get-Date

function Has-Row($cat, $model, $n, $seed) {
    if (-not (Test-Path $Out)) { return $false }
    $rows = Import-Csv $Out | Where-Object { $_.category -eq $cat -and $_.model -eq $model -and $_.n_train -eq "$n" -and $_.seed -eq "$seed" }
    return (@($rows).Count -gt 0)
}

foreach ($cat in $Cats) {
    $root = Join-Path $Data $cat
    if (-not (Test-Path (Join-Path $root "test"))) {
        Write-Host "SKIP $cat : no folder $root\test"
        [void]$missing.Add($cat)
        continue
    }
    foreach ($job in $Jobs) {
        $jobSeeds = $job.Seeds
        if ($Quick) { $jobSeeds = @(42) }
        foreach ($seed in $jobSeeds) {
            if (Has-Row $cat $job.Model $job.N $seed) { Write-Host "done already: $cat $($job.Model) N=$($job.N) seed $seed"; continue }
            $t0 = Get-Date
            Write-Host ("[{0}] {1} {2} N={3} seed {4}" -f $t0.ToString("HH:mm:ss"), $cat, $job.Model, $job.N, $seed)
            $argList = @("run_experiment.py","--root",$root,"--category",$cat) + $job.Extra + @("--seeds","$seed","--save-scores","--out",$Out)
            & python @argList 2>&1 | Out-File -FilePath $Log -Append -Encoding utf8
            if ($LASTEXITCODE -ne 0 -or -not (Has-Row $cat $job.Model $job.N $seed)) {
                Write-Host "   FAILED: $cat $($job.Model) N=$($job.N) seed $seed (details in $Log)"
                [void]$failed.Add("$cat $($job.Model) N=$($job.N) seed $seed")
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
