param([string]$Category = "metal_nut", [int]$ImgSize = 640, [switch]$DataOnly)

$Repo = "C:\Users\barte\Documents\mvtec-anomaly-detection"
$Root = "C:\Users\barte\Documents\$Category"
$Log  = "run_yolo_folds_${Category}_log.txt"
Set-Location $Repo

if (-not (Test-Path "$Root\test")) { Write-Host "No folder $Root\test"; exit 1 }
if (-not (Test-Path "splits\${Category}_folds.csv")) {
    Write-Host "Creating splits\${Category}_folds.csv"
    & python make_splits.py --root $Root
}
$prefix = ""; if ($Category -ne "screw") { $prefix = "${Category}_" }
$dataPrefix = ""; if ($Category -ne "screw") { $dataPrefix = "$Category/" }
$start = Get-Date
$failed = New-Object System.Collections.ArrayList

foreach ($k in 0..3) {
    $run  = "${prefix}fold${k}_all_${ImgSize}"
    $data = "yolo_data/${dataPrefix}fold${k}"
    $t0 = Get-Date
    Write-Host ("[{0}] fold {1}: data" -f $t0.ToString("HH:mm:ss"), $k)
    if (-not (Test-Path "$data\data.yaml")) {
        & python make_yolo_data.py --root $Root --category $Category --fold $k 2>&1 | Out-File -FilePath $Log -Append -Encoding utf8
    }
    if ($DataOnly) { continue }

    $weights = "runs\detect\$run\weights\last.pt"
    if (-not (Test-Path $weights)) {
        Write-Host "   training $run"
        & yolo detect train data="$data/data.yaml" model=yolov8s.pt imgsz=$ImgSize epochs=100 batch=8 seed=42 val=False name=$run 2>&1 |
            Out-File -FilePath $Log -Append -Encoding utf8
    } else { Write-Host "   weights exist: $run" }

    $scoreFile = "scores_yolo\${run}_fold${k}.csv"
    if ((Test-Path $weights) -and -not (Test-Path $scoreFile)) {
        Write-Host "   scoring fold $k"
        & python yolo_scores.py --weights $weights --data-dir $data --root $Root --category $Category --fold $k --imgsz $ImgSize `
            --name $run --out "results/yolo_folds_${Category}.csv" 2>&1 | Out-File -FilePath $Log -Append -Encoding utf8
    }
    if (-not (Test-Path $scoreFile)) {
        Write-Host "   FAILED: fold $k (details in $Log)"; [void]$failed.Add("fold $k")
    } else {
        Write-Host ("   ok, {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)
    }
}

if (-not $DataOnly -and $failed.Count -eq 0) {
    & python merge_yolo_scores.py --category $Category --sizes $ImgSize
    & python eval_folds.py --category $Category
}
Write-Host ("FINISHED after {0:N1} h" -f ((Get-Date) - $start).TotalHours)
if ($failed.Count -gt 0) { Write-Host "Failed:"; $failed | ForEach-Object { Write-Host "  $_" } } else { Write-Host "No failed folds." }
