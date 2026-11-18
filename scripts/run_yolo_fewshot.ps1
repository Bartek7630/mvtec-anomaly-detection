# YOLOv8s "few defective images" on one MVTec AD category.
# For every N (number of DEFECTIVE training images), every draw (seed) and every fold k = 0..3:
#   1. build the data:  make_yolo_data.py --n-defect N --seed S   (N defective + as many good images as the pool has)
#   2. train:           epochs = ceil(1500 / ceil(n_train_images / 8)), so every run has about 1500 iterations
#   3. score the held-out fold:  yolo_scores.py  -> one row in results\yolo_fewshot_<category>.csv
#   4. delete the weights and the data folder of this run (disk space)
# Resumable: a run whose name already has a row in the CSV is skipped. Settings are FIXED, do not change them after seeing results.

param([string]$Category = "metal_nut", [string]$Ns = "10,50", [string]$Seeds = "42,1,2", [int]$ImgSize = 640, [switch]$Quick)

$Repo = "C:\Users\barte\Documents\mvtec-anomaly-detection"
$Root = "C:\Users\barte\Documents\$Category"
$Log  = "run_yolo_fewshot_${Category}_log.txt"
Set-Location $Repo

if (-not (Test-Path "$Root\test")) { Write-Host "No folder $Root\test"; exit 1 }
if (-not (Test-Path "splits\${Category}_folds.csv")) { Write-Host "No splits\${Category}_folds.csv (run make_splits.py first)"; exit 1 }

$nList    = @($Ns    -split "," | ForEach-Object { [int]$_.Trim() })
$seedList = @($Seeds -split "," | ForEach-Object { [int]$_.Trim() })
$folds    = 0..3
$Out      = "results\yolo_fewshot_${Category}.csv"
$suffix   = ""
if ($Quick) { $nList = @($nList[0]); $seedList = @($seedList[0]); $folds = @(0); $Out = "results\yolo_quick.csv"; $suffix = "_quick" }

$prefix = ""; if ($Category -ne "screw") { $prefix = "${Category}_" }
$dataPrefix = ""; if ($Category -ne "screw") { $dataPrefix = "$Category/" }

function Has-Row($run) {
    return ((Test-Path $Out) -and (Select-String -Path $Out -Pattern ("^" + [regex]::Escape($run) + ",") -Quiet))
}

$total = $nList.Count * $seedList.Count * $folds.Count
$done = 0
$start = Get-Date
$failed = New-Object System.Collections.ArrayList
Write-Host ("{0} runs planned: N = {1}, seeds = {2}, folds = {3}" -f $total, ($nList -join ","), ($seedList -join ","), ($folds -join ","))

foreach ($n in $nList) {
  foreach ($s in $seedList) {
    foreach ($k in $folds) {
        $done++
        $run  = "${prefix}fold${k}_n${n}_seed${s}_${ImgSize}${suffix}"
        $data = "yolo_data/${dataPrefix}fold${k}_n${n}_seed${s}"
        if (Has-Row $run) { Write-Host "[$done/$total] $run already done"; continue }

        $t0 = Get-Date
        Write-Host ("[{0}] [{1}/{2}] {3}" -f $t0.ToString("HH:mm:ss"), $done, $total, $run)
        & python make_yolo_data.py --root $Root --category $Category --fold $k --n-defect $n --seed $s --preview 0 2>&1 |
            Out-File -FilePath $Log -Append -Encoding utf8
        if (-not (Test-Path "$data\data.yaml")) { Write-Host "   FAILED: data"; [void]$failed.Add($run); continue }

        $nTrain = (Get-ChildItem "$data\images\train" -File).Count
        $epochs = [int][math]::Ceiling(1500 / [math]::Ceiling($nTrain / 8))
        if ($Quick) { $epochs = 2 }
        Write-Host "   $nTrain training images, $epochs epochs"

        if (Test-Path "runs\detect\$run") { Remove-Item -Recurse -Force "runs\detect\$run" }
        & yolo detect train data="$data/data.yaml" model=yolov8s.pt imgsz=$ImgSize epochs=$epochs batch=8 seed=$s val=False name=$run 2>&1 |
            Out-File -FilePath $Log -Append -Encoding utf8

        $weights = "runs\detect\$run\weights\last.pt"
        if (Test-Path $weights) {
            & python yolo_scores.py --weights $weights --data-dir $data --root $Root --category $Category --fold $k --imgsz $ImgSize `
                --name $run --n-defect $n --train-seed $s --out $Out 2>&1 | Out-File -FilePath $Log -Append -Encoding utf8
        }
        if (Has-Row $run) {
            Write-Host ("   ok, {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)
            Remove-Item -Recurse -Force "runs\detect\$run\weights" -ErrorAction SilentlyContinue
            Remove-Item -Recurse -Force $data -ErrorAction SilentlyContinue
        } else {
            Write-Host "   FAILED (details in $Log)"; [void]$failed.Add($run)
        }
    }
  }
}

if ($Quick) {
    Write-Host "Quick test finished. Look at the line above: if it says ok, delete the test files:"
    Write-Host "  Remove-Item results\yolo_quick.csv; Remove-Item scores_yolo\*_quick_*.csv; Remove-Item -Recurse runs\detect\*_quick"
}
Write-Host ("FINISHED after {0:N1} h" -f ((Get-Date) - $start).TotalHours)
if ($failed.Count -gt 0) { Write-Host "Failed:"; $failed | ForEach-Object { Write-Host "  $_" } } else { Write-Host "No failed runs." }
