# 强制开启报错即停止
$ErrorActionPreference = "Stop"

Write-Host "开始提取特征..." -ForegroundColor Cyan
python scripts/export_ml_features.py

Write-Host "开始训练模型..." -ForegroundColor Cyan
python scripts/train_svm_baseline.py

Write-Host "开始生成评估图表..." -ForegroundColor Cyan
python scripts/plot_metrics.py

Write-Host "所有任务已完成！Good Morning!" -ForegroundColor Green