# reproduce_all.ps1 —— 一键完整复现：顺序运行第3、4、5章全部核心实验
# 用法: powershell -ExecutionPolicy Bypass -File scripts/reproduce_all.ps1

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\.."

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  CagePredict 一键完整复现" -ForegroundColor Cyan
Write-Host "  顺序运行第 3、4、5 章全部核心实验（完整参数）" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $ProjectRoot
python -m cage_predict run-all

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "完整复现中有实验失败，请检查上方日志." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "全部实验完成." -ForegroundColor Green
