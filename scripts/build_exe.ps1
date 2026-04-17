# ==============================================================================
# Build Script — 目前為 placeholder
# ==============================================================================
#
# 預期流程(見 docs/packaging.md):
#   1. 執行 PyInstaller 產出 dist\ZebraGuard\
#   2. 執行 Inno Setup 產出 dist\ZebraGuard-Setup-x.y.z.exe
#   3. 呼叫 packaging\sign.ps1 簽章(目前 no-op)
#
# 啟用前:
#   - 選定依賴管理工具並產生 requirements.txt / pyproject.toml
#   - 完成 packaging/pyinstaller.spec
#   - 完成 packaging/installer.iss
#   - 安裝 Inno Setup 6(若要本機 build)

param(
    [switch]$SkipSign
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "[build_exe] 尚未實作。見 docs/packaging.md。" -ForegroundColor Yellow
Write-Host "[build_exe] 預期步驟:"
Write-Host "           1. pyinstaller $Root\packaging\pyinstaller.spec"
Write-Host "           2. iscc $Root\packaging\installer.iss"
if (-not $SkipSign) {
    Write-Host "           3. & $Root\packaging\sign.ps1"
}
exit 1
