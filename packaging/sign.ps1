# ==============================================================================
# Code Signing Script — CURRENTLY A PLACEHOLDER
# ==============================================================================
#
# Status : 延後實作。見 docs/packaging.md 「程式碼簽章 — 延後事項」。
# Trigger: 當有實際使用者因 SmartScreen / Defender 警告無法安裝,或進入
#          企業環境分發,再啟用此腳本。
#
# ------------------------------------------------------------------------------
# 未來要填的事 (To-Do when this hole gets filled):
#
#   1. 選定簽章方案:
#        - Azure Trusted Signing (推薦小團隊/個人)
#        - EV Code Signing (DigiCert / Sectigo / GlobalSign)
#   2. 憑證/金鑰存放:
#        - 本機開發:Windows Credential Manager
#        - CI:GitHub Actions secrets(或等效機制)
#   3. 實作以下動作:
#        (a) 呼叫 signtool.exe 簽 dist\ZebraGuard\zebraguard.exe (內部 exe)
#        (b) 呼叫 Inno Setup 產出 ZebraGuard-Setup-x.y.z.exe
#        (c) 再呼叫 signtool.exe 簽最終的 Setup.exe
#        (d) 使用 RFC 3161 timestamp server (timestamp.digicert.com 等)
#   4. 驗證:
#        - 在乾淨 VM 上雙擊,確認 SmartScreen 不再出現警告
#        - signtool verify /pa /all 檢查簽章鏈完整
#
# 範例 (未驗證,真正啟用時根據所選方案調整):
#
#   param(
#       [Parameter(Mandatory)][string]$ExePath,
#       [string]$TimestampUrl = "http://timestamp.digicert.com",
#       [string]$CertThumbprint = $env:ZEBRA_CODE_SIGN_THUMBPRINT
#   )
#
#   if (-not $CertThumbprint) {
#       throw "未設定簽章憑證 thumbprint;見 docs/packaging.md"
#   }
#
#   & signtool.exe sign `
#       /sha1 $CertThumbprint `
#       /tr $TimestampUrl /td sha256 /fd sha256 `
#       /a $ExePath
#
#   if ($LASTEXITCODE -ne 0) { throw "簽章失敗" }
#   & signtool.exe verify /pa /all $ExePath
# ==============================================================================

Write-Host "[sign.ps1] 程式碼簽章尚未實作,跳過。" -ForegroundColor Yellow
Write-Host "[sign.ps1] 詳見 docs/packaging.md。" -ForegroundColor Yellow
exit 0
