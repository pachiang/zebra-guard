# 打包與發佈

## 目標

產出一個單一的 `ZebraGuard-Setup-x.y.z.exe` 安裝檔,Windows 10/11 x64 使用者下載後雙擊即可安裝到 `%LOCALAPPDATA%\Programs\ZebraGuard\`,無需系統管理員權限。

## 工具鏈

```
src/ + resources/
       │
       │ PyInstaller (packaging/pyinstaller.spec)
       ▼
dist/ZebraGuard/            ← 資料夾:zebraguard.exe + DLLs + 資源
       │
       │ Inno Setup (packaging/installer.iss)
       ▼
dist/ZebraGuard-Setup.exe   ← 最終安裝檔
       │
       │ [DEFERRED] 簽章 (packaging/sign.ps1)
       ▼
已簽章的安裝檔
```

## 步驟

### 1. 準備執行環境

```powershell
# 建議用 uv 或 venv;此處以 venv 為例
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pyinstaller
```

### 2. 下載綁附資源(未納入版本控管)

```powershell
python scripts/download_models.py --target resources/models
# 下載 ffmpeg 靜態編譯版到 resources/ffmpeg/
```

### 3. PyInstaller 打包

```powershell
pyinstaller packaging/pyinstaller.spec
```

`pyinstaller.spec` 需要設定:
- `hiddenimports` 包含 `ultralytics`、`onnxruntime`、`PySide6`、`cv2` 等套件的動態載入模組
- `datas` 包含 `resources/icons/`、`resources/translations/`、`resources/models/yolo11n.pt`(或未來轉出的 .onnx)、`resources/ffmpeg/`
- `excludes` 排除 `matplotlib`、`pandas`、`scipy` 中不使用的部分以減少體積

### 4. Inno Setup 安裝檔

```powershell
# 需要先安裝 Inno Setup 6: https://jrsoftware.org/isinfo.php
iscc packaging/installer.iss
```

`installer.iss` 需要設定:
- `PrivilegesRequired=lowest` — 免管理員權限
- `DefaultDirName={localappdata}\Programs\ZebraGuard`
- `UsePreviousAppDir=yes` — 升級沿用原安裝路徑
- `Uninstallable=yes` 且解除安裝時詢問是否保留 `%APPDATA%\ZebraGuard\` 與使用者專案檔

### 5. (延後) 簽章

見下節。

---

## 程式碼簽章 — 延後事項

### 現階段決定:**不做簽章**

理由:
- 簽章憑證需申請流程,取得前無法開始
- 沒簽章的 exe 仍可執行,使用者會看到 SmartScreen 警告但可點「其他資訊 → 仍要執行」
- 早期使用者對警告有較高容忍度,未簽章對 MVP 驗證影響小
- 簽章流程是打包後的最後一步,未來導入不影響程式碼本身

### 未來實作時的選項

| 方案 | 年費 `[需再查證]` | 備註 |
|---|---|---|
| Azure Trusted Signing | ~USD 120 | 微軟官方,2024 起 GA,小公司 / 個人開發者最划算 |
| EV Code Signing (DigiCert 等) | USD 300+ | 即刻建立 SmartScreen 信譽,適合有營業登記者 |
| OV Code Signing | USD 150+ | 需累積信譽才能消除 SmartScreen 警告 |

### 屆時需要做的事

填入 `packaging/sign.ps1`,呼叫 `signtool.exe` 或 Azure Trusted Signing 的 CLI:

```powershell
# packaging/sign.ps1 — 目前為 placeholder
# 未來應接收 $exe 參數並執行:
# signtool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 `
#   /a "$exe"
# 憑證、金鑰、或 Azure Trusted Signing credentials 請勿 commit 到 repo
```

建議做法:
1. 憑證 / credentials 存在 CI secret 或本機的 Windows Credential Manager
2. `sign.ps1` 讀取 secret 後呼叫 signtool
3. 主 build pipeline 在 Inno Setup 產出後呼叫 `sign.ps1 -Exe dist/ZebraGuard-Setup.exe`
4. 也要簽章 PyInstaller 產出的內部 `zebraguard.exe`(否則使用者執行時仍會警告)

### 使用者體驗:簽章前 vs 簽章後

| 情境 | 未簽章 | 已簽章 |
|---|---|---|
| 下載 exe 後雙擊 | SmartScreen 警告「未知的發行者」,需點「其他資訊」展開才能執行 | 直接執行 |
| Defender 掃描 | 偶發誤判隔離 | 幾乎無誤判 |
| 企業 AppLocker 環境 | 通常無法執行 | 依政策可能可執行 |

MVP 階段在 README 和首次啟動教學裡教使用者「點其他資訊 → 仍要執行」即可。

---

## 版本管理

- `packaging/version_info.txt` 為 Windows 版本資源(影響右鍵內容的版本資訊)
- 版號來源:`src/zebraguard/__init__.py` 的 `__version__`
- Build script 會把版號同步寫入 `version_info.txt`、`installer.iss`

## 減少安裝檔體積的方向

目前預估 PyInstaller + torch + onnxruntime + PySide6 + ffmpeg ≈ 400-800 MB。可優化:

- 使用 **onnxruntime-directml** 且不裝 torch(inference 只用 ONNX)
- 不裝 ultralytics,自己寫 YOLO ONNX pre/post-process(少 ~100 MB 但要自己維護)
- 考慮 **Nuitka** 取代 PyInstaller(產物較小、較快,但編譯時間長)
- ffmpeg 用最小組態(不帶 GPL 解碼器)

## 測試

- 在乾淨的 Windows 11 VM(無 Python、無 Visual C++ Redist)上測試安裝
- 檢查 SmartScreen 行為
- 檢查 Defender 是否誤判(新版本有時要等幾天才會建立信譽)

## 待辦

- [ ] 選定依賴管理工具並產生 `requirements.txt` / `pyproject.toml`
- [ ] 撰寫 `packaging/pyinstaller.spec` 的實際內容
- [ ] 撰寫 `packaging/installer.iss` 的實際內容
- [ ] 撰寫 CI 流程(GitHub Actions build-on-release)
