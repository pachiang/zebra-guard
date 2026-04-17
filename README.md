# ZebraGuard

本機桌面應用程式,協助識別車輛未禮讓斑馬線行人之違規片段,產生檢舉輔助包(違規片段影片、關鍵幀、檢舉書草稿)。

- 平台:Windows 10/11(x64)
- 運算:全部在使用者本機執行,不上傳任何影片至雲端
- 授權:TBD

## 專案結構

```
zebra-guard/
├── src/zebraguard/   # 主程式 (PySide6 UI + Python ML 管線)
│   ├── ui/           # 介面層 (僅此層依賴 Qt)
│   ├── core/         # 專案、管線編排、背景 worker
│   ├── ml/           # 偵測、追蹤、幾何、違規規則 (無 Qt 依賴)
│   ├── export/       # 片段截取、PDF、檢舉書
│   ├── storage/      # SQLite、模型管理
│   └── utils/        # GPU 偵測、ffmpeg 路徑、log
├── resources/        # icons、翻譯、綁附模型與 ffmpeg
├── packaging/        # PyInstaller / Inno Setup / 簽章腳本
├── tests/
├── docs/             # 架構、MVP、法規、隱私、免責、打包文件
└── scripts/          # 開發、打包、benchmark 輔助腳本
```

## 文件索引

- [架構](docs/architecture.md) — 元件、資料流、執行緒模型、專案檔格式
- [MVP 功能清單](docs/mvp.md) — P0 / P1 / Phase 2 / Phase 3
- [法規依據](docs/legal-rules.md) — 未禮讓行人、三枕木紋、違規判定條件
- [隱私聲明](docs/privacy.md) — 使用者面向
- [免責聲明](docs/disclaimer.md) — 使用者面向
- [打包與簽章](docs/packaging.md) — Build pipeline、簽章延後規劃
- [使用手冊](docs/user-guide/README.md) — (撰寫中)

## 開發環境

TBD — 待決定 `uv` / `poetry` / `pip-tools`。先保留 `pyproject.toml` 空位。

## 快速開始

尚未實作。見 [MVP 功能清單](docs/mvp.md) 了解目前進度。
