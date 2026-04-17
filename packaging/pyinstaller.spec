# PyInstaller spec placeholder.
#
# 目前尚未實作。真正要用時:
#   1. 決定依賴管理工具並產生 requirements.txt
#   2. 確認需要的 hidden imports (ultralytics / onnxruntime / cv2 通常需要)
#   3. datas 納入 resources/ 目錄 (icons, translations, models, ffmpeg)
#   4. 調整 excludes 以瘦身
#
# 詳見 docs/packaging.md。
#
# 範例框架(未驗證,真正啟用前先測):
#
# # -*- mode: python ; coding: utf-8 -*-
# block_cipher = None
# a = Analysis(
#     ['..\\src\\zebraguard\\__main__.py'],
#     pathex=['..\\src'],
#     binaries=[],
#     datas=[
#         ('..\\resources\\icons', 'resources\\icons'),
#         ('..\\resources\\translations', 'resources\\translations'),
#         ('..\\resources\\models\\yolo11n.pt', 'resources\\models'),
#         ('..\\resources\\ffmpeg', 'resources\\ffmpeg'),
#     ],
#     hiddenimports=['onnxruntime', 'PySide6', 'cv2'],
#     hookspath=[],
#     runtime_hooks=[],
#     excludes=['matplotlib', 'pandas', 'scipy', 'tkinter'],
#     win_no_prefer_redirects=False,
#     win_private_assemblies=False,
#     cipher=block_cipher,
#     noarchive=False,
# )
# pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
# exe = EXE(
#     pyz, a.scripts, [],
#     exclude_binaries=True,
#     name='zebraguard',
#     debug=False,
#     bootloader_ignore_signals=False,
#     strip=False,
#     upx=False,
#     console=False,
#     icon='..\\resources\\icons\\app.ico',
#     version='version_info.txt',
# )
# coll = COLLECT(
#     exe, a.binaries, a.zipfiles, a.datas,
#     strip=False, upx=False, upx_exclude=[],
#     name='ZebraGuard',
# )
