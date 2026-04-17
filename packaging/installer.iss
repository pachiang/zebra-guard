; Inno Setup script placeholder — 詳見 docs/packaging.md
;
; 真正啟用前要做:
;   1. 把 {#MyAppVersion} 接到 src/zebraguard/__init__.py 的 __version__
;   2. 決定 AppId(GUID),產生後固定不要再改(解除安裝依此辨識)
;   3. 測試乾淨 Win11 VM 上的安裝/解除安裝流程
;   4. 解除安裝時詢問是否保留 %APPDATA%\ZebraGuard 與使用者專案檔
;
; #define MyAppName "ZebraGuard"
; #define MyAppVersion "0.1.0"
; #define MyAppPublisher "TBD"
; #define MyAppExeName "zebraguard.exe"
;
; [Setup]
; AppId={{REPLACE-WITH-A-FIXED-GUID}}
; AppName={#MyAppName}
; AppVersion={#MyAppVersion}
; AppPublisher={#MyAppPublisher}
; DefaultDirName={localappdata}\Programs\{#MyAppName}
; DefaultGroupName={#MyAppName}
; PrivilegesRequired=lowest
; PrivilegesRequiredOverridesAllowed=dialog
; OutputDir=..\dist
; OutputBaseFilename=ZebraGuard-Setup-{#MyAppVersion}
; Compression=lzma2
; SolidCompression=yes
; WizardStyle=modern
; UsePreviousAppDir=yes
; DisableDirPage=auto
; SetupIconFile=..\resources\icons\app.ico
;
; [Languages]
; Name: "cht"; MessagesFile: "compiler:Languages\ChineseTraditional.isl"
;
; [Files]
; Source: "..\dist\ZebraGuard\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
;
; [Icons]
; Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
; Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
;
; [Tasks]
; Name: "desktopicon"; Description: "建立桌面捷徑"; GroupDescription: "附加工作:"
;
; [Run]
; Filename: "{app}\{#MyAppExeName}"; Description: "啟動 {#MyAppName}"; Flags: nowait postinstall skipifsilent
