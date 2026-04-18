; Inno Setup script for TranslatorHoi4
; Usage: Compile with Inno Setup 6.x or later (ISCC.exe)

#define MyAppName "TranslatorHoi4"
#ifndef APP_VERSION
#define MyAppVersion "0.0.1"
#else
#define MyAppVersion APP_VERSION
#endif
#define MyAppPublisher "Locon213"
#define MyAppURL "https://github.com/Locon213/TranslatorHoi4"
#define MyAppExeName "TranslatorHoi4.exe"

#ifndef APP_ARCH
#define AppArchitecture "x64"
#else
#define AppArchitecture APP_ARCH
#endif

#ifndef APP_SOURCE_DIR
#define AppSourceDir "..\dist\TranslatorHoi4"
#else
#define AppSourceDir APP_SOURCE_DIR
#endif

#ifndef APP_OUTPUT_STEM
#define AppOutputStem "TranslatorHoi4_Setup"
#else
#define AppOutputStem APP_OUTPUT_STEM
#endif

; Pre-defined paths to avoid macro nesting issues
#define AppInstallDir "{autopf}\TranslatorHoi4"
#define AppUninstallIcon "{app}\TranslatorHoi4.exe"
#define AppSourceExe AppSourceDir + "\TranslatorHoi4.exe"
#define AppIconSource "..\assets\icon.ico"
#define AppLicenseSource "..\LICENSE"
#define AppOutputDir "..\dist"
#define AppOutputName AppOutputStem + "_" + MyAppVersion
#define AppGroupIcon "{group}\TranslatorHoi4"
#define AppDesktopIcon "{autodesktop}\TranslatorHoi4"
#define AppRunExe "{app}\TranslatorHoi4.exe"
#define AppUninstallMenu "{group}\Uninstall TranslatorHoi4"

[Setup]
AppId={{A8B3C2D1-E4F5-6789-ABCD-EF0123456789}
AppName=TranslatorHoi4
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={#AppInstallDir}
DefaultGroupName=TranslatorHoi4
AllowNoIcons=yes
LicenseFile={#AppLicenseSource}
OutputDir={#AppOutputDir}
OutputBaseFilename={#AppOutputName}
SetupIconFile={#AppIconSource}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed={#AppArchitecture}
ArchitecturesInstallIn64BitMode={#AppArchitecture}
UninstallDisplayIcon={app}\TranslatorHoi4.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#AppSourceExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#AppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{#AppGroupIcon}"; Filename: "{#AppRunExe}"
Name: "{#AppUninstallMenu}"; Filename: "{uninstallexe}"
Name: "{#AppDesktopIcon}"; Filename: "{#AppRunExe}"; Tasks: desktopicon

[Run]
Filename: "{#AppRunExe}"; Description: "{cm:LaunchProgram,TranslatorHoi4}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Post-install tasks can go here
  end;
end;
