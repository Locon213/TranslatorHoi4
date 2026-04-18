param(
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$Architecture,
    [Parameter(Mandatory = $true)][string]$SourceDir
)

$ErrorActionPreference = "Stop"

choco install innosetup --no-progress
python -m pip install Pillow

python -c @"
from PIL import Image
img = Image.open('assets/icon.png')
img.save('assets/icon.ico', format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
"@

if (-not (Test-Path "assets\icon.ico")) {
    throw "Icon file assets\icon.ico not found after conversion"
}

$outputStem = if ($Architecture -eq "arm64") { "TranslatorHoi4_Setup_arm64" } else { "TranslatorHoi4_Setup" }
& iscc `
    "/DAPP_VERSION=$Version" `
    "/DAPP_ARCH=$Architecture" `
    "/DAPP_SOURCE_DIR=$SourceDir" `
    "/DAPP_OUTPUT_STEM=$outputStem" `
    packaging\translatorhoi4-setup.iss
