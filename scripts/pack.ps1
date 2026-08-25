$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
$out = Join-Path $root 'dist\scanrs-workstation'
Remove-Item -Recurse -Force $out -ErrorAction Ignore
New-Item -ItemType Directory -Force -Path "$out\bin" | Out-Null
Write-Host "building scanrs..."
cargo build --release
Copy-Item target\release\scanrs.exe "$out\bin\" -Force
Write-Host "collecting tesseract..."
$tess = (Get-Command tesseract).Source
$tessDir = Split-Path -Parent $tess
Copy-Item $tessDir "$out\tesseract" -Recurse -Force
if (-not (Test-Path "$out\tesseract\tessdata\eng.traineddata")) {
    throw "eng.traineddata not found next to $tess - copy tessdata into the package manually"
}
Copy-Item scripts\workstation_install.template "$out\install.ps1" -Force
Copy-Item scripts\AUTOUNATTEND_NOTE.txt "$out\" -Force
Write-Host "package ready: $out"
Pop-Location
