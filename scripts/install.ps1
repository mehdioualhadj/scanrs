$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
cargo build --release
$dest = Join-Path $env:LOCALAPPDATA 'scanrs'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item target\release\scanrs.exe $dest -Force
$p = [Environment]::GetEnvironmentVariable('Path','User')
if ($p -notlike "*$dest*") {
    [Environment]::SetEnvironmentVariable('Path', "$p;$dest", 'User')
}
$exe = Join-Path $dest 'scanrs.exe'
reg add "HKCU\Software\Classes\SystemFileAssociations\image\shell\scanrs" /ve /d "Scan with scanrs" /f | Out-Null
reg add "HKCU\Software\Classes\SystemFileAssociations\image\shell\scanrs\command" /ve /d "`"$exe`" `"%1`"" /f | Out-Null
Pop-Location
Write-Host "installed: $exe"
Write-Host "open a NEW terminal for PATH. right-click any image -> 'Scan with scanrs'"

