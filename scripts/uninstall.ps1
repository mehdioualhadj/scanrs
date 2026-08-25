$dest = Join-Path $env:LOCALAPPDATA 'scanrs'
reg delete "HKCU\Software\Classes\SystemFileAssociations\image\shell\scanrs" /f 2>$null | Out-Null
$p = [Environment]::GetEnvironmentVariable('Path','User')
[Environment]::SetEnvironmentVariable('Path', (($p -split ';' | Where-Object { $_ -ne $dest }) -join ';'), 'User')
Remove-Item -Recurse -Force $dest -ErrorAction Ignore
Write-Host 'uninstalled'
