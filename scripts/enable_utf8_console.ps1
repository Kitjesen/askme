# Enable UTF-8 for the current PowerShell session only.
# This does not edit the user's PowerShell profile.

chcp.com 65001 > $null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "UTF-8 console enabled for this PowerShell session."
