[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$BuildRoot = (Join-Path $PSScriptRoot "..\.build")
)

$ErrorActionPreference = "Stop"
$commit = "846fe90a289f58b7c9303a635142aa2c7caa93e5"
$repository = "https://gitlab.freedesktop.org/pulseaudio/webrtc-audio-processing.git"
$source = Join-Path $BuildRoot "source"
$build = Join-Path $BuildRoot "meson-build"
$prefix = Join-Path $BuildRoot "prefix"

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$Arguments = @()
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Native command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

New-Item -ItemType Directory -Force $BuildRoot | Out-Null
if (-not (Test-Path (Join-Path $source ".git"))) {
    New-Item -ItemType Directory -Force $source | Out-Null
    Invoke-NativeCommand git @("-C", $source, "init")
    Invoke-NativeCommand git @("-C", $source, "remote", "add", "origin", $repository)
}
# Fetch the pinned tree directly.  A blobless clone of the repository's
# default branch makes Git lazily request this older commit one blob at a time
# from GitLab, which is substantially slower and fragile on Windows.
git -C $source config --unset-all remote.origin.promisor 2>$null
if ($LASTEXITCODE -notin @(0, 5)) {
    throw "Could not clear Git promisor configuration (exit ${LASTEXITCODE})"
}
git -C $source config --unset-all remote.origin.partialclonefilter 2>$null
if ($LASTEXITCODE -notin @(0, 5)) {
    throw "Could not clear Git partial-clone configuration (exit ${LASTEXITCODE})"
}
Invoke-NativeCommand git @("-C", $source, "fetch", "--depth", "1", "origin", $commit)
Invoke-NativeCommand git @("-C", $source, "checkout", "--detach", $commit)
$actualCommit = (Invoke-NativeCommand git @("-C", $source, "rev-parse", "HEAD")).Trim()
if ($actualCommit -ne $commit) {
    throw "WebRTC APM checkout mismatch: expected $commit, got $actualCommit"
}

$setupArgs = @(
    "-m", "mesonbuild.mesonmain", "setup",
    "--vsenv",
    "--prefix", $prefix,
    "--buildtype", "release",
    "--wrap-mode", "forcefallback",
    "-Ddefault_library=shared",
    # WebRTC APM v2.1 contains a designated initializer.  GCC/Clang accept it
    # as a C++17 extension, while MSVC correctly requires C++20.
    "-Dcpp_std=c++20",
    $build,
    $source
)
if (Test-Path (Join-Path $build "meson-private\coredata.dat")) {
    $setupArgs = $setupArgs[0..2] + @("--wipe") + $setupArgs[3..($setupArgs.Length - 1)]
}
Invoke-NativeCommand $Python $setupArgs
Invoke-NativeCommand $Python @("-m", "mesonbuild.mesonmain", "compile", "-C", $build)
Invoke-NativeCommand $Python @("-m", "mesonbuild.mesonmain", "install", "-C", $build)

$stampDirectory = Join-Path $prefix "share\askme-webrtc-apm"
New-Item -ItemType Directory -Force $stampDirectory | Out-Null
Set-Content -NoNewline -Encoding ascii (Join-Path $stampDirectory "SOURCE_COMMIT.txt") $commit
Copy-Item -Force (Join-Path $source "COPYING") (Join-Path $stampDirectory "SOURCE_LICENSE.txt")
Write-Output "WEBRTC_APM_ROOT=$prefix"
