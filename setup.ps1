<#
.SYNOPSIS
Set up ASL Shop on Windows, with optional neural dependencies and OpenFS data.
.EXAMPLE
.\setup.ps1 -Neural
.EXAMPLE
.\setup.ps1 -ExtractOpenFS -OpenFSDir 'D:\datasets\openfs'
#>
[CmdletBinding()]
param(
    [switch]$Neural,
    [switch]$OpenFS,
    [switch]$ExtractOpenFS,
    [string]$OpenFSDir = '',
    [switch]$Help,
    [switch]$Plan
)

$ErrorActionPreference = 'Stop'
$repoDirectory = $PSScriptRoot
if ($Help) {
    Write-Output 'Usage: .\setup.ps1 [-Neural] [-OpenFS] [-ExtractOpenFS] [-OpenFSDir PATH]'
    Write-Output 'Requires uv and Node.js 22.12+ with npm. OpenFS is optional and approximately 12.7 GB compressed.'
    Write-Output 'Existing ZIPs are reused, interrupted downloads resume, and extraction preserves ZIPs.'
    return
}
if (-not $OpenFSDir) {
    $OpenFSDir = Join-Path $repoDirectory 'data/external/openfs'
} elseif (-not [System.IO.Path]::IsPathRooted($OpenFSDir)) {
    $OpenFSDir = Join-Path $repoDirectory $OpenFSDir
}
$OpenFSDir = [System.IO.Path]::GetFullPath($OpenFSDir)
$installNeural = $Neural -or $OpenFS -or $ExtractOpenFS
$prepareOpenFS = $OpenFS -or $ExtractOpenFS
$syncArguments = @('sync', '--locked', '--python', '3.12')
if ($installNeural) { $syncArguments += @('--extra', 'neural') }
$dataArguments = @('run', '--locked', '--with', 'gdown', 'python',
    'scripts/download_openfs.py', '--directory', $OpenFSDir)
if ($ExtractOpenFS) { $dataArguments += '--extract' }

# A read-only plan supports inspection and CLI regression tests without installing
# dependencies, starting downloads, or creating directories.
if ($Plan) {
    [ordered]@{
        repository = $repoDirectory
        neural = [bool]$installNeural
        openfs = [bool]$prepareOpenFS
        extract = [bool]$ExtractOpenFS
        openfsDirectory = $OpenFSDir
        pythonSync = $syncArguments
        openfsCommand = $(if ($prepareOpenFS) { $dataArguments } else { @() })
    } | ConvertTo-Json -Depth 4
    return
}

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE."
    }
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uvExecutable = $uvCommand.Source
} else {
    $uvExecutable = Join-Path $env:USERPROFILE '.local/bin/uv.exe'
    if (-not (Test-Path -LiteralPath $uvExecutable -PathType Leaf)) {
        throw 'Install uv using https://docs.astral.sh/uv/getting-started/installation/ and rerun setup.ps1.'
    }
}
$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
$npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
$npxCommand = Get-Command npx.cmd -ErrorAction SilentlyContinue
if (-not $nodeCommand -or -not $npmCommand -or -not $npxCommand) {
    throw 'Install Node.js 22.12+ with npm using https://nodejs.org/en/download and rerun setup.ps1.'
}
Invoke-Checked $nodeCommand.Source @('-e', 'const [major, minor] = process.versions.node.match(/\d+/g).map(Number); process.exit(major > 22 || (major === 22 && minor >= 12) ? 0 : 1)')

Push-Location -LiteralPath $repoDirectory
try {
    Write-Output 'Setting up Python 3.12 and project dependencies...'
    Invoke-Checked $uvExecutable $syncArguments
    Write-Output 'Installing frontend and desktop dependencies...'
    Invoke-Checked $npmCommand.Source @('ci', '--include=dev')
    Invoke-Checked $npxCommand.Source @('--no-install', 'install-electron')
    Invoke-Checked $npxCommand.Source @('--no-install', 'playwright', 'install', 'chromium')

    $dataDirectory = Join-Path $repoDirectory 'data'
    New-Item -ItemType Directory -Path $dataDirectory -Force | Out-Null
    $landmarkerPath = Join-Path $dataDirectory 'hand_landmarker.task'
    if (-not (Test-Path -LiteralPath $landmarkerPath) -or (Get-Item -LiteralPath $landmarkerPath).Length -eq 0) {
        Write-Output 'Downloading MediaPipe Hand Landmarker...'
        $partialPath = "$landmarkerPath.part"
        Invoke-WebRequest -UseBasicParsing -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task' -OutFile $partialPath
        Invoke-Checked $uvExecutable @('run', '--no-sync', 'python', 'scripts/validate_landmarker.py', $partialPath)
    } else {
        Write-Output 'MediaPipe model is already installed.'
    }
    if ($prepareOpenFS) {
        Invoke-Checked $uvExecutable $dataArguments
    }
    Write-Output 'Setup complete. Start the app with: npm run dev'
} finally {
    Pop-Location
}
