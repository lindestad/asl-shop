<# Launch the desktop app with explicit trained checkpoints and a separate runtime interpreter. #>
[CmdletBinding()]
param(
    [string]$Python = '',
    [string]$HandshapeModel = '',
    [string]$SequenceModel = '',
    [string]$LandmarkerModel = '',
    [string]$OpenFSSource = '',
    [string]$OpenFSModelDir = '',
    [switch]$Plan
)
$ErrorActionPreference = 'Stop'
$repoDirectory = Split-Path -Parent $PSScriptRoot
function Resolve-LaunchPath {
    param([string]$Value, [string]$Default)
    if (-not $Value) { $Value = $Default }
    if (-not [System.IO.Path]::IsPathRooted($Value)) {
        $Value = Join-Path $repoDirectory $Value
    }
    return [System.IO.Path]::GetFullPath($Value)
}
$runtimeEnvironment = [ordered]@{
    ASL_SHOP_PYTHON = Resolve-LaunchPath $Python '.venv/Scripts/python.exe'
    ASL_HANDSHAPE_MODEL = Resolve-LaunchPath $HandshapeModel 'data/handshape.pt'
    ASL_SEQUENCE_MODEL = Resolve-LaunchPath $SequenceModel 'data/sequence.pt'
    ASL_MEDIAPIPE_MODEL = Resolve-LaunchPath $LandmarkerModel 'data/hand_landmarker.task'
}
if ($OpenFSSource -or $OpenFSModelDir) {
    if (-not $OpenFSSource -or -not $OpenFSModelDir) {
        throw 'Optional OpenFS refinement requires both -OpenFSSource and -OpenFSModelDir.'
    }
    $runtimeEnvironment['ASL_OPENFS_SOURCE'] = Resolve-LaunchPath $OpenFSSource ''
    $runtimeEnvironment['ASL_OPENFS_MODEL_DIR'] = Resolve-LaunchPath $OpenFSModelDir ''
}
if ($Plan) {
    [ordered]@{ repository = $repoDirectory; environment = $runtimeEnvironment; command = @('npm.cmd', 'run', 'dev') } |
        ConvertTo-Json -Depth 4
    return
}
foreach ($entry in $runtimeEnvironment.GetEnumerator()) {
    $pathType = if ($entry.Key -in @('ASL_OPENFS_SOURCE', 'ASL_OPENFS_MODEL_DIR')) { 'Container' } else { 'Leaf' }
    if (-not (Test-Path -LiteralPath $entry.Value -PathType $pathType)) {
        throw "Missing $($entry.Key): $($entry.Value). Run setup.ps1 -Neural or supply an existing file path."
    }
}
$npmCommand = Get-Command npm.cmd -ErrorAction Stop
$priorEnvironment = @{}
foreach ($entry in $runtimeEnvironment.GetEnumerator()) {
    $priorEnvironment[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, 'Process')
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
}
Push-Location -LiteralPath $repoDirectory
try {
    Write-Output 'Starting local neural recognition. Camera images stay local.'
    & $npmCommand.Source run dev
    if ($LASTEXITCODE -ne 0) { throw "Desktop launch failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
    foreach ($entry in $runtimeEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $priorEnvironment[$entry.Key], 'Process')
    }
}
