# Laster ned ASL Alphabet Dataset (Cabana 2025, Mendeley) til data/Train og data/Test
# https://data.mendeley.com/datasets/jdyksv2jhh/1  (CC BY 4.0, ca. 900 MB)
# Windows-utgaven av setup/data.sh. Krever Windows 10 1803 eller nyere (tar.exe).
$ErrorActionPreference = 'Stop'

$dataDir = Join-Path (Split-Path -Parent $PSScriptRoot) 'data'
$zip = Join-Path $dataDir 'asl-alphabet.zip'

if ((Test-Path -LiteralPath (Join-Path $dataDir 'Train') -PathType Container) -and
    (Test-Path -LiteralPath (Join-Path $dataDir 'Test') -PathType Container)) {
    Write-Output "$dataDir\Train og Test finnes allerede, hopper over"
    return
}

New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
# Fremdriftslinjen gjør Invoke-WebRequest svært treg i Windows PowerShell 5.1.
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -UseBasicParsing -OutFile $zip -Uri 'https://data.mendeley.com/public-files/datasets/jdyksv2jhh/files/d3aafafe-7672-40d1-b390-992e2a0c0a95/file_downloaded'
# tar er mye raskere enn Expand-Archive for så mange små filer.
tar -xf $zip -C $dataDir
if ($LASTEXITCODE -ne 0) { throw "Kunne ikke pakke ut $zip (tar avsluttet med $LASTEXITCODE)." }
Remove-Item -LiteralPath $zip

# macOS-søppel i zip-filen
Remove-Item -LiteralPath (Join-Path $dataDir '__MACOSX') -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -LiteralPath $dataDir -Recurse -Force -Filter '.DS_Store' | Remove-Item -Force

Write-Output "Ferdig: $dataDir\Train og $dataDir\Test"
