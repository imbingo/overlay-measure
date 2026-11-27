[CmdletBinding()]
param(
    [switch]$ForceInstall,
    [switch]$ValidateOnly,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ApplicationArguments
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$requirementsPath = Join-Path $projectRoot "requirements.txt"
$mainPath = Join-Path $projectRoot "main.py"
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$dependencyStamp = Join-Path $venvPath ".overlay_requirements.sha256"

function Write-Step([string]$message) {
    Write-Host "[Overlay Measure] $message" -ForegroundColor Cyan
}

function Test-CompatiblePython([string]$command, [string[]]$prefixArguments) {
    try {
        $probe = & $command @prefixArguments -c "import struct, sys; print(f'{sys.version_info.major}.{sys.version_info.minor}:{struct.calcsize(\"P\")*8}')" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $probe) {
            return $null
        }

        $tokens = $probe.Trim().Split(":")
        $version = $tokens[0]
        $bits = if ($tokens.Length -gt 1) { [int]$tokens[1] } else { 0 }
        $parts = $version.Split(".")
        $major = [int]$parts[0]
        $minor = [int]$parts[1]
        if ($major -eq 3 -and $minor -ge 10 -and $minor -lt 14 -and $bits -eq 64) {
            return @{
                Command = $command
                PrefixArguments = $prefixArguments
                Version = $version
            }
        }
    }
    catch {
        return $null
    }
    return $null
}

function Find-CompatiblePython {
    $pyLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($version in @("3.13", "3.12", "3.11", "3.10")) {
            $candidate = Test-CompatiblePython $pyLauncher.Source @("-$version")
            if ($candidate) {
                return $candidate
            }
        }
    }

    foreach ($name in @("python.exe", "python3.exe")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            $candidate = Test-CompatiblePython $command.Source @()
            if ($candidate) {
                return $candidate
            }
        }
    }
    return $null
}

if (-not (Test-Path -LiteralPath $requirementsPath)) {
    throw "Missing dependency file: $requirementsPath"
}
if (-not (Test-Path -LiteralPath $mainPath)) {
    throw "Missing application entry point: $mainPath"
}

$basePython = Find-CompatiblePython
if (-not $basePython) {
    throw "Python 3.10-3.13 (64-bit) was not found. Install Python from python.org and enable the Python launcher, then run this file again."
}

Write-Step "Found Python $($basePython.Version)."
if ($ValidateOnly) {
    Write-Step "Bootstrap validation passed."
    exit 0
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Step "Creating project environment: $venvPath"
    & $basePython.Command @($basePython.PrefixArguments) -m venv $venvPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython)) {
        throw "Failed to create the project virtual environment."
    }
}

$requirementsHash = (Get-FileHash -LiteralPath $requirementsPath -Algorithm SHA256).Hash
$installedHash = ""
if (Test-Path -LiteralPath $dependencyStamp) {
    $installedHash = (Get-Content -LiteralPath $dependencyStamp -Raw).Trim()
}

$importsHealthy = $false
if (-not $ForceInstall -and $requirementsHash -eq $installedHash) {
    & $venvPython -c "import PySide6, cv2, numpy, pandas, openpyxl, PIL" 2>$null
    $importsHealthy = ($LASTEXITCODE -eq 0)
}

if ($ForceInstall -or $requirementsHash -ne $installedHash -or -not $importsHealthy) {
    Write-Step "Installing required libraries. The first run may take several minutes."
    & $venvPython -m pip install --disable-pip-version-check -r $requirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed. Check the network or Python package source and run again."
    }

    & $venvPython -c "import PySide6, cv2, numpy, pandas, openpyxl, PIL"
    if ($LASTEXITCODE -ne 0) {
        throw "Dependencies were installed, but the import verification failed."
    }
    Set-Content -LiteralPath $dependencyStamp -Value $requirementsHash -Encoding Ascii
    Write-Step "Dependencies are ready."
}
else {
    Write-Step "Existing project environment is ready."
}

Write-Step "Starting application."
Push-Location $projectRoot
try {
    & $venvPython $mainPath @ApplicationArguments
    $applicationExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($applicationExitCode -ne 0) {
    throw "Application exited with code $applicationExitCode."
}
