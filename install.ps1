#
# Transcribe Tool - Quick Installation Script (Windows)
#
# This script automates the installation process for Transcribe Tool on native
# Windows 10/11. It creates a virtual environment, installs dependencies, and
# verifies the installation. It mirrors install.sh step for step.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\install.ps1              # Core features
#   powershell -ExecutionPolicy Bypass -File .\install.ps1 --all        # All features (recommended)
#   powershell -ExecutionPolicy Bypass -File .\install.ps1 --full       # Alias for --all
#   powershell -ExecutionPolicy Bypass -File .\install.ps1 --dev        # Development tools
#
# Options:
#   --yes / -y            Answer yes to every prompt (unattended)
#   --cpu                 Skip the CUDA build of PyTorch even if an NVIDIA GPU is present
#   --cuda-index <url>    Override the PyTorch CUDA wheel index
#   --python <path>       Use a specific python.exe instead of the py launcher
#                         (also settable via the TRANSCRIBE_TOOL_PYTHON variable)
#
# Author: Antoine Lemor

$ErrorActionPreference = 'Stop'

# Forces UTF-8 in every child Python process: some sdists (mosestokenizer, uctools)
# read setup.py with the ANSI codepage and die on JP/CN/KR Windows installs.
$env:PYTHONUTF8 = '1'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Rule {
    Write-Host "=========================================================="
}

function Write-Ok {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Bad {
    param([string]$Message)
    Write-Host "[X] $Message" -ForegroundColor Red
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "[-] $Message" -ForegroundColor Cyan
}

# $ErrorActionPreference does not trap native executable failures, so every
# pip/python/winget/ffmpeg call must be followed by an explicit check.
function Assert-LastExitCode {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Bad "$What failed (exit code $LASTEXITCODE)"
        exit 1
    }
}

function Confirm-Step {
    param(
        [string]$Message,
        [bool]$DefaultYes = $false
    )
    if ($script:assumeYes) { return $true }
    if ([Console]::IsInputRedirected) { return $DefaultYes }
    $answer = Read-Host $Message
    if ([string]::IsNullOrWhiteSpace($answer)) { return $DefaultYes }
    return ($answer -match '^[Yy]')
}

function Test-Elevated {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# $env:PATH is a snapshot taken when this process started; winget/choco edit the
# registry, so PATH must be rebuilt before re-probing for a freshly installed tool.
function Update-PathFromRegistry {
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:PATH = $machinePath + ';' + $userPath
}

# Windows PowerShell 5.1's -Encoding UTF8 always emits a BOM; every file this
# script writes must be BOM-free (site.py reads .pth before any codec fallback).
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# 1638 = a newer version is already installed, 3010 = success with reboot pending,
# -1978335189 = winget "no applicable upgrade found".
function Test-InstallerExitCode {
    param([int]$Code)
    return (@(0, 1638, 3010, -1978335189) -contains $Code)
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "=============================================================="
Write-Host "                                                              "
Write-Host "                   T R A N S C R I B E   T O O L              "
Write-Host "                                                              "
Write-Host "           Audio Extraction & Transcription Pipeline          "
Write-Host "   YouTube, TikTok, Local Files -> Whisper -> Diarization     "
Write-Host "                                                              "
Write-Host "=============================================================="
Write-Host ""

# ---------------------------------------------------------------------------
# Argument parsing
#
# A param([switch]$All) block cannot bind '--all' ("A positional parameter
# cannot be found that accepts argument '--all'"), so $args is parsed by hand.
# ---------------------------------------------------------------------------

$installType = 'core'
$assumeYes = $false
$forceCpu = $false
$CudaIndex = 'https://download.pytorch.org/whl/cu128'
$expectCudaIndex = $false
$PythonExe = $null
$expectPython = $false

foreach ($arg in $args) {
    if ($expectCudaIndex) {
        $CudaIndex = $arg
        $expectCudaIndex = $false
        continue
    }
    if ($expectPython) {
        $PythonExe = $arg
        $expectPython = $false
        continue
    }
    switch -Regex ($arg) {
        '^(--all|--full|-All|-Full)$' { $installType = 'full' }
        '^(--dev|-Dev)$'              { $installType = 'dev' }
        '^(--yes|-Yes|-y)$'           { $assumeYes = $true }
        '^(--cpu|-Cpu)$'              { $forceCpu = $true }
        '^(--cuda-index|-CudaIndex)$' { $expectCudaIndex = $true }
        '^(--python|-Python)$'        { $expectPython = $true }
        '^(-h|--help|-Help|/\?)$' {
            Write-Host "Usage: powershell -ExecutionPolicy Bypass -File .\install.ps1 [options]"
            Write-Host ""
            Write-Host "  --all, --full        Install all features (diarization, voice separation)"
            Write-Host "  --dev                Install development tools"
            Write-Host "  --yes, -y            Answer yes to every prompt (unattended)"
            Write-Host "  --cpu                Skip the CUDA PyTorch build even with an NVIDIA GPU"
            Write-Host "  --cuda-index <url>   PyTorch CUDA wheel index (default: $CudaIndex)"
            Write-Host "  --python <path>      Use this python.exe instead of the py launcher"
            Write-Host "                       (also settable as TRANSCRIBE_TOOL_PYTHON)"
            Write-Host "  -h, --help           Show this message"
            Write-Host ""
            exit 0
        }
        default {
            Write-Bad "Unknown option: $arg"
            Write-Host "  Run '.\install.ps1 --help' for the list of supported options."
            exit 1
        }
    }
}

if ($expectCudaIndex) {
    Write-Bad "--cuda-index requires a URL argument"
    exit 1
}

if ($expectPython) {
    Write-Bad "--python requires a path to python.exe"
    exit 1
}

if ($installType -eq 'full') {
    Write-Host "Installing with ALL features (diarization, voice separation)..." -ForegroundColor Blue
} elseif ($installType -eq 'dev') {
    Write-Host "Installing with DEVELOPMENT tools..." -ForegroundColor Blue
} else {
    Write-Host "Installing with CORE features..." -ForegroundColor Blue
    Write-Warn "Tip: Use '.\install.ps1 --all' for all features (diarization, voice separation)"
}
Write-Host ""

Set-Location -LiteralPath $PSScriptRoot

# ---------------------------------------------------------------------------
# Step 1: Python
# ---------------------------------------------------------------------------

Write-Rule
Write-Host "Step 1: Checking Python version..."
Write-Rule

# 'python3' on Windows is the Microsoft Store App Execution Alias stub: it opens
# the Store and exits 9009. Prefer the py launcher, then a real python.exe.
$pyExe = $null
$pyArgs = @()

# An explicit interpreter wins over discovery: the py launcher resolves -3 through
# the registry, which is the wrong build when several Pythons are installed.
if (-not $PythonExe) { $PythonExe = $env:TRANSCRIBE_TOOL_PYTHON }
if ($PythonExe) {
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        Write-Bad "The interpreter passed with --python does not exist: $PythonExe"
        exit 1
    }
    $pyExe = (Resolve-Path -LiteralPath $PythonExe).Path
    $pyArgs = @()
    Write-Info "Using the interpreter given by --python / TRANSCRIBE_TOOL_PYTHON"
}

$pyLauncher = $null
if (-not $pyExe) { $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue }
if ($pyLauncher) {
    $pyExe = $pyLauncher.Source
    $pyArgs = @('-3')
} elseif (-not $pyExe) {
    $candidates = @(Get-Command python.exe -All -ErrorAction SilentlyContinue)
    foreach ($candidate in $candidates) {
        if ($candidate.Source -notlike '*\WindowsApps\*') {
            $pyExe = $candidate.Source
            $pyArgs = @()
            break
        }
    }
}

if (-not $pyExe) {
    Write-Bad "Python 3 not found"
    Write-Host "  Install Python 3.10 or higher (3.12 is the best-tested version) from:"
    Write-Host "    https://www.python.org/downloads/windows/"
    Write-Host "  In the installer, tick 'Add python.exe to PATH' before clicking Install."
    exit 1
}

$pyVersion = [string](& $pyExe @pyArgs -c "import sys;print('%d.%d' % sys.version_info[:2])")
Assert-LastExitCode "Python version check"
if ([string]::IsNullOrWhiteSpace($pyVersion)) {
    Write-Bad "Could not determine the Python version"
    Write-Host "  Install Python 3.10 or higher from https://www.python.org/downloads/windows/"
    exit 1
}

$pyVersion = $pyVersion.Trim()
$versionParts = $pyVersion.Split('.')
$pyMajor = [int]$versionParts[0]
$pyMinor = [int]$versionParts[1]

if (($pyMajor -lt 3) -or (($pyMajor -eq 3) -and ($pyMinor -lt 10))) {
    Write-Bad "Python $pyVersion is too old"
    Write-Host "  Required: Python 3.10 or higher (3.12 is the best-tested version)"
    Write-Host "  Found: Python $pyVersion"
    Write-Host "  Download: https://www.python.org/downloads/windows/"
    Write-Host "  In the installer, tick 'Add python.exe to PATH' before clicking Install."
    exit 1
}

Write-Ok "Python $pyVersion found"
Write-Host ""

# ---------------------------------------------------------------------------
# Step 1b: Architecture
# ---------------------------------------------------------------------------

Write-Rule
Write-Host "Step 1b: Checking interpreter architecture..."
Write-Rule

# Read the architecture from the interpreter, not from $env:PROCESSOR_ARCHITECTURE:
# an amd64 Python running under emulation on an ARM64 machine is fully supported.
$pyArch = [string](& $pyExe @pyArgs -c "import platform;print(platform.machine())")
Assert-LastExitCode "Python architecture check"
$pyArch = $pyArch.Trim()

if ($pyArch -match '^(ARM64|aarch64)$') {
    Write-Bad "Python is the ARM64 (win_arm64) build"
    Write-Host "  torch, tiktoken (via openai-whisper) and pyreadr publish no win_arm64 wheels,"
    Write-Host "  so this build cannot install Transcribe Tool."
    Write-Host ""
    Write-Host "  Install the 64-bit AMD64 build instead - it runs under emulation on"
    Write-Host "  Windows on Arm (slower, but fully functional):"
    Write-Host "    https://www.python.org/downloads/windows/  ->  'Windows installer (64-bit)'"
    exit 1
}

Write-Ok "Architecture $pyArch supported"
Write-Host ""

# ---------------------------------------------------------------------------
# Step 1c: Microsoft Visual C++ 2015-2022 Redistributable
# ---------------------------------------------------------------------------

Write-Rule
Write-Host "Step 1c: Checking Visual C++ runtime (required by PyTorch)..."
Write-Rule

# Probing first avoids a UAC prompt on every run of an already-provisioned machine.
$system32 = Join-Path $env:WINDIR 'System32'
$missingRuntimes = @()
foreach ($dll in @('vcruntime140.dll', 'msvcp140.dll', 'vcruntime140_1.dll')) {
    if (-not (Test-Path -LiteralPath (Join-Path $system32 $dll))) {
        $missingRuntimes += $dll
    }
}

if ($missingRuntimes.Count -eq 0) {
    Write-Ok "Visual C++ 2015-2022 runtime present"
} else {
    Write-Warn "Visual C++ runtime incomplete (missing: $($missingRuntimes -join ', '))"
    $vcInstalled = $false

    $wingetCmd = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-Host "  Installing via winget..."
        & winget install --id 'Microsoft.VCRedist.2015+.x64' -e --accept-source-agreements --accept-package-agreements
        if (Test-InstallerExitCode $LASTEXITCODE) { $vcInstalled = $true }
    }

    if (-not $vcInstalled) {
        Write-Host "  Downloading the redistributable from Microsoft..."
        $progressBackup = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            $vcInstaller = Join-Path $env:TEMP 'vc_redist.x64.exe'
            Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $vcInstaller -UseBasicParsing
            $proc = Start-Process -FilePath $vcInstaller -ArgumentList '/install', '/quiet', '/norestart' -Wait -PassThru
            if (Test-InstallerExitCode $proc.ExitCode) { $vcInstalled = $true }
        } catch {
            $vcInstalled = $false
        } finally {
            $ProgressPreference = $progressBackup
        }
    }

    if ($vcInstalled) {
        Write-Ok "Visual C++ 2015-2022 runtime installed"
    } else {
        Write-Warn "Could not install the Visual C++ runtime (this step needs Administrator)"
        Write-Host "  Install it manually from https://aka.ms/vs/17/release/vc_redist.x64.exe"
        Write-Host "  Without it 'import torch' fails with:"
        Write-Host '    OSError: [WinError 126] ... Error loading "...\torch\lib\fbgemm.dll"'
        Write-Host "  Continuing anyway."
    }
}
Write-Host ""

# ---------------------------------------------------------------------------
# Step 2: FFmpeg
# ---------------------------------------------------------------------------

Write-Rule
Write-Host "Step 2: Checking FFmpeg..."
Write-Rule

$ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue

if (-not $ffmpegCmd) {
    Write-Warn "FFmpeg not found"
    Write-Host "  FFmpeg is required for audio extraction."
    Write-Host ""

    $wingetCmd = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        if (Confirm-Step "  Install FFmpeg now with winget? [Y/n]" $true) {
            & winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
            if (Test-InstallerExitCode $LASTEXITCODE) {
                Update-PathFromRegistry
                $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
            } else {
                Write-Warn "winget could not install FFmpeg (exit code $LASTEXITCODE)"
            }
        }
    }

    if (-not $ffmpegCmd) {
        $chocoCmd = Get-Command choco.exe -ErrorAction SilentlyContinue
        if ($chocoCmd) {
            if (Test-Elevated) {
                if (Confirm-Step "  Install FFmpeg now with Chocolatey? [Y/n]" $true) {
                    & choco install ffmpeg -y
                    if ($LASTEXITCODE -eq 0) {
                        Update-PathFromRegistry
                        $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
                    } else {
                        Write-Warn "Chocolatey could not install FFmpeg (exit code $LASTEXITCODE)"
                    }
                }
            } else {
                Write-Warn "Chocolatey is installed but needs an Administrator shell"
                Write-Host "  Right-click PowerShell -> 'Run as administrator', then: choco install ffmpeg -y"
            }
        }
    }

    if (-not $ffmpegCmd) {
        Write-Host ""
        Write-Host "  Install FFmpeg manually:"
        Write-Host "    1. Download a build from https://www.gyan.dev/ffmpeg/builds/"
        Write-Host "    2. Extract it, then add its 'bin' folder to your user PATH"
        Write-Host "    3. Close and reopen this terminal so PATH refreshes"
        Write-Host ""
        Write-Warn "FFmpeg still not on PATH - if you just installed it, restart your shell"
        Write-Host ""
        if (-not (Confirm-Step "  Continue anyway? [y/N]" $false)) {
            exit 1
        }
    }
}

if ($ffmpegCmd) {
    $ffmpegBanner = & $ffmpegCmd.Source -version | Select-Object -First 1
    $ffmpegVersion = 'unknown'
    if ($ffmpegBanner) {
        $bannerFields = ([string]$ffmpegBanner).Split(' ')
        if ($bannerFields.Count -ge 3) { $ffmpegVersion = $bannerFields[2] }
    }
    Write-Ok "FFmpeg $ffmpegVersion found"
}
Write-Host ""

# ---------------------------------------------------------------------------
# Step 3: Virtual environment
# ---------------------------------------------------------------------------

Write-Rule
Write-Host "Step 3: Creating virtual environment..."
Write-Rule

$venvDir = Join-Path $PSScriptRoot '.venv'

if (Test-Path -LiteralPath $venvDir) {
    Write-Warn "Virtual environment already exists at .venv/"
    # Answering no here KEEPS the existing environment and continues.
    if (Confirm-Step "  Remove and recreate? [y/N]" $false) {
        Remove-Item -LiteralPath $venvDir -Recurse -Force
        Write-Ok "Removed existing virtual environment"
    } else {
        Write-Info "Using existing virtual environment"
    }
}

if (-not (Test-Path -LiteralPath $venvDir)) {
    & $pyExe @pyArgs -m venv $venvDir
    Assert-LastExitCode "Virtual environment creation"
    Write-Ok "Virtual environment created at .venv/"
}

# Everything runs through this interpreter directly: never activating also
# sidesteps the ExecutionPolicy requirement of Activate.ps1.
$venvPy = Join-Path $venvDir 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPy)) {
    Write-Bad "Virtual environment is incomplete (missing $venvPy)"
    Write-Host "  Delete the .venv folder and run this script again."
    exit 1
}
Write-Host ""

# ---------------------------------------------------------------------------
# Step 4: VS Code
# ---------------------------------------------------------------------------

Write-Rule
Write-Host "Step 4: Configuring VS Code..."
Write-Rule

$vscodeDir = Join-Path $PSScriptRoot '.vscode'
New-Item -ItemType Directory -Force -Path $vscodeDir | Out-Null

$vscodeSettings = @'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "python.terminal.activateEnvInCurrentTerminal": true,
    "python.analysis.extraPaths": [
        "${workspaceFolder}"
    ],
    "python.autoComplete.extraPaths": [
        "${workspaceFolder}"
    ],
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        }
    }
}
'@

$vscodeExtensions = @'
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "eamodio.gitlens"
    ]
}
'@

[System.IO.File]::WriteAllText((Join-Path $vscodeDir 'settings.json'), $vscodeSettings + "`n", $utf8NoBom)
[System.IO.File]::WriteAllText((Join-Path $vscodeDir 'extensions.json'), $vscodeExtensions + "`n", $utf8NoBom)

Write-Ok "VS Code configured to use .venv as default Python interpreter"
Write-Host ""

# ---------------------------------------------------------------------------
# Step 5: pip
# ---------------------------------------------------------------------------

Write-Rule
Write-Host "Step 5: Upgrading pip..."
Write-Rule

& $venvPy -m pip install --upgrade pip setuptools wheel | Out-Null
Assert-LastExitCode "pip upgrade"
Write-Ok "pip upgraded to latest version"
Write-Host ""

# ---------------------------------------------------------------------------
# Step 6: PyTorch
# ---------------------------------------------------------------------------

Write-Rule
Write-Host "Step 6: Selecting the PyTorch build..."
Write-Rule

# Get-Command, never a bare nvidia-smi call: the latter throws terminatingly
# under $ErrorActionPreference = 'Stop' on machines without an NVIDIA GPU.
$hasNvidia = ($null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue))
if (-not $hasNvidia) {
    try {
        $nvidiaAdapters = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'NVIDIA' })
        $hasNvidia = ($nvidiaAdapters.Count -gt 0)
    } catch {
        $hasNvidia = $false
    }
}

$installedCuda = $false
if ($hasNvidia -and (-not $forceCpu)) {
    Write-Info "NVIDIA GPU detected - installing the CUDA build of PyTorch"
    Write-Host "  Index: $CudaIndex"
    # torchaudio must come from the same index: diarization.py imports it directly,
    # and a PyPI CPU torchaudio next to a CUDA torch is a WinError 126 ABI mismatch.
    & $venvPy -m pip install torch torchaudio --index-url $CudaIndex
    Assert-LastExitCode "CUDA PyTorch install"
    $installedCuda = $true
    Write-Ok "CUDA PyTorch installed"
} elseif ($hasNvidia) {
    Write-Info "NVIDIA GPU detected but --cpu was requested - using the CPU build"
} else {
    Write-Info "No NVIDIA GPU detected - using the CPU build of PyTorch"
}
Write-Host ""

# ---------------------------------------------------------------------------
# Step 7: Transcribe Tool
# ---------------------------------------------------------------------------

Write-Rule
Write-Host "Step 7: Installing Transcribe Tool..."
Write-Rule

if ($installType -eq 'full') {
    Write-Host "  Installing with full features (this may take a few minutes)..."
    & $venvPy -m pip install -e ".[full]"
    Assert-LastExitCode "Transcribe Tool install"
} elseif ($installType -eq 'dev') {
    Write-Host "  Installing in editable mode for development..."
    & $venvPy -m pip install -e ".[dev]"
    Assert-LastExitCode "Transcribe Tool install"
    # Python >=3.13 skips .pth files starting with __ (treats them as hidden),
    # which breaks setuptools editable installs. Fix by renaming the files.
    if ($pyMinor -ge 13) {
        $sitePackages = [string](& $venvPy -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
        Assert-LastExitCode "site-packages lookup"
        $sitePackages = $sitePackages.Trim()
        $pthFile = Join-Path $sitePackages '__editable__.transcribe_tool-1.0.0.pth'
        $finderFile = Join-Path $sitePackages '__editable___transcribe_tool_1_0_0_finder.py'
        if (Test-Path -LiteralPath $pthFile) {
            $newPth = Join-Path $sitePackages 'editable-transcribe_tool-1.0.0.pth'
            $newFinder = Join-Path $sitePackages 'editable_transcribe_tool_1_0_0_finder.py'
            Move-Item -LiteralPath $pthFile -Destination $newPth -Force
            $pthLines = @(Get-Content -LiteralPath $newPth) -replace '__editable___transcribe_tool_1_0_0_finder', 'editable_transcribe_tool_1_0_0_finder'
            [System.IO.File]::WriteAllLines($newPth, [string[]]$pthLines, $utf8NoBom)
            if (Test-Path -LiteralPath $finderFile) {
                Move-Item -LiteralPath $finderFile -Destination $newFinder -Force
                $finderLines = @(Get-Content -LiteralPath $newFinder) -replace '__editable__\.transcribe_tool-1\.0\.0\.finder', 'editable.transcribe_tool-1.0.0.finder'
                [System.IO.File]::WriteAllLines($newFinder, [string[]]$finderLines, $utf8NoBom)
            }
            Write-Ok "Applied Python 3.13+ editable install fix"
        }
    }
} else {
    & $venvPy -m pip install -e .
    Assert-LastExitCode "Transcribe Tool install"
}

Write-Ok "Transcribe Tool installed successfully"

# A dependency bound can silently pull the CPU wheel from PyPI over the CUDA one,
# and the install still reports success - so verify and repair.
if ($installedCuda) {
    & $venvPy -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "PyTorch was replaced by a CPU-only build - reinstalling the CUDA wheels"
        & $venvPy -m pip install torch torchaudio --index-url $CudaIndex --force-reinstall --no-deps
        if ($LASTEXITCODE -eq 0) {
            & $venvPy -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)"
        }
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "CUDA PyTorch restored"
        } else {
            Write-Warn "PyTorch is still CPU-only - transcription will run on the CPU"
            Write-Host "  Retry manually with:"
            Write-Host "    .venv\Scripts\python.exe -m pip install torch torchaudio --index-url $CudaIndex --force-reinstall"
        }
    } else {
        Write-Ok "CUDA PyTorch verified"
    }
}
Write-Host ""

# ---------------------------------------------------------------------------
# Step 8: Language models (optional)
# ---------------------------------------------------------------------------

Write-Rule
Write-Host "Step 8: Installing language models (optional)..."
Write-Rule

$optionalModels = @(
    @{ Args = @('-m', 'spacy', 'download', 'en_core_web_sm'); Ok = 'English spaCy model installed'; Skip = 'English model not installed (optional)' },
    @{ Args = @('-m', 'spacy', 'download', 'fr_core_news_sm'); Ok = 'French spaCy model installed'; Skip = 'French model not installed (optional)' },
    @{ Args = @('-c', "from wtpsplit import SaT; SaT('sat-12l'); print('OK')"); Ok = 'wtpsplit sat-12l model downloaded'; Skip = 'wtpsplit model not downloaded (optional)' }
)

foreach ($model in $optionalModels) {
    $modelArgs = $model.Args
    try {
        & $venvPy @modelArgs | Out-Null
    } catch {
        $global:LASTEXITCODE = 1
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  " -NoNewline
        Write-Ok $model.Ok
    } else {
        Write-Host "  " -NoNewline
        Write-Warn $model.Skip
    }
}
Write-Host ""

# ---------------------------------------------------------------------------
# Step 9: Verification
# ---------------------------------------------------------------------------

Write-Rule
Write-Host "Step 9: Verifying installation..."
Write-Rule
Write-Host ""

$verifyScript = Join-Path $PSScriptRoot 'verify_installation.py'
if (Test-Path -LiteralPath $verifyScript) {
    try {
        & $venvPy $verifyScript
    } catch {
        Write-Warn "Verification script could not be run"
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Verification reported issues - see the report above"
    }
} else {
    Write-Warn "verify_installation.py not found - skipping verification"
}

# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "=========================================================="
Write-Host "                INSTALLATION COMPLETE!                    "
Write-Host "=========================================================="
Write-Host ""
Write-Host "Next steps:"
Write-Host ""
Write-Host "  1. Activate the virtual environment:"
Write-Host "     PowerShell: .\.venv\Scripts\Activate.ps1" -ForegroundColor Blue
Write-Host "     cmd.exe:    .venv\Scripts\activate.bat" -ForegroundColor Blue
Write-Host "     (if PowerShell refuses, run once:"
Write-Host "      Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)"
Write-Host ""
Write-Host "  2. Launch Transcribe Tool:"
Write-Host "     transcribe-tool" -ForegroundColor Blue
Write-Host "     # or use the short alias:" -ForegroundColor Cyan
Write-Host "     tt" -ForegroundColor Blue
Write-Host ""
Write-Host "  3. Quick transcription:"
Write-Host "     transcribe-tool --transcribe audio.mp3" -ForegroundColor Blue
Write-Host ""
Write-Host "  4. Download from YouTube:"
Write-Host '     transcribe-tool --youtube "URL"' -ForegroundColor Blue
Write-Host ""
Write-Host "  5. VS Code users:"
Write-Ok "     VS Code is already configured to use .venv"
Write-Host "       Just open/reload the workspace in VS Code!"
Write-Host ""
Write-Host "Optional setup for speaker diarization:"
Write-Host ""
Write-Host "  Set your HuggingFace token (required for diarization):"
Write-Host '    $env:HF_TOKEN = "your_huggingface_token"      # this session only' -ForegroundColor Yellow
Write-Host '    setx HF_TOKEN "your_huggingface_token"        # persistent, open a new terminal' -ForegroundColor Yellow
Write-Host ""
Write-Host "  Accept model terms at:"
Write-Host "    https://huggingface.co/pyannote/speaker-diarization-3.1" -ForegroundColor Cyan
Write-Host ""
Write-Host "For help and support:"
Write-Host "  README: README.md"
Write-Host "  GitHub: https://github.com/antoine-lemor/Transcribe-tool"
Write-Host ""
Write-Host "Happy transcribing!"
Write-Host ""
