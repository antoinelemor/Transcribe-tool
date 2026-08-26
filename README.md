# Transcribe Tool

Audio extraction and transcription pipeline for YouTube, TikTok, and local files.

## Features

- **Multi-source extraction**: YouTube videos/channels/playlists, TikTok videos/profiles, local audio/video files
- **Whisper transcription**: OpenAI Whisper (large-v3) with word-level timestamps
- **Speaker diarization**: Multi-speaker identification with Pyannote
- **SOTA sentence segmentation**: wtpsplit transformer model for accurate sentence boundaries
- **Voice separation**: Demucs vocal isolation for noisy audio
- **Language detection**: Automatic detection for 45+ languages
- **Multiple output formats**: TXT, CSV, JSON, SRT, WebVTT
- **R interoperability**: CSV and `.rds` export for downstream analysis

## Platform Support

| Platform | Shell | Installer |
|----------|-------|-----------|
| macOS (Intel / Apple Silicon) | bash / zsh | `./install.sh --all` |
| Linux | bash | `./install.sh --all` |
| Windows 10/11 (native, no WSL) | PowerShell 5.1 / cmd.exe | `powershell -ExecutionPolicy Bypass -File .\install.ps1 --all` |

Windows users can also double-click `install.bat`, which runs the same installer. Double-clicking
passes no arguments, so it installs the core features only; run `install.bat --all` from a terminal
for the full set.

## Installation

### Quick Start (macOS / Linux)

```bash
git clone https://github.com/antoinelemor/Transcribe-tool.git
cd Transcribe-tool
./install.sh --all
```

### Quick Start (Windows)

```powershell
git clone https://github.com/antoinelemor/Transcribe-tool.git
cd Transcribe-tool
powershell -ExecutionPolicy Bypass -File .\install.ps1 --all
```

Use that exact form rather than a bare `.\install.ps1`. If the repository was downloaded as a
ZIP instead of cloned, every extracted file carries the Mark-of-the-Web and is blocked even under
the default `RemoteSigned` policy. The alternative is to clear the mark once:

```powershell
Unblock-File .\install.ps1
.\install.ps1 --all
```

The least technical path is to double-click **`install.bat`** in File Explorer: it launches the
installer and keeps the console window open so errors stay readable. A double-click passes no
arguments, which installs the core features only — for diarization and voice separation, run
`install.bat --all` from cmd.exe or PowerShell instead.

There is no need to change the machine-wide execution policy — `-ExecutionPolicy Bypass` applies to
that single `powershell.exe` process only.

Installer options: `--all` / `--full` (all features), `--dev` (development tools), `--yes`
(non-interactive). `install.ps1` additionally accepts `--cpu` (skip the CUDA torch build) and
`--cuda-index <url>`, which takes a full wheel-index URL and defaults to
`https://download.pytorch.org/whl/cu128`.

### Manual Installation

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -e ".[full]"

python verify_installation.py
```

#### Windows — PowerShell

```powershell
py -3 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

$env:PYTHONUTF8 = "1"
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[full]"

python verify_installation.py
```

#### Windows — cmd.exe

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat

set PYTHONUTF8=1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[full]"

python verify_installation.py
```

Notes for Windows:

- Use `py -3`, not `python3`. The `python3` that is on `PATH` by default is the Microsoft Store App
  Execution Alias stub: it opens the Store page and exits with code 9009 without ever running Python.
- `PYTHONUTF8=1` before `pip install` matters on systems whose ANSI code page is not Latin-1
  (Japanese, Chinese, Korean, Cyrillic locales). Without it, source builds of a few dependencies read
  their `setup.py` with the legacy code page and fail with `UnicodeDecodeError`.
- Do not use `chcp 65001` for this — it changes the console code page, not the process code page.

### Requirements

- Python 3.10+ (3.12 is the best-tested version)
- FFmpeg on `PATH`

#### FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

```powershell
# Windows
winget install --id Gyan.FFmpeg -e
# or, with Chocolatey in an Administrator shell
choco install ffmpeg
```

After installing FFmpeg on Windows, **close and reopen the terminal** so that `PATH` is refreshed —
the current shell holds a snapshot taken at startup and will keep reporting `ffmpeg` as missing.

Manual fallback: download a build from https://www.gyan.dev/ffmpeg/builds/, extract it, and add the
`bin\` folder to your user `PATH`.

#### Windows: Microsoft Visual C++ Redistributable

PyTorch links against the Visual C++ 2015-2022 runtime. Install
https://aka.ms/vs/17/release/vc_redist.x64.exe if `import torch` fails with a DLL load error
(`OSError: [WinError 126] ... Error loading "...\torch\lib\fbgemm.dll"`). `install.ps1` probes for
it and installs it when missing.

#### Windows on Arm

Install the **amd64** Python build from python.org, not the ARM64 one. `torch`, `tiktoken` (pulled in
by openai-whisper) and `pyreadr` publish no `win_arm64` wheels; the amd64 interpreter runs under
emulation and works, just slower.

#### Windows: NVIDIA GPU

The `torch` wheel published on PyPI for Windows is **CPU-only** — installing it gives a working but
GPU-less setup. `install.ps1` detects an NVIDIA adapter and pulls torch from the CUDA index
automatically. Manual installs need it done explicitly, **before** the package install:

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[full]"
```

Install `torchaudio` from the same index as `torch`; a CPU `torchaudio` next to a CUDA `torch` is an
ABI mismatch that surfaces as a DLL load error at runtime. To stay on CPU, simply skip this step —
or pass `--cpu` to `install.ps1` if you are using the installer.

## Usage

### Interactive Mode

```bash
transcribe-tool
# or
tt
```

### Command Line

```bash
# YouTube extraction
transcribe-tool --youtube "https://www.youtube.com/watch?v=VIDEO_ID"

# Transcribe with diarization
transcribe-tool --transcribe audio.wav --diarize -l en -f csv

# Batch transcription
transcribe-tool --scan /path/to/audio -f csv
```

The same commands work verbatim in PowerShell and cmd.exe once the virtual environment is activated.

### Options

| Option | Description |
|--------|-------------|
| `-y, --youtube URL` | Download from YouTube |
| `-t, --tiktok URL` | Download from TikTok |
| `--transcribe FILE` | Transcribe audio file |
| `--scan DIR` | Batch process directory |
| `-l, --language CODE` | Language code (auto-detect if omitted) |
| `-m, --model MODEL` | Whisper model (tiny/base/small/medium/large-v3) |
| `-f, --format FORMAT` | Output format (txt/csv/json/srt/vtt) |
| `--diarize` | Enable speaker diarization |
| `--hf-token TOKEN` | HuggingFace token (overrides `HF_TOKEN` and the saved token) |
| `-o, --output DIR` | Output directory |

## Output Formats

### CSV

```csv
"segment_id","speaker","text","video_id","title","date","channel"
"1","SPEAKER_00","First sentence.","abc123","Title","20260103","Channel"
"2","SPEAKER_01","Response from second speaker.","abc123","Title","20260103","Channel"
```

CSV files are written as plain UTF-8 **without a BOM**, so R reads them directly:

```r
df <- read.csv("transcript.csv")
```

On Windows, do not double-click a CSV to open it in Excel: Excel then decodes it with the local ANSI
code page and mangles accented characters. Import it instead via **Data > From Text/CSV** and pick
**UTF-8** as the file origin.

### TXT

```
# Transcription

Date: 20260103
Title: Video Title
Language: English

---

[SPEAKER_00]
Transcribed text from first speaker.

[SPEAKER_01]
Text from second speaker.
```

## Speaker Diarization

Requires a HuggingFace token with access to the Pyannote models.

```bash
# macOS / Linux
export HF_TOKEN="your_token"
```

```powershell
# Windows PowerShell - current session
$env:HF_TOKEN = "your_token"

# Windows PowerShell - persistent (open a new terminal afterwards)
setx HF_TOKEN "your_token"
```

```bat
:: Windows cmd.exe - current session
set HF_TOKEN=your_token
```

Alternatively, save the token once from the interactive **Settings** menu. It is stored in the OS
credential store (Keychain on macOS, Credential Manager on Windows, Secret Service on Linux) and
reused on every run, so no environment variable is needed.

Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1

## Troubleshooting (Windows)

| Symptom | Fix |
|---------|-----|
| `install.ps1 cannot be loaded because running scripts is disabled` / `...is not digitally signed` | Run `powershell -ExecutionPolicy Bypass -File .\install.ps1 --all`, or `Unblock-File .\install.ps1` first (a ZIP download carries the Mark-of-the-Web). |
| `python3 : The term 'python3' is not recognized` — or a Microsoft Store page opens | Use `py -3`. `python3` on `PATH` is the Store App Execution Alias stub. |
| FFmpeg installed but still reported as missing | Close the terminal and open a new one; `PATH` is snapshotted at process start. If it still fails, check that the `bin\` folder is in your user `PATH`. |
| `error: Microsoft Visual C++ 14.0 or greater is required` | A dependency is being built from source. Use Python 3.10-3.12, where prebuilt wheels exist for every dependency, or install the Visual Studio Build Tools (C++ build tools workload). |
| `OSError: [WinError 126]` on `import torch` | Install the Visual C++ 2015-2022 Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe |
| `UnicodeEncodeError` when redirecting output to a file | Set `$env:PYTHONUTF8 = "1"` (PowerShell) or `set PYTHONUTF8=1` (cmd) before running. |
| `FileNotFoundError` on a long output filename, or paths silently failing | `MAX_PATH` (260 characters). Clone to a short path such as `C:\dev\Transcribe-tool` rather than a long OneDrive-redirected `Documents` folder, or enable `LongPathsEnabled` in the registry (requires Administrator). |
| `transcribe-tool` / `tt` is not recognized | The virtual environment is not activated: `.\.venv\Scripts\Activate.ps1` (PowerShell) or `.venv\Scripts\activate.bat` (cmd). |
| Stale build artifacts after cloning | `git clean -ndX` lists ignored files; drop `-n` to remove them. |

`python verify_installation.py` reports the state of every component and is the fastest way to see
what is missing.

## Project Structure

```
Transcribe-tool/
├── transcribe_tool/
│   ├── cli/                 # Interactive menu and command-line interface
│   ├── config/              # Settings and defaults
│   ├── extractors/          # YouTube, TikTok, local file handlers
│   ├── transcriber/         # Whisper, diarization, text processing
│   └── utils/               # Language detection, tokenization, audio processing
├── data/
│   ├── audio/               # Extracted audio files
│   ├── cache/               # Model and download cache
│   └── transcripts/         # Output transcriptions
├── install.sh               # macOS / Linux installer
├── install.ps1              # Windows installer (PowerShell 5.1)
├── install.bat              # Windows double-click wrapper for install.ps1
├── verify_installation.py   # Post-install component check
├── .gitattributes           # Line-ending policy (LF for .sh/.ps1/.py, CRLF for .bat)
├── pyproject.toml
└── README.md
```

## Dependencies

**Core**: openai-whisper, torch, yt-dlp, pandas, pydub, spacy, nltk, wtpsplit,
lingua-language-detector, tqdm, keyring, rich

**Optional extras**:

| Extra | Command | Contents |
|-------|---------|----------|
| `full` | `pip install -e ".[full]"` | pyannote.audio, demucs, psutil, pyreadr, and the `documents` packages |
| `diarization` | `pip install -e ".[diarization]"` | pyannote.audio |
| `rdata` | `pip install -e ".[rdata]"` | pyreadr (`.rds` export) |
| `documents` | `pip install -e ".[documents]"` | pdfplumber, pypdf, python-docx, beautifulsoup4 |
| `dev` | `pip install -e ".[dev]"` | pytest, black, ruff |

## License

MIT License

## Author

Antoine Lemor
