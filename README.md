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

## Installation

### Quick Start

```bash
git clone https://github.com/antoinelemor/Transcribe-tool.git
cd Transcribe-tool
./install.sh --all
```

### Manual Installation

```bash
python3 -m venv .venv
source .venv/bin/activate

# Full installation (recommended)
pip install -e ".[full]"

# Optional: SOTA sentence segmentation
pip install wtpsplit

# Verify installation
python verify_installation.py
```

### Requirements

- Python 3.9+
- FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

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
| `-o, --output DIR` | Output directory |

## Output Formats

### CSV

```csv
"segment_id","speaker","text","video_id","title","date","channel"
"1","SPEAKER_00","First sentence.","abc123","Title","20260103","Channel"
"2","SPEAKER_01","Response from second speaker.","abc123","Title","20260103","Channel"
```

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

Requires HuggingFace token with access to Pyannote models:

```bash
export HF_TOKEN="your_token"
```

Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1

## Project Structure

```
Transcribe-tool/
├── transcribe_tool/
│   ├── cli/                 # Command-line interface
│   ├── extractors/          # YouTube, TikTok, local file handlers
│   ├── transcriber/         # Whisper, diarization, text processing
│   └── utils/               # Language detection, tokenization
├── data/
│   ├── audio/               # Extracted audio files
│   └── transcripts/         # Output transcriptions
├── install.sh
├── pyproject.toml
└── README.md
```

## Dependencies

**Core**: openai-whisper, torch, yt-dlp, pandas, spacy, rich

**Optional** (`pip install -e ".[full]"`):
- pyannote.audio (diarization)
- wtpsplit (SOTA segmentation)
- demucs (voice separation)

## License

MIT License

## Author

Antoine Lemor
