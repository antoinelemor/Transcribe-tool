#!/bin/bash
#
# Transcribe Tool - Quick Installation Script
#
# This script automates the installation process for Transcribe Tool.
# It creates a virtual environment, installs dependencies, and verifies the installation.
#
# Usage:
#   ./install.sh              # Install core features
#   ./install.sh --all        # Install all features (recommended)
#   ./install.sh --full       # Alias for --all
#
# Author: Antoine Lemor

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Banner
echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                               ║"
echo "║  ████████╗██████╗  █████╗ ███╗   ██╗███████╗ ██████╗██████╗ ██╗██████╗ ███████╗║"
echo "║  ╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔══██╗██║██╔══██╗██╔════╝║"
echo "║     ██║   ██████╔╝███████║██╔██╗ ██║███████╗██║     ██████╔╝██║██████╔╝█████╗  ║"
echo "║     ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██║     ██╔══██╗██║██╔══██╗██╔══╝  ║"
echo "║     ██║   ██║  ██║██║  ██║██║ ╚████║███████║╚██████╗██║  ██║██║██████╔╝███████╗║"
echo "║     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝╚═════╝ ╚══════╝║"
echo "║                                                                               ║"
echo "║                  TOOL                                                         ║"
echo "║                                                                               ║"
echo "║           Audio Extraction & Transcription Pipeline                           ║"
echo "║     YouTube, TikTok, Local Files -> Whisper -> Diarization -> Export          ║"
echo "║                                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Parse arguments
INSTALL_TYPE="core"
if [ "$1" == "--all" ] || [ "$1" == "--full" ]; then
    INSTALL_TYPE="full"
    echo -e "${BLUE}Installing with ALL features (diarization, voice separation)...${NC}"
elif [ "$1" == "--dev" ]; then
    INSTALL_TYPE="dev"
    echo -e "${BLUE}Installing with DEVELOPMENT tools...${NC}"
else
    echo -e "${BLUE}Installing with CORE features...${NC}"
    echo -e "${YELLOW}Tip: Use './install.sh --all' for all features (diarization, voice separation)${NC}"
fi
echo ""

# Check Python version
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Checking Python version..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "  Please install Python 3.9 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}✗ Python $PYTHON_VERSION is too old${NC}"
    echo "  Required: Python 3.9 or higher"
    echo "  Found: Python $PYTHON_VERSION"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
echo ""

# Check FFmpeg
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Checking FFmpeg..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}⚠ FFmpeg not found${NC}"
    echo "  FFmpeg is required for audio extraction."
    echo ""
    echo "  Install FFmpeg:"
    echo "    macOS:   brew install ffmpeg"
    echo "    Ubuntu:  sudo apt install ffmpeg"
    echo "    Windows: Download from https://ffmpeg.org/download.html"
    echo ""
    read -p "  Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | cut -d' ' -f3)
    echo -e "${GREEN}✓ FFmpeg $FFMPEG_VERSION found${NC}"
fi
echo ""

# Create virtual environment
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Creating virtual environment..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d ".venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists at .venv/${NC}"
    read -p "  Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        echo -e "${GREEN}✓ Removed existing virtual environment${NC}"
    else
        echo -e "${BLUE}→ Using existing virtual environment${NC}"
    fi
fi

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created at .venv/${NC}"
fi
echo ""

# Configure VS Code
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Configuring VS Code..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p .vscode

cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
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
EOF

cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "eamodio.gitlens"
    ]
}
EOF

echo -e "${GREEN}✓ VS Code configured to use .venv as default Python interpreter${NC}"
echo ""

# Activate virtual environment
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Activating virtual environment..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: Upgrading pip..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded to latest version${NC}"
echo ""

# Install Transcribe Tool
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: Installing Transcribe Tool..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$INSTALL_TYPE" == "full" ]; then
    echo "  Installing with full features (this may take a few minutes)..."
    pip install -e ".[full]"
elif [ "$INSTALL_TYPE" == "dev" ]; then
    pip install -e ".[dev]"
else
    pip install -e .
fi

echo -e "${GREEN}✓ Transcribe Tool installed successfully${NC}"
echo ""

# Install spaCy models (optional)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 8: Installing language models (optional)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -m spacy download en_core_web_sm 2>/dev/null && echo -e "  ${GREEN}✓ English spaCy model installed${NC}" || echo -e "  ${YELLOW}- English model not installed (optional)${NC}"
python -m spacy download fr_core_news_sm 2>/dev/null && echo -e "  ${GREEN}✓ French spaCy model installed${NC}" || echo -e "  ${YELLOW}- French model not installed (optional)${NC}"
echo ""

# Verify installation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 9: Verifying installation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python verify_installation.py

# Success message
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║              INSTALLATION COMPLETE!                      ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the virtual environment:"
echo -e "     ${BLUE}source .venv/bin/activate${NC}"
echo ""
echo "  2. Launch Transcribe Tool:"
echo -e "     ${BLUE}transcribe-tool${NC}"
echo -e "     ${CYAN}# or use the short alias:${NC}"
echo -e "     ${BLUE}tt${NC}"
echo ""
echo "  3. Quick transcription:"
echo -e "     ${BLUE}transcribe-tool --transcribe audio.mp3${NC}"
echo ""
echo "  4. Download from YouTube:"
echo -e "     ${BLUE}transcribe-tool --youtube \"URL\"${NC}"
echo ""
echo "  5. VS Code users:"
echo -e "     ${GREEN}✓ VS Code is already configured to use .venv${NC}"
echo -e "     ${GREEN}  Just open/reload the workspace in VS Code!${NC}"
echo ""
echo "Optional setup for speaker diarization:"
echo ""
echo "  Set your HuggingFace token (required for diarization):"
echo -e "  ${YELLOW}export HF_TOKEN=\"your_huggingface_token\"${NC}"
echo ""
echo "  Accept model terms at:"
echo -e "  ${CYAN}https://huggingface.co/pyannote/speaker-diarization-3.1${NC}"
echo ""
echo "For help and support:"
echo "  README: README.md"
echo "  GitHub: https://github.com/antoine-lemor/Transcribe-tool"
echo ""
echo "Happy transcribing!"
echo ""
