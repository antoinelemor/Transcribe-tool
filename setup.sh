#!/bin/bash
# Transcribe-Tool Setup Script
# This script is deprecated. Please use install.sh instead.

echo "======================================"
echo "  DEPRECATED: Please use install.sh"
echo "======================================"
echo ""
echo "This script has been replaced by install.sh"
echo ""
echo "Usage (macOS / Linux):"
echo "  ./install.sh          # Core features"
echo "  ./install.sh --all    # All features (recommended)"
echo ""
echo "Usage (Windows 10/11, PowerShell or cmd.exe):"
echo "  powershell -ExecutionPolicy Bypass -File .\\install.ps1 --all"
echo "  or double-click install.bat"
echo ""
echo "  (setup.sh itself requires bash and will not run on native Windows.)"
echo ""

# Forward to install.sh if it exists
if [ -f "install.sh" ]; then
    echo "Running install.sh $@..."
    echo ""
    exec ./install.sh "$@"
else
    echo "Error: install.sh not found"
    exit 1
fi
