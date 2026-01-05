#!/bin/bash
# Transcribe-Tool Setup Script
# This script is deprecated. Please use install.sh instead.

echo "======================================"
echo "  DEPRECATED: Please use install.sh"
echo "======================================"
echo ""
echo "This script has been replaced by install.sh"
echo ""
echo "Usage:"
echo "  ./install.sh          # Core features"
echo "  ./install.sh --all    # All features (recommended)"
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
