#!/bin/bash
# Script to run the video viewer with virtual environment

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate venv
if [ -d "$DIR/.venv" ]; then
    source "$DIR/.venv/bin/activate"
else
    echo "Virtual environment not found at $DIR/.venv"
    echo "Please set it up first."
    exit 1
fi

# Run the viewer
video_viewer "$@"
