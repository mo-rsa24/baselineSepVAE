#!/usr/bin/env bash
set -euo pipefail

# This script finds all .sh files in the launchers directory
# and makes them executable.

echo "Making launcher scripts executable..."
find launchers -type f -name "*.sh" -print0 | xargs -0 chmod +x

# You can add other directories here if needed
# find other_scripts -type f -name "*.sh" -print0 | xargs -0 chmod +x

echo "Done."