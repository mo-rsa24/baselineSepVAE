#!/usr/bin/env bash
set -euo pipefail

# Wrapper script to automatically pull the latest code,
# set permissions, and execute a command.

echo ">>> 1. Pulling latest changes from Git..."
git pull

echo ">>> 2. Ensuring all launcher scripts are executable..."
# Assumes the helper script from Step 1 exists and is executable
./make_scripts_executable.sh

echo ">>> 3. Executing your command: $@"
echo "----------------------------------------------------"

# Execute whatever command was passed to this script
"$@"