#!/usr/bin/env bash
# Download CheXmask VinDr-CXR preprocessed masks from PhysioNet.
# Only fetches the VinDr-CXR CSV (~309 MB) — not the full 15.6 GB zip.
# No authentication required (CC-BY 4.0 public access).
set -euo pipefail

DEST=/datasets/mmolefe/chexmask
mkdir -p "$DEST"

BASE="https://physionet.org/files/chexmask-cxr-segmentation-data/1.0.0"
OUT="${DEST}/VinDr-CXR_preprocessed.csv"

echo "[1/1] Downloading VinDr-CXR preprocessed masks (1024×1024, ~309 MB)..."
echo "      Destination: ${OUT}"

if wget -N -c --show-progress \
        "${BASE}/Preprocessed/VinDr-CXR.csv" \
        -O "${OUT}" 2>&1; then
    echo "Done (wget) → ${OUT}"
else
    echo "wget failed, trying curl..."
    curl -L --retry 5 --retry-delay 3 --progress-bar \
         "${BASE}/Preprocessed/VinDr-CXR.csv" \
         -o "${OUT}"
    echo "Done (curl) → ${OUT}"
fi

echo "File size: $(du -sh "${OUT}" | cut -f1)"
echo "Row count: $(wc -l < "${OUT}") lines (including header)"
