#!/bin/bash
# Quick start script for downloading and preprocessing video data

set -e

# Configuration
INPUT_CSV="example_videos.csv"
ANNOTATIONS_CSV="example_annotations.csv"
VIDEO_DIR="../data/raw_videos"
OUTPUT_DIR="../data/train"
ANNOTATIONS_JSON="${OUTPUT_DIR}/annotations.json"

# Create directories
mkdir -p "${VIDEO_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "=== Step 1: Downloading videos ==="
python download_data.py \
    --input "${INPUT_CSV}" \
    --format csv \
    --url-column url \
    --id-column video_id \
    --output-dir "${VIDEO_DIR}" \
    --metadata-output "${OUTPUT_DIR}/download_metadata.json"

echo ""
echo "=== Step 2: Preprocessing annotations ==="
python preprocess_data.py \
    --input "${ANNOTATIONS_CSV}" \
    --format csv \
    --video-dir "${VIDEO_DIR}" \
    --output "${ANNOTATIONS_JSON}" \
    --validate

echo ""
echo "=== Done! ==="
echo "Annotations saved to: ${ANNOTATIONS_JSON}"
echo "Videos saved to: ${VIDEO_DIR}"

