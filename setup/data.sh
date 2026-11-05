#!/bin/sh
# Laster ned ASL Alphabet Dataset (Cabana 2025, Mendeley) til data/Train og data/Test
# https://data.mendeley.com/datasets/jdyksv2jhh/1  (CC BY 4.0, ca. 900 MB)
set -e

DATA_DIR="$(dirname "$0")/../data"
ZIP="$DATA_DIR/asl-alphabet.zip"

if [ -d "$DATA_DIR/Train" ] && [ -d "$DATA_DIR/Test" ]; then
    echo "$DATA_DIR/Train og Test finnes allerede, hopper over"
    exit 0
fi

mkdir -p "$DATA_DIR"
curl -fL -o "$ZIP" https://data.mendeley.com/public-files/datasets/jdyksv2jhh/files/d3aafafe-7672-40d1-b390-992e2a0c0a95/file_downloaded
unzip -q "$ZIP" -d "$DATA_DIR"
rm "$ZIP"

# macOS-søppel i zip-filen
rm -rf "$DATA_DIR/__MACOSX"
find "$DATA_DIR" -name .DS_Store -delete

echo "Ferdig: $DATA_DIR/Train og $DATA_DIR/Test"
