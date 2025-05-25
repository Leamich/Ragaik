#!/usr/bin/env python
"""ru_ocr_folder.py

Minimal script: takes one folder path, runs Russian TroCR OCR on every
*.png image inside, prints a tab-separated line per image:

    filename.png[TAB]Recognised text

No extra output, no visualisation.
Requires: transformers, pillow, tqdm (tqdm optional but left out here).

Usage
-----
python ru_ocr_folder.py /path/to/folder
"""

import argparse
from pathlib import Path
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Run Russian OCR (TroCR) on all *.png files in a folder"
    )
    parser.add_argument(
        "folder",
        help="Directory with PNG images"
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder '{folder}' not found or is not a directory.")

    # Load OCR pipeline once
    pipe = pipeline("image-to-text", model="raxtemur/trocr-base-ru")

    # Iterate over images and output OCR line by line
    for img_path in sorted(folder.glob("*.png")):
        text = pipe(str(img_path))[0]["generated_text"]
        print(f"{img_path.name}\t{text}")

if __name__ == "__main__":
    main()
