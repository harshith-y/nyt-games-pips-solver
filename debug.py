#!/usr/bin/env python3
"""
Debug script to diagnose OCR issues in constraint_extract.py

Usage:
    python debug_ocr.py data/samples/IMG_0920.PNG
    
Or configure IMAGE_PATH below and just run:
    python debug_ocr.py
"""

import sys
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGE_PATH = "data/samples/IMG_0962.PNG"  # Image to debug
OUTPUT_DIR = "data/debug"                 # Where to save debug images
# ============================================================================

# Add Vision to path
sys.path.insert(0, 'Vision')

import cv2
import numpy as np
from Vision.board_crop import crop_puzzle_boards
from Vision.cell_grid import detect_cells, CellDetectConfig
from Vision.section_detect import assign_sections
from Vision.constraint_extract import extract_constraints
import easyocr

PALETTE_INCLUDE = [
    "#c3a2bf",  # purple
    "#e1cbc5",  # light beige
    "#b6b18a",  # olive
    "#b2a5bf",  # lilac
    "#e9bd8c",  # peach
    "#e89fae",  # pink
    "#9dbfc1",  # teal-gray
]

def debug_single_image(image_path: str, output_dir: str = OUTPUT_DIR):
    """Run vision pipeline with maximum debug output"""
    
    image_name = Path(image_path).stem
    debug_dir = os.path.join(output_dir, image_name)
    os.makedirs(debug_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"DEBUGGING: {image_path}")
    print(f"{'='*70}\n")
    
    # Crop board
    print("Step 1: Cropping board...")
    crops, boxes = crop_puzzle_boards(image_path, mode="separate")
    print(f"  Found {len(crops)} board(s)\n")
    
    if not crops:
        print("ERROR: No boards detected!")
        return
    
    crop = crops[0]  # Use first board
    
    # Detect cells
    print("Step 2: Detecting cells...")
    cfg = CellDetectConfig()
    grid = detect_cells(crop, PALETTE_INCLUDE, cfg)
    print(f"  Found {len(grid.cells)} cells\n")
    
    # Assign sections
    print("Step 3: Assigning sections...")
    grid = assign_sections(grid, crop, PALETTE_INCLUDE, color_tolerance=5.0, debug=False)
    print(f"  Found {len(set(c.section for c in grid.cells))} sections\n")
    
    # Extract constraints WITH DEBUG
    print("Step 4: Extracting constraints (WITH DEBUG)...")
    print("="*70)
    
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)
    
    grid = extract_constraints(
        grid, 
        crop, 
        debug=True,  # ← KEY: Turn on debug
        debug_dir=debug_dir,
        reader=reader
    )
    
    print("\n" + "="*70)
    print("FINAL RESULTS:")
    print("="*70)
    
    if hasattr(grid, 'badges'):
        for i, badge in enumerate(grid.badges, 1):
            print(f"\nBadge {i}:")
            print(f"  Position: {badge.center}")
            print(f"  Section: {badge.section_id}")
            print(f"  Color: {badge.color_hex}")
            print(f"  OCR Result: '{badge.text}'")
            
            # Flag potential issues
            if not badge.text:
                print(f"  ⚠️  WARNING: Empty text!")
            elif badge.text.isdigit() and int(badge.text) == 0:
                print(f"  ✓ Detected '0'")
            elif badge.text == "≠":
                print(f"  ✓ Detected not-equals")
            elif badge.text.startswith("<") or badge.text.startswith(">"):
                print(f"  ✓ Detected comparison: {badge.text}")
    else:
        print("No badges detected!")
    
    print(f"\n{'='*70}")
    print(f"Debug images saved to: {debug_dir}/")
    print(f"{'='*70}\n")


def main():
    # Check if image path provided via command line
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use configured IMAGE_PATH
        image_path = IMAGE_PATH
        print(f"Using configured IMAGE_PATH: {image_path}\n")
    
    if not os.path.exists(image_path):
        print(f"ERROR: File not found: {image_path}")
        print(f"\nUsage:")
        print(f"  1. Configure IMAGE_PATH at top of script")
        print(f"  2. Or run: python debug_ocr.py <image_path>")
        sys.exit(1)
    
    debug_single_image(image_path, OUTPUT_DIR)


if __name__ == "__main__":
    main()