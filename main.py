# main.py
# Driver: Phase-1 crop → Phase-2 cell detection → Phase-3 section detection → Domino + Pip detection
# Saves overlays and a JSON summary.

# ==================================================================
# CONFIGURATION: Easy Toggle
# ==================================================================
BATCH_MODE = True  # Set to True to process ALL images in data/samples/
IMAGE_PATH = "data/samples/IMG_0654.PNG"  # Used when BATCH_MODE = False
OUTPUT_DIR = "data/debug"
# ==================================================================

import os, json, cv2
import numpy as np

from Vision.board_crop import crop_puzzle_boards, detect_dominoes  # Phase 1 + Domino detection
from Vision.pip_detect import detect_all_pips, visualize_pip_counts  # Pip detection
from Vision.cell_grid import detect_cells, CellDetectConfig  # Phase 2
from Vision.section_detect import assign_sections, visualize_sections  # Phase 3
from Vision.constraint_extract import extract_constraints, visualize_badges  # Phase 4

# Your manual palette (INCLUDE beige so undotted beige sections are detected)
PALETTE_INCLUDE = [
    "#c3a2bf",  # purple
    "#e1cbc5",  # light beige
    "#b6b18a",  # olive
    "#b2a5bf",  # lilac
    "#e9bd8c",  # peach
    "#e89fae",  # pink
    "#9dbfc1",  # teal-gray
]

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def draw_cells(rgb, cells, pitch):
    """Draw cell grid overlay (green boxes + red dots)"""
    bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    for c in cells:
        x, y, w, h = c.bbox
        cv2.rectangle(bgr, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.circle(bgr, c.center, max(1, int(round(pitch*0.06))), (0, 0, 255), -1)
    return bgr

def process_image(image_path: str, output_dir: str = OUTPUT_DIR):
    """
    Process a single image through the vision pipeline.
    
    Args:
        image_path: Path to the input image
        output_dir: Base directory for output (creates subfolder per image)
    """
    # Extract image name without extension for subfolder
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(output_dir, image_basename)
    ensure_dir(image_output_dir)

    print(f"\n{'='*70}")
    print(f"Starting NYT Pips Vision Pipeline")
    print(f"{'='*70}")
    print(f"Input: {image_path}")
    print(f"Output directory: {image_output_dir}")

    # Phase 0: Detect dominoes in the tray
    print(f"\n{'='*70}")
    print(f"Phase 0: Domino Detection")
    print(f"{'='*70}")
    
    domino_boxes = detect_dominoes(image_path)
    print(f"Detected {len(domino_boxes)} dominoes in the tray")
    
    # Phase 0.5: Detect pips on each domino
    print(f"\nDetecting pips on dominoes...")
    pip_counts = detect_all_pips(image_path, domino_boxes, debug=False)
    
    # Show pip counts
    for i, (left, right) in enumerate(pip_counts):
        print(f"  Domino {i+1}: [{left}|{right}]")
    
    # Save domino + pip visualization
    domino_vis = visualize_pip_counts(image_path, domino_boxes, pip_counts)
    out_dominoes = os.path.join(image_output_dir, "dominoes.png")
    cv2.imwrite(out_dominoes, domino_vis)
    print(f"[output] Domino + pip visualization: {out_dominoes}")

    # Phase 1: crop 1..N boards
    print(f"\n{'='*70}")
    print(f"Phase 1: Board Cropping")
    print(f"{'='*70}")
    
    crops, boxes = crop_puzzle_boards(image_path, mode="separate")  # returns [RGB np.uint8], [(x,y,w,h)]
    print(f"Detected {len(crops)} puzzle board(s)")
    
    all_boards = []

    for i, crop in enumerate(crops, 1):
        print(f"\n{'='*70}")
        print(f"Processing Board {i}")
        print(f"{'='*70}")
        
        # Phase 2: Detect cell grid
        cfg = CellDetectConfig()  # tweak here if needed
        grid = detect_cells(crop, PALETTE_INCLUDE, cfg)
        
        # Phase 3: Assign sections (NEW!)
        grid = assign_sections(
            grid, 
            crop, 
            PALETTE_INCLUDE, 
            color_tolerance=5.0,  # Lowered from 15.0 to prevent merging distinct sections
            debug=True
        )

        # Phase 4: Extract constraints from badges (NEW!)
        grid = extract_constraints(
            grid,
            crop,
            debug=True
        )

        # Debug overlay: cell grid
        dbg_img = draw_cells(crop, grid.cells, grid.pitch)
        out_dbg = os.path.join(image_output_dir, f"board{i}_cells.png")
        cv2.imwrite(out_dbg, dbg_img)
        
        # Debug overlay: sections (NEW!)
        section_img = visualize_sections(crop, grid, alpha=0.4)
        out_section = os.path.join(image_output_dir, f"board{i}_sections.png")
        cv2.imwrite(out_section, section_img)
        
        # Debug overlay: badges (NEW!)
        badge_img = visualize_badges(crop, grid, alpha=0.6)
        out_badge = os.path.join(image_output_dir, f"board{i}_badges.png")
        cv2.imwrite(out_badge, badge_img)

        all_boards.append({
            "board_index": i,
            "board_w": int(crop.shape[1]),
            "board_h": int(crop.shape[0]),
            "pitch": grid.pitch,
            "offset": [grid.offset[0], grid.offset[1]],
            "cells": [
                {
                    "id": c.id, 
                    "row": c.row, 
                    "col": c.col,
                    "bbox": [int(c.bbox[0]), int(c.bbox[1]), int(c.bbox[2]), int(c.bbox[3])],
                    "center": [int(c.center[0]), int(c.center[1])],
                    "section": c.section  # Now populated!
                } for c in grid.cells
            ],
            "badges": [
                {
                    "bbox": [int(b.bbox[0]), int(b.bbox[1]), int(b.bbox[2]), int(b.bbox[3])],
                    "center": [int(b.center[0]), int(b.center[1])],
                    "color_hex": b.color_hex,
                    "section_color": b.section_color,
                    "section_id": b.section_id,
                    "text": b.text  # Will be None for now (OCR in step 7)
                } for b in (grid.badges if hasattr(grid, 'badges') else [])
            ],
            "debug": grid.debug
        })

    # Add dominoes with pip counts to JSON output
    result = {
        "boards": all_boards,
        "dominoes": [
            {
                "id": i,
                "bbox": [int(x), int(y), int(w), int(h)],
                "pips_left": int(left),
                "pips_right": int(right)
            } for i, ((x, y, w, h), (left, right)) in enumerate(zip(domino_boxes, pip_counts))
        ]
    }

    # Save JSON to two locations:
    # 1. Debug folder (with visualizations): data/debug/IMG_0904/IMG_0904.json
    # 2. Dedicated JSON folder (for solver): data/json/IMG_0904.json
    
    json_filename = f"{image_basename}.json"
    
    # Location 1: Debug folder (with images)
    out_json_debug = os.path.join(image_output_dir, json_filename)
    with open(out_json_debug, "w") as f:
        json.dump(result, f, indent=2)
    
    # Location 2: Dedicated JSON folder (easy access for solver)
    json_folder = "data/json"
    ensure_dir(json_folder)
    out_json_main = os.path.join(json_folder, json_filename)
    with open(out_json_main, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Pipeline Complete")
    print(f"{'='*70}")
    print(f"[output] JSON (debug): {out_json_debug}")
    print(f"[output] JSON (main):  {out_json_main}")
    print(f"[output] Dominoes: {out_dominoes} ({len(domino_boxes)} detected)")
    for i in range(len(crops)):
        print(f"[output] Board {i+1} cell grid: {os.path.join(image_output_dir, f'board{i+1}_cells.png')}")
        print(f"[output] Board {i+1} sections: {os.path.join(image_output_dir, f'board{i+1}_sections.png')}")
        print(f"[output] Board {i+1} badges: {os.path.join(image_output_dir, f'board{i+1}_badges.png')}")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if BATCH_MODE:
        # Batch mode: process all images in samples folder
        samples_dir = "data/samples"
        output_dir = OUTPUT_DIR
        
        print("\n" + "="*70)
        print("BATCH PROCESSING MODE")
        print("="*70)
        
        samples_path = Path(samples_dir)
        if not samples_path.exists():
            print(f"Error: Directory not found: {samples_dir}")
            sys.exit(1)
        
        # Find all images
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        image_files = [f for f in samples_path.iterdir() 
                      if f.is_file() and f.suffix in image_extensions]
        
        if not image_files:
            print(f"No images found in {samples_dir}")
            sys.exit(1)
        
        print(f"Found {len(image_files)} images to process\n")
        
        results = {'success': [], 'failed': []}
        
        for i, image_path in enumerate(sorted(image_files), 1):
            print(f"\n{'='*70}")
            print(f"[{i}/{len(image_files)}] Processing: {image_path.name}")
            print(f"{'='*70}")
            
            try:
                process_image(str(image_path), output_dir)
                results['success'].append(image_path.name)
            except Exception as e:
                print(f"\n❌ ERROR: {e}")
                results['failed'].append((image_path.name, str(e)))
        
        # Summary
        print("\n" + "="*70)
        print("BATCH PROCESSING COMPLETE")
        print("="*70)
        print(f"\n✅ Successful: {len(results['success'])}/{len(image_files)}")
        for name in results['success']:
            print(f"   - {name}")
        
        if results['failed']:
            print(f"\n❌ Failed: {len(results['failed'])}/{len(image_files)}")
            for name, error in results['failed']:
                print(f"   - {name}: {error}")
        
        print(f"\nResults saved to: {output_dir}/")
    
    elif len(sys.argv) > 1:
        # Command line argument provided
        process_image(sys.argv[1], OUTPUT_DIR)
    
    else:
        # Default: process single IMAGE_PATH
        process_image(IMAGE_PATH, OUTPUT_DIR)