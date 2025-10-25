# main.py
# Driver: Phase-1 crop → Phase-2 cell detection → Phase-3 section detection → Domino + Pip detection
# Saves overlays and a JSON summary.

import os, json, cv2
import numpy as np

from Vision.board_crop import crop_puzzle_boards, detect_dominoes  # Phase 1 + Domino detection
from Vision.pip_detect import detect_all_pips, visualize_pip_counts  # Pip detection
from Vision.cell_grid import detect_cells, CellDetectConfig  # Phase 2
from Vision.section_detect import assign_sections, visualize_sections  # Phase 3
from Vision.constraint_extract import extract_constraints, visualize_badges  # Phase 4

IMAGE_PATH = "data/samples/pips_example1.PNG"
OUTPUT_DIR = "data/debug"

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

def main():
    # Extract image name without extension for subfolder
    image_basename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    image_output_dir = os.path.join(OUTPUT_DIR, image_basename)
    ensure_dir(image_output_dir)

    print(f"\n{'='*70}")
    print(f"Starting NYT Pips Vision Pipeline")
    print(f"{'='*70}")
    print(f"Input: {IMAGE_PATH}")
    print(f"Output directory: {image_output_dir}")

    # Phase 0: Detect dominoes in the tray
    print(f"\n{'='*70}")
    print(f"Phase 0: Domino Detection")
    print(f"{'='*70}")
    
    domino_boxes = detect_dominoes(IMAGE_PATH)
    print(f"Detected {len(domino_boxes)} dominoes in the tray")
    
    # Phase 0.5: Detect pips on each domino
    print(f"\nDetecting pips on dominoes...")
    pip_counts = detect_all_pips(IMAGE_PATH, domino_boxes, debug=False)
    
    # Show pip counts
    for i, (left, right) in enumerate(pip_counts):
        print(f"  Domino {i+1}: [{left}|{right}]")
    
    # Save domino + pip visualization
    domino_vis = visualize_pip_counts(IMAGE_PATH, domino_boxes, pip_counts)
    out_dominoes = os.path.join(image_output_dir, "dominoes.png")
    cv2.imwrite(out_dominoes, domino_vis)
    print(f"[output] Domino + pip visualization: {out_dominoes}")

    # Phase 1: crop 1..N boards
    print(f"\n{'='*70}")
    print(f"Phase 1: Board Cropping")
    print(f"{'='*70}")
    
    crops, boxes = crop_puzzle_boards(IMAGE_PATH, mode="separate")  # returns [RGB np.uint8], [(x,y,w,h)]
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
            color_tolerance=15.0,  # Adjust if needed
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

    # Save JSON
    out_json = os.path.join(image_output_dir, "phase2_cells.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Pipeline Complete")
    print(f"{'='*70}")
    print(f"[output] JSON: {out_json}")
    print(f"[output] Dominoes: {out_dominoes} ({len(domino_boxes)} detected)")
    for i in range(len(crops)):
        print(f"[output] Board {i+1} cell grid: {os.path.join(image_output_dir, f'board{i+1}_cells.png')}")
        print(f"[output] Board {i+1} sections: {os.path.join(image_output_dir, f'board{i+1}_sections.png')}")
        print(f"[output] Board {i+1} badges: {os.path.join(image_output_dir, f'board{i+1}_badges.png')}")

if __name__ == "__main__":
    main()