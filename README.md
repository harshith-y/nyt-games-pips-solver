# NYT Pips Solver

Automated solver for NYT Pips puzzles using computer vision and constraint satisfaction algorithms.

**Read the full story**: [Solving NYT Pips: A Story of Dominoes, Logic, and Questionable Life Choices](https://medium.com/@hyerraguntla/solving-nyt-pips-a-story-of-dominoes-logic-and-questionable-life-choices-3fe6471fab72)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nyt-pips-solver.git
cd nyt-pips-solver

# Install dependencies
pip install opencv-python easyocr numpy
```

## Quick Start

### 1. Extract Puzzle from Screenshot

Take a screenshot of the NYT Pips puzzle and save it to your input directory. Then run:

```bash
python Vision/main.py
```

**Input**: PNG/JPG screenshot of the puzzle
**Output**: JSON file in `puzzles/` directory with extracted puzzle data

Example output structure:
```json
{
  "dominoes": [[1,2], [3,4], ...],
  "sections": {
    "red": {"cells": [[0,0], [0,1], ...], "constraint": {"operator": "=", "value": 10}},
    ...
  },
  "grid_size": [6, 6]
}
```

### 2. Solve the Puzzle

```bash
python solver.py
```

**Input**: JSON files from `puzzles/` directory
**Output**: Solution JSON in `solutions/` directory

Example solution:
```json
{
  "puzzle_name": "puzzle_001",
  "solution": {
    "dominoes": [
      {"domino": [1,2], "cells": [[0,0], [0,1]], "orientation": "horizontal"},
      ...
    ]
  },
  "solve_time": 0.45,
  "backtracks": 127
}
```

## Usage Examples

### Process a Single Screenshot

```bash
# Place your screenshot in the input folder
cp ~/Downloads/pips_screenshot.png input/

# Run vision pipeline
python Vision/main.py

# The extracted puzzle will be in puzzles/pips_screenshot.json
```

### Batch Process Multiple Puzzles

```bash
# Place all screenshots in input/
cp ~/Downloads/pips_*.png input/

# Run vision pipeline (processes all images)
python Vision/main.py

# Run solver on all extracted puzzles
python solver.py
```

### Working with Existing Puzzle Data

If you already have puzzle JSON files, skip the vision step:

```bash
# Place JSON files in puzzles/
cp my_puzzles/*.json puzzles/

# Solve directly
python solver.py
```

## Configuration

### Vision Pipeline Toggles

Edit flags in `Vision/main.py`:

```python
PROCESS_DOMINOES = True   # Extract domino positions
PROCESS_SECTIONS = True   # Detect colored sections
PROCESS_GRID = True       # Extract cell grid
PROCESS_OCR = True        # Read constraint badges
```

### Solver Options

Edit flags in `solver.py`:

```python
USE_HEURISTICS = True     # Enable conservative heuristic moves
MAX_BACKTRACKS = 1000000  # Backtrack limit before restart
VERBOSE = False           # Print detailed solving steps
```

## Troubleshooting

### Vision Pipeline Issues

**Problem**: Dominoes not detected correctly
- Ensure screenshot shows the entire puzzle board
- Works best with screenshots from NYT mobile app, tablet, or desktop
- Avoid cropped or partial puzzle images

**Problem**: OCR misreading constraint badges
- Ensure badges are clearly visible in screenshot
- System handles =, ≠, >, < symbols
- Higher resolution screenshots improve accuracy

### Solver Issues

**Problem**: Solver takes too long
- Some puzzles require 100K+ backtracks (this is normal)
- Most puzzles solve in < 1 second
- If exceeding 1M backtracks, the puzzle may have no valid solution

**Problem**: No solution found
- Verify the extracted puzzle JSON is correct
- Check that all constraints are properly formatted
- Some puzzles are intentionally unsolvable

## Project Structure

```
nyt-pips-solver/
├── Vision/
│   ├── main.py                  # Vision pipeline orchestrator
│   ├── domino_detection.py      # Detect domino pieces
│   ├── section_detection.py     # Identify colored sections
│   ├── cell_detection.py        # Extract grid cells
│   └── ocr_detection.py         # Read constraint badges
├── solver.py                    # CSP solver
├── input/                       # Place screenshots here
├── puzzles/                     # Extracted puzzle JSON files
├── solutions/                   # Solver output
└── debug/                       # Debug visualizations (auto-generated)
```

## How It Works

**Vision Pipeline**: Uses OpenCV for image processing, spatial clustering for adaptive domino detection, CIE Lab color space for section identification, and EasyOCR for constraint badge reading.

**Solver**: Four-phase CSP approach with conservative heuristics, MRV backtracking with thrashing detection, heuristic backtracking for recovery, and randomized restarts for difficult puzzles.

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- EasyOCR (`easyocr`)
- NumPy (`numpy`)


## Acknowledgments

This solver is for educational purposes. Support the NYT by playing their puzzles at [nytimes.com/puzzles](https://www.nytimes.com/puzzles).
