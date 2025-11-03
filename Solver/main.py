#!/usr/bin/env python3
"""
Pips Solver - Main Entry Point (Improved Version)

Usage:
    python -m Solver.main data/json/puzzle.json
    python -m Solver.main  # Solves all puzzles in data/json/
"""

import sys
import os
from pathlib import Path

from puzzle import PipsPuzzle
from solver import CSPSolver    
from output import SolutionFormatter

# Optional diagnostics (if available)
try:
    from diagnose_solver import SolverDiagnostics
    HAS_DIAGNOSTICS = True
except ImportError:
    HAS_DIAGNOSTICS = False

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGE_PATH = "data/json/IMG_0880.json"   # Puzzle to solve by default
OUTPUT_DIR = "data/debug"                # Base output directory
SOLVE_ALL = True                      # Set True to solve all JSON puzzles
USE_LCV = True                           # Toggle least-constraining-value ordering (RECOMMENDED: True)
# ============================================================================


def solve_puzzle(input_path: str, output_dir: str = None, verbose: bool = True):
    """
    Solve a single puzzle and save results.

    Args:
        input_path: Path to input JSON file
        output_dir: Directory for output files (default: data/debug/<puzzle_name>/)
        verbose: Print detailed solving progress
    """
    puzzle_name = Path(input_path).stem

    # Default output directory: data/debug/<puzzle_name>/
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "data" / "debug" / puzzle_name

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Loading puzzle: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    try:
        # ---------------------------
        # Load and solve the puzzle
        # ---------------------------
        puzzle = PipsPuzzle(str(input_path))
        solver = CSPSolver(puzzle, verbose=verbose, use_lcv=USE_LCV)

        print("\n💡 Tip: Press Ctrl+C at any time to stop solving\n")
        solved = solver.solve()

        # ---------------------------
        # Post-solve output
        # ---------------------------
        if solved:
            print(f"\n{'='*60}")
            print("SUCCESS! Puzzle solved ✓")
            print(f"{'='*60}")

            # Save outputs
            json_output = output_dir / "solution.json"
            text_output = output_dir / "solution.txt"

            SolutionFormatter.save_solution(puzzle, solver.stats, str(json_output))
            SolutionFormatter.save_human_readable(puzzle, str(text_output))

            if verbose:
                print("\n" + SolutionFormatter.format_solution_human_readable(puzzle))
                print(SolutionFormatter.format_grid_visualization(puzzle))

            # Optional diagnostics summary
            if HAS_DIAGNOSTICS:
                SolverDiagnostics.print_summary(solver, puzzle, output_dir)

            return True, puzzle, solver
        else:
            print(f"\n{'='*60}")
            print("FAILED: Could not solve puzzle ✗")
            print(f"{'='*60}")
            return False, puzzle, solver

    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("⚠ Solving interrupted by user (Ctrl+C)")
        print(f"{'='*60}")
        if 'puzzle' in locals() and 'solver' in locals():
            completion = puzzle.get_completion_percentage()
            print(f"\nProgress when stopped: {completion:.1%} complete")
            solver._print_stats()
        return False, None, None

    except Exception as e:
        print(f"\nError while solving {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def solve_all_puzzles(data_dir: str = None, output_dir: str = None):
    """
    Solve all puzzles in data/json/ (or a specified directory)
    """
    if data_dir is None:
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data" / "json"

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory not found: {data_dir}")
        return

    json_files = list(data_path.glob("*.json"))
    if not json_files:
        print(f"No JSON puzzles found in {data_dir}")
        return

    print(f"\nFound {len(json_files)} puzzle(s) to solve\n")
    results = []

    for i, json_file in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] Solving {json_file.name}...")
        
        solved, puzzle, solver = solve_puzzle(
            str(json_file),
            output_dir=output_dir,
            verbose=False
        )
        
        result = {
            'file': json_file.name,
            'solved': bool(solved),
            'cells': len(puzzle.cells) if puzzle else None,
            'moves': len(puzzle.move_history) if puzzle else None,
            'backtracks': solver.stats['backtracks'] if solver else None
        }
        results.append(result)
        
        # Print immediate result
        status = "✓ SOLVED" if solved else "✗ FAILED"
        print(f"  {status}")

    # ---------------------------
    # Print summary
    # ---------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    solved_count = sum(1 for r in results if r['solved'])
    total_count = len(results)
    solve_rate = (solved_count / total_count * 100) if total_count > 0 else 0
    
    print(f"Solved: {solved_count}/{total_count} puzzles ({solve_rate:.1f}%)")
    print(f"{'='*60}\n")

    for r in results:
        status = "✓" if r['solved'] else "✗"
        print(f"{status} {r['file']:30s}", end="")
        if r['solved']:
            print(f" - {r['cells']} cells, {r['moves']} moves, {r['backtracks']} backtracks")
        else:
            print(" - Failed")
    
    print(f"\n{'='*60}")
    print(f"Final Solve Rate: {solve_rate:.1f}% ({solved_count}/{total_count})")
    print(f"{'='*60}")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Solve a specific puzzle
        input_file = sys.argv[1]
        if not os.path.isabs(input_file):
            project_root = Path(__file__).parent.parent
            input_file = project_root / input_file

        if not os.path.exists(input_file):
            print(f"Error: File not found: {input_file}")
            sys.exit(1)

        solve_puzzle(str(input_file), verbose=True)

    elif SOLVE_ALL:
        print("SOLVE_ALL mode enabled - solving all puzzles in data/json/")
        solve_all_puzzles()

    else:
        print(f"Using configured IMAGE_PATH: {IMAGE_PATH}")
        project_root = Path(__file__).parent.parent
        input_file = project_root / IMAGE_PATH

        if not os.path.exists(input_file):
            print(f"Error: File not found: {input_file}")
            sys.exit(1)

        solve_puzzle(str(input_file), verbose=True)


if __name__ == "__main__":
    main()