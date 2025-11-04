#!/usr/bin/env python3
"""
Pips Solver - Main Entry Point (Updated with Configurable Heuristics)

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
IMAGE_PATH = "data/json/IMG_0925.json"   # Puzzle to solve by default
OUTPUT_DIR = "data/debug"                  # Base output directory
SOLVE_ALL = True                          # Set True to solve all JSON puzzles

# --- Solver Strategy Configuration ---
# These flags control the solver's behavior:

USE_LCV = True
# Least-Constraining-Value ordering during backtracking
# RECOMMENDED: True (helps find solutions faster)

USE_AGGRESSIVE_HEURISTICS = False
# Controls aggressive heuristic techniques (e.g., region sum forcing)
# - True: Uses more aggressive heuristics (faster but riskier - may miss solutions)
# - False: Uses only conservative heuristics (safer, more reliable)
# RECOMMENDED: False for puzzles with complex constraints

SKIP_HEURISTICS = False
# Completely skip heuristics and go straight to backtracking
# - True: Pure backtracking from the start (slowest but most general)
# - False: Apply heuristics first, then backtrack
# RECOMMENDED: False (heuristics help reduce search space)

TIMEOUT_SECONDS = 300
# Maximum time to spend solving a single puzzle
# ============================================================================


def solve_puzzle(input_path: str, output_dir: str = None, verbose: bool = True,
                 use_lcv: bool = USE_LCV,
                 use_aggressive_heuristics: bool = USE_AGGRESSIVE_HEURISTICS,
                 skip_heuristics: bool = SKIP_HEURISTICS,
                 timeout_seconds: int = TIMEOUT_SECONDS):
    """
    Solve a single puzzle and save results.

    Args:
        input_path: Path to input JSON file
        output_dir: Directory for output files (default: data/debug/<puzzle_name>/)
        verbose: Print detailed solving progress
        use_lcv: Enable least-constraining-value ordering
        use_aggressive_heuristics: Enable aggressive heuristic techniques
        skip_heuristics: Skip heuristics phase entirely
        timeout_seconds: Maximum solving time in seconds
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
        # Load and configure solver
        # ---------------------------
        puzzle = PipsPuzzle(str(input_path))
        solver = CSPSolver(
            puzzle, 
            verbose=verbose, 
            use_lcv=use_lcv,
            use_aggressive_heuristics=use_aggressive_heuristics,
            skip_heuristics=skip_heuristics
        )

        if verbose:
            print("\nSolver Configuration:")
            print(f"  LCV ordering: {'ON' if use_lcv else 'OFF'}")
            print(f"  Aggressive heuristics: {'ON' if use_aggressive_heuristics else 'OFF'}")
            print(f"  Skip heuristics: {'YES' if skip_heuristics else 'NO'}")
            print(f"  Timeout: {timeout_seconds}s")
            
            if skip_heuristics:
                print("\n  ⚠ Pure backtracking mode (no heuristics)")
            elif use_aggressive_heuristics:
                print("\n  ⚠ Aggressive mode (faster but may miss solutions)")
            else:
                print("\n  ✓ Conservative mode (safer, more reliable)")

        print("\n💡 Tip: Press Ctrl+C at any time to stop solving\n")
        solved = solver.solve(timeout_seconds=timeout_seconds)

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
            
            if verbose and not skip_heuristics and use_aggressive_heuristics:
                print("\n💡 Tip: Try setting USE_AGGRESSIVE_HEURISTICS = False")
                print("   Aggressive heuristics can sometimes block valid solutions.")
            
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


def solve_all_puzzles(data_dir: str = None, output_dir: str = None,
                      use_lcv: bool = USE_LCV,
                      use_aggressive_heuristics: bool = USE_AGGRESSIVE_HEURISTICS,
                      skip_heuristics: bool = SKIP_HEURISTICS,
                      timeout_seconds: int = TIMEOUT_SECONDS):
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

    print(f"\nFound {len(json_files)} puzzle(s) to solve")
    print("\nSolver Configuration:")
    print(f"  LCV ordering: {'ON' if use_lcv else 'OFF'}")
    print(f"  Aggressive heuristics: {'ON' if use_aggressive_heuristics else 'OFF'}")
    print(f"  Skip heuristics: {'YES' if skip_heuristics else 'NO'}")
    print(f"  Timeout per puzzle: {timeout_seconds}s\n")
    
    results = []

    for i, json_file in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] Solving {json_file.name}...")
        
        solved, puzzle, solver = solve_puzzle(
            str(json_file),
            output_dir=output_dir,
            verbose=False,
            use_lcv=use_lcv,
            use_aggressive_heuristics=use_aggressive_heuristics,
            skip_heuristics=skip_heuristics,
            timeout_seconds=timeout_seconds
        )
        
        result = {
            'file': json_file.name,
            'solved': bool(solved),
            'cells': len(puzzle.cells) if puzzle else None,
            'moves': len(puzzle.move_history) if puzzle else None,
            'backtracks': solver.stats['backtracks'] if solver else None,
            'heuristic_moves': solver.stats['heuristic_moves'] if solver else None
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
            heur_info = f", {r['heuristic_moves']} heuristic" if r['heuristic_moves'] else ""
            print(f" - {r['cells']} cells, {r['moves']} moves{heur_info}, {r['backtracks']} backtracks")
        else:
            print(" - Failed")
    
    print(f"\n{'='*60}")
    print(f"Final Solve Rate: {solve_rate:.1f}% ({solved_count}/{total_count})")
    print(f"{'='*60}")
    
    # Additional insights
    if results:
        avg_backtracks = sum(r['backtracks'] for r in results if r['solved'] and r['backtracks']) / max(solved_count, 1)
        avg_heuristics = sum(r['heuristic_moves'] for r in results if r['solved'] and r['heuristic_moves']) / max(solved_count, 1)
        print(f"\nAverage for solved puzzles:")
        print(f"  Heuristic moves: {avg_heuristics:.1f}")
        print(f"  Backtracks: {avg_backtracks:.1f}")


def run_comparison_test(input_path: str):
    """
    Test the same puzzle with different solver configurations to compare performance.
    """
    puzzle_name = Path(input_path).stem
    
    configs = [
        {
            'name': 'Conservative (Recommended)',
            'use_lcv': True,
            'use_aggressive_heuristics': False,
            'skip_heuristics': False
        },
        {
            'name': 'Aggressive',
            'use_lcv': True,
            'use_aggressive_heuristics': True,
            'skip_heuristics': False
        },
        {
            'name': 'Pure Backtracking',
            'use_lcv': True,
            'use_aggressive_heuristics': False,
            'skip_heuristics': True
        },
        {
            'name': 'No LCV',
            'use_lcv': False,
            'use_aggressive_heuristics': False,
            'skip_heuristics': False
        }
    ]
    
    print(f"\n{'='*60}")
    print(f"COMPARISON TEST: {puzzle_name}")
    print(f"Testing {len(configs)} different solver configurations")
    print(f"{'='*60}\n")
    
    results = []
    
    for config in configs:
        print(f"\n--- Testing: {config['name']} ---")
        
        solved, puzzle, solver = solve_puzzle(
            input_path,
            output_dir=None,
            verbose=False,
            use_lcv=config['use_lcv'],
            use_aggressive_heuristics=config['use_aggressive_heuristics'],
            skip_heuristics=config['skip_heuristics'],
            timeout_seconds=60  # Shorter timeout for comparison
        )
        
        result = {
            'config': config['name'],
            'solved': solved,
            'heuristic_moves': solver.stats['heuristic_moves'] if solver else 0,
            'backtracks': solver.stats['backtracks'] if solver else 0,
            'total_attempts': solver.stats['total_attempts'] if solver else 0
        }
        results.append(result)
        
        status = "✓ SOLVED" if solved else "✗ FAILED"
        print(f"{status} - {result['heuristic_moves']} heuristic, {result['backtracks']} backtracks")
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Configuration':<25} {'Result':<10} {'Heur.':<8} {'Backtr.':<10} {'Total':<10}")
    print(f"{'-'*25} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
    
    for r in results:
        status = "SOLVED" if r['solved'] else "FAILED"
        print(f"{r['config']:<25} {status:<10} {r['heuristic_moves']:<8} {r['backtracks']:<10} {r['total_attempts']:<10}")
    
    print(f"\n{'='*60}")


def main():
    """Main entry point"""
    
    # Check for special commands
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        # Comparison test mode
        if command == "--compare" or command == "-c":
            if len(sys.argv) < 3:
                print("Usage: python -m Solver.main --compare <puzzle.json>")
                sys.exit(1)
            
            input_file = sys.argv[2]
            if not os.path.isabs(input_file):
                project_root = Path(__file__).parent.parent
                input_file = project_root / input_file
            
            if not os.path.exists(input_file):
                print(f"Error: File not found: {input_file}")
                sys.exit(1)
            
            run_comparison_test(str(input_file))
            return
        
        # Solve specific puzzle
        input_file = command
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