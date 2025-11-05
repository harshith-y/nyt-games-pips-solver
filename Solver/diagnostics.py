"""
Diagnostic Script: Understand WHY puzzles fail at 87%

This will help us add TARGETED fixes instead of generic "improvements"
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from Solver
# Assumes this file is in: /Users/harshith/Documents/Projects/NYTGames/Pips/Solver/
sys.path.insert(0, str(Path(__file__).parent))

from puzzle import PipsPuzzle
from solver import CSPSolver
import time


def analyze_failure(json_path: str, timeout: int = 60):
    """Deeply analyze why a puzzle fails."""
    
    puzzle = PipsPuzzle(json_path)
    solver = CSPSolver(
        puzzle,
        verbose=True,  # CRITICAL: see what it's doing
        use_lcv=True,
        use_aggressive_heuristics=False,
        skip_heuristics=False
    )
    
    print(f"\n{'='*80}")
    print(f"ANALYZING: {Path(json_path).name}")
    print(f"{'='*80}")
    print(f"Cells: {len(puzzle.cells)}")
    print(f"Regions: {len(puzzle.regions)}")
    print(f"Dominoes: {len(puzzle.dominoes)}")
    
    # Print region constraints
    print("\nRegion Constraints:")
    for region_id, region in sorted(puzzle.regions.items()):
        print(f"  Region {region_id}: {len(region.cells)} cells, "
              f"constraint: {region.constraint_type} {region.constraint_value}")
    
    start = time.time()
    solved = solver.solve(timeout_seconds=timeout)
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    if solved:
        print(f"✓ SOLVED in {elapsed:.2f}s")
    else:
        print(f"✗ FAILED after {elapsed:.2f}s")
        
        # Analyze the failure
        completion = puzzle.get_completion_percentage()
        print(f"\nCompletion when stopped: {completion:.1%}")
        print(f"Cells filled: {sum(1 for c in puzzle.cells if c.occupied)}/{len(puzzle.cells)}")
        
        # Which regions are complete/incomplete?
        print("\nRegion Status:")
        for region_id, region in sorted(puzzle.regions.items()):
            filled = len(region.get_filled_cells())
            total = len(region.cells)
            status = "✓" if filled == total else "✗"
            print(f"  {status} Region {region_id}: {filled}/{total} cells filled, "
                  f"sum={region.current_sum}, target={region.constraint_value}")
        
        # Where did it get stuck?
        print(f"\nStats when failed:")
        print(f"  Heuristic moves: {solver.stats['heuristic_moves']}")
        print(f"  Search moves: {solver.stats['search_moves']}")
        print(f"  Backtracks: {solver.stats['backtracks']}")
        print(f"  Total attempts: {solver.stats['total_attempts']}")
        
        # Is it thrashing? (lots of backtracks but no progress)
        if solver.stats['backtracks'] > 10000:
            print("\n⚠️  HIGH BACKTRACK COUNT - Likely thrashing in search space")
            print("   Suggestion: Need better pruning or different variable ordering")
        
        # Did heuristics help at all?
        if solver.stats['heuristic_moves'] == 0:
            print("\n⚠️  NO HEURISTIC MOVES - Puzzle is hard from the start")
            print("   Suggestion: Need more powerful heuristics for early game")
        
        # Timeout or exhaustion?
        if elapsed >= timeout - 1:
            print("\n⚠️  TIMEOUT - Didn't exhaust search space")
            print("   Suggestion: Need to prune search space more aggressively")
        else:
            print("\n⚠️  SEARCH EXHAUSTED - No solution found in explored space")
            print("   Suggestion: May need to backtrack further or try different starting points")
    
    print(f"{'='*80}\n")
    
    return {
        'solved': solved,
        'elapsed': elapsed,
        'completion': puzzle.get_completion_percentage(),
        'stats': solver.stats.copy()
    }


def main():
    """Analyze all puzzles and identify patterns in failures."""
    
    # UPDATE THIS: Path to your JSON puzzle files
    # Adjust this to match YOUR project structure
    project_root = Path(__file__).parent.parent  # Goes up to NYTGames/Pips/
    data_dir = project_root / "data" / "json"
    
    # Option 1: Auto-find all JSON files
    if data_dir.exists():
        json_files = sorted(data_dir.glob("*.json"))
        test_files = [str(f) for f in json_files]
        print(f"Found {len(test_files)} JSON files in {data_dir}")
    else:
        # Option 2: Manually specify files (if data/json doesn't exist)
        print(f"Warning: {data_dir} not found. Using manual file list.")
        test_files = [
            # UPDATE THIS LIST with your actual puzzle files
            "IMG_0958.json",
            "IMG_0962.json", 
            "IMG_0963.json",
            "IMG_0971.json",
            "IMG_0999.json",
            "IMG_1001.json",
            "IMG_1002.json",
            "IMG_1011.json",
            "IMG_1012.json",
            "IMG_1013.json",
            "IMG_1014.json"
        ]
        # Prepend the data directory path
        test_files = [str(project_root / "data" / "json" / f) for f in test_files]
    
    if not test_files:
        print("ERROR: No test files found!")
        print(f"Please check that puzzles exist in: {data_dir}")
        print("Or update the 'test_files' list in this script.")
        return
    
    results = []
    
    for filepath in test_files:
        if not Path(filepath).exists():
            print(f"WARNING: File not found: {filepath}")
            continue
        
        result = analyze_failure(filepath, timeout=60)
        result['filename'] = Path(filepath).name
        results.append(result)
    
    # Summary of failures
    failures = [r for r in results if not r['solved']]
    successes = [r for r in results if r['solved']]
    
    print("\n" + "="*80)
    print("FAILURE ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nSolved: {len(successes)}/{len(results)} ({len(successes)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failures)}/{len(results)} ({len(failures)/len(results)*100:.1f}%)\n")
    
    if failures:
        print("Failed puzzles:")
        for r in failures:
            print(f"\n  {r['filename']}:")
            print(f"    Completion: {r['completion']:.1%}")
            print(f"    Backtracks: {r['stats']['backtracks']}")
            print(f"    Heuristic moves: {r['stats']['heuristic_moves']}")
            
            # Classify failure type
            if r['stats']['backtracks'] > 10000:
                print(f"    Type: THRASHING (stuck in local search)")
            elif r['stats']['heuristic_moves'] == 0:
                print(f"    Type: HARD START (no forced moves)")
            elif r['elapsed'] >= 59:
                print(f"    Type: TIMEOUT (search too large)")
            else:
                print(f"    Type: EXHAUSTED (wrong initial choices)")
        
        # Look for patterns
        print("\n" + "-"*80)
        print("PATTERNS IN FAILURES:")
        print("-"*80)
        
        avg_backtracks = sum(r['stats']['backtracks'] for r in failures) / len(failures)
        avg_completion = sum(r['completion'] for r in failures) / len(failures)
        
        print(f"\nAverage completion when failed: {avg_completion:.1%}")
        print(f"Average backtracks: {avg_backtracks:.0f}")
        
        if avg_backtracks > 10000:
            print("\n💡 INSIGHT: High backtrack count suggests search is thrashing")
            print("   RECOMMENDATION: Add restart mechanism or better pruning")
        
        if avg_completion < 0.5:
            print("\n💡 INSIGHT: Failures happen early in solving")
            print("   RECOMMENDATION: Improve variable selection or add more heuristics")
        
        if avg_completion > 0.8:
            print("\n💡 INSIGHT: Failures happen late in solving")
            print("   RECOMMENDATION: Better forward checking for end-game constraints")
    else:
        print("🎉 ALL PUZZLES SOLVED! No failures to analyze.")
        print("\nAverage solve time:", sum(r['elapsed'] for r in successes) / len(successes), "seconds")


if __name__ == "__main__":
    main()