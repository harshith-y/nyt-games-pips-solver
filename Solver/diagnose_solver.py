#!/usr/bin/env python3
"""
Diagnostic script to analyze solver convergence issues
"""
import sys
sys.path.insert(0, '/mnt/user-data/uploads')

from puzzle import PipsPuzzle
from constraints import ConstraintChecker, HeuristicDetector
from solver import CSPSolver

def analyze_puzzle_structure(puzzle):
    """Analyze the puzzle structure for potential issues"""
    print("\n" + "="*70)
    print("PUZZLE STRUCTURE ANALYSIS")
    print("="*70)
    
    print(f"\nTotal cells: {len(puzzle.cells)}")
    print(f"Total regions: {len(puzzle.regions)}")
    print(f"Total dominoes: {len(puzzle.dominoes)}")
    print(f"Boards: {puzzle.num_boards}")
    
    # Analyze regions
    print("\n--- REGION ANALYSIS ---")
    for rid, region in sorted(puzzle.regions.items()):
        print(f"Region {rid}: {len(region.cells)} cells, "
              f"constraint={region.constraint_type} {region.constraint_value}")
    
    # Check for disconnected cells
    print("\n--- CONNECTIVITY ANALYSIS ---")
    isolated_cells = []
    for cell in puzzle.cells:
        adj = puzzle.get_adjacent_cells(cell)
        if len(adj) == 0:
            isolated_cells.append(cell)
        elif len(adj) == 1:
            print(f"Cell {cell.id} at ({cell.row},{cell.col}) has only 1 neighbor (board {cell.board_index})")
    
    if isolated_cells:
        print(f"\n⚠ WARNING: {len(isolated_cells)} isolated cells found!")
        for cell in isolated_cells:
            print(f"  Cell {cell.id} at ({cell.row},{cell.col}) board {cell.board_index}")
    
    # Analyze domino constraints
    print("\n--- DOMINO ANALYSIS ---")
    domino_values = {}
    for d in puzzle.dominoes:
        key = tuple(sorted([d.pips_left, d.pips_right]))
        domino_values[key] = domino_values.get(key, 0) + 1
    
    print("Available domino pip combinations:")
    for key, count in sorted(domino_values.items()):
        print(f"  {key}: {count}x")
    
    # Check constraint feasibility
    print("\n--- CONSTRAINT FEASIBILITY ---")
    for rid, region in sorted(puzzle.regions.items()):
        if region.constraint_type == 'sum':
            target = region.constraint_value
            num_cells = len(region.cells)
            min_sum = 0
            max_sum = 6 * num_cells
            print(f"Region {rid}: target={target}, cells={num_cells}, "
                  f"range=[{min_sum}, {max_sum}]", end="")
            if target < min_sum or target > max_sum:
                print(" ⚠ IMPOSSIBLE!")
            elif target == 0 and num_cells > 0:
                print(" (requires all 0s)")
            else:
                print(" ✓")


def analyze_initial_moves(puzzle):
    """Analyze what moves are available at the start"""
    print("\n" + "="*70)
    print("INITIAL MOVE ANALYSIS")
    print("="*70)
    
    # Check forced moves
    forced = HeuristicDetector.find_forced_moves(puzzle)
    print(f"\nForced moves: {len(forced)}")
    if forced:
        for mv in forced[:5]:  # Show first 5
            c1, c2 = mv['cell1'], mv['cell2']
            placements = mv['placements']
            if placements:
                d = placements[0]['domino']
                ori = placements[0]['orientation']
                print(f"  Cells {c1.id}-{c2.id}: {d} ({ori})")
    
    # Check constrained placements
    constrained = HeuristicDetector.find_constrained_placements(puzzle)
    print(f"\nConstrained placements: {len(constrained)}")
    for mv in constrained[:5]:
        c1, c2 = mv['cell1'], mv['cell2']
        print(f"  Cells {c1.id}-{c2.id}: {mv['num_options']} options ({mv['reason']})")
    
    # Check corner constraints
    corners = HeuristicDetector.find_corner_constraints(puzzle)
    print(f"\nCorner constraints: {len(corners)}")
    for mv in corners[:5]:
        c1, c2 = mv['cell1'], mv['cell2']
        print(f"  Cell {c1.id} → {c2.id}: {mv['num_options']} options")
    
    # Count all valid moves
    all_moves = HeuristicDetector.get_all_valid_moves(puzzle)
    print(f"\nTotal valid moves at start: {len(all_moves)}")
    
    # Analyze branching factor
    move_counts = {}
    for cell in puzzle.cells:
        if cell.occupied:
            continue
        count = 0
        for mv in all_moves:
            if mv['cell1'].id == cell.id or mv['cell2'].id == cell.id:
                count += 1
        if count > 0:
            move_counts[cell.id] = count
    
    if move_counts:
        avg_moves = sum(move_counts.values()) / len(move_counts)
        max_moves = max(move_counts.values())
        print(f"\nBranching factor per cell: avg={avg_moves:.1f}, max={max_moves}")
        
        # Show cells with highest branching
        top_cells = sorted(move_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Cells with most options:")
        for cell_id, count in top_cells:
            cell = puzzle.cell_by_id[cell_id]
            print(f"  Cell {cell_id} at ({cell.row},{cell.col}): {count} moves")


def simulate_heuristic_phase(puzzle):
    """Simulate the heuristic phase to see what gets solved"""
    print("\n" + "="*70)
    print("HEURISTIC PHASE SIMULATION")
    print("="*70)
    
    solver = CSPSolver(puzzle, verbose=False)
    
    rounds = 0
    max_rounds = 50
    
    while rounds < max_rounds and not puzzle.is_complete():
        if not solver._apply_heuristics():
            break
        rounds += 1
        completion = puzzle.get_completion_percentage()
        print(f"Round {rounds}: {completion:.1%} complete ({len(puzzle.move_history)} placements)")
    
    print(f"\nHeuristic phase complete after {rounds} rounds")
    print(f"Final completion: {puzzle.get_completion_percentage():.1%}")
    print(f"Cells remaining: {sum(1 for c in puzzle.cells if not c.occupied)}")
    
    # Show which cells are still empty
    empty_cells = [c for c in puzzle.cells if not c.occupied]
    if empty_cells:
        print(f"\nEmpty cells ({len(empty_cells)}):")
        for cell in empty_cells[:10]:  # Show first 10
            adj = puzzle.get_empty_adjacent_cells(cell)
            region = puzzle.get_region(cell)
            print(f"  Cell {cell.id} at ({cell.row},{cell.col}): "
                  f"region={region.section_id}, {len(adj)} empty neighbors")
    
    return solver


def analyze_search_space(puzzle, solver):
    """Analyze the search space complexity"""
    print("\n" + "="*70)
    print("SEARCH SPACE ANALYSIS")
    print("="*70)
    
    # Get current valid moves
    valid_moves = HeuristicDetector.get_all_valid_moves(puzzle)
    print(f"\nValid moves after heuristics: {len(valid_moves)}")
    
    # Count moves per cell pair
    pair_counts = {}
    for mv in valid_moves:
        key = tuple(sorted([mv['cell1'].id, mv['cell2'].id]))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    
    if pair_counts:
        avg_per_pair = sum(pair_counts.values()) / len(pair_counts)
        max_per_pair = max(pair_counts.values())
        print(f"Moves per cell-pair: avg={avg_per_pair:.1f}, max={max_per_pair}")
        
        # Show pairs with most options
        top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nCell pairs with most domino options:")
        for (c1_id, c2_id), count in top_pairs:
            c1 = puzzle.cell_by_id[c1_id]
            c2 = puzzle.cell_by_id[c2_id]
            print(f"  Cells {c1_id}-{c2_id}: {count} valid dominoes")
    
    # Estimate search tree size
    empty_cells = sum(1 for c in puzzle.cells if not c.occupied)
    remaining_dominoes = sum(1 for d in puzzle.dominoes if not d.placed)
    
    print(f"\nRemaining: {empty_cells} cells, {remaining_dominoes} dominoes")
    print(f"Naive search space: ~{remaining_dominoes}^{empty_cells//2} (very rough upper bound)")
    
    # Check for bottlenecks
    print("\n--- BOTTLENECK DETECTION ---")
    bottlenecks = []
    for cell in puzzle.cells:
        if cell.occupied:
            continue
        adj = puzzle.get_empty_adjacent_cells(cell)
        if len(adj) == 0:
            bottlenecks.append((cell, "isolated"))
        elif len(adj) == 1:
            # Check if the neighbor also only has this cell
            neighbor = adj[0]
            neighbor_adj = puzzle.get_empty_adjacent_cells(neighbor)
            if len(neighbor_adj) == 1:
                bottlenecks.append((cell, f"forced pair with cell {neighbor.id}"))
    
    if bottlenecks:
        print(f"\n⚠ Found {len(bottlenecks)} potential bottlenecks:")
        for cell, reason in bottlenecks:
            print(f"  Cell {cell.id}: {reason}")
    else:
        print("No obvious bottlenecks detected")


def main():
    puzzle_path = '/mnt/user-data/uploads/IMG_0654.json'
    
    print(f"Loading puzzle: {puzzle_path}")
    puzzle = PipsPuzzle(puzzle_path)
    
    # Run analyses
    analyze_puzzle_structure(puzzle)
    analyze_initial_moves(puzzle)
    solver = simulate_heuristic_phase(puzzle)
    analyze_search_space(puzzle, solver)
    
    # Final recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    completion = puzzle.get_completion_percentage()
    if completion < 0.5:
        print("\n⚠ Heuristics solving <50% of puzzle")
        print("   → Need stronger constraint propagation")
        print("   → Consider more sophisticated heuristics")
    elif completion < 0.8:
        print("\n⚠ Heuristics solving 50-80% of puzzle")
        print("   → Backtracking search is handling remainder")
        print("   → May benefit from better move ordering")
    else:
        print("\n✓ Heuristics solving >80% of puzzle")
        print("   → Backtracking just finishing up")
    
    remaining = sum(1 for c in puzzle.cells if not c.occupied)
    if remaining > 0:
        valid_moves = HeuristicDetector.get_all_valid_moves(puzzle)
        if valid_moves:
            branching = len(valid_moves) / (remaining // 2) if remaining > 0 else 0
            print(f"\nBranching factor in search: {branching:.1f} options per placement")
            if branching > 10:
                print("   → ⚠ High branching factor - search will be slow")
                print("   → Consider move ordering heuristics (MRV, LCV)")
            elif branching > 5:
                print("   → Moderate branching - search may be slow")
            else:
                print("   → Low branching - search should be fast")


if __name__ == "__main__":
    main()
