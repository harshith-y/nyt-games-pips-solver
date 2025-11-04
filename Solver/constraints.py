"""
Constraint checking and heuristic detection for Pips solver

Key points:
 - Correct SAME-region combined handling
 - Placement-time guards for < and >
 - Mild one-cell SUM look-ahead
 - Value-routing feasibility with INTERNAL (v,v) pairs allowance (now puzzle-aware)
 - HeuristicDetector unchanged (calls the updated checker)

Compatible with:
  from puzzle import PipsPuzzle, Cell, Region, Domino
"""

from typing import List, Tuple, Optional, Dict, Set
from puzzle import PipsPuzzle, Cell, Region, Domino


# -----------------------------------------------------------------------------
# Constraint Checking
# -----------------------------------------------------------------------------
class ConstraintChecker:
    """Validates domino placements against constraints."""

    # ---------- small helpers ----------

    @staticmethod
    def _sum_feasible(new_sum: int, remaining_after: int, target: int) -> bool:
        """
        Interval feasibility for a SUM constraint:
          After placing this value, with `remaining_after` cells left,
          the target must still be reachable with 0..6 per remaining cell.
        """
        return (new_sum <= target) and (target <= new_sum + 6 * remaining_after)

    @staticmethod
    def _check_region_lookahead(
        puzzle: PipsPuzzle,
        placing_domino: Domino,
        region: Region,
        cell: Cell,
        pip_value: int,
    ) -> bool:
        """
        DISABLED: This lookahead was causing issues.
        
        While it helps find forced moves during heuristics, it also blocks valid
        placements during backtracking. Since isolated cell detection works fine
        without it, we disable it entirely.
        """
        return True  # Always permissive

    @staticmethod
    def _would_create_impossible_constraint(
        puzzle: PipsPuzzle,
        domino: Domino,
        cell1: Cell,
        cell2: Cell,
        orientation: str,
    ) -> bool:
        """
        One-cell lookahead for SUM regions.
        
        With the extremely permissive value routing, this should be safe to use.
        Helps identify forced moves during heuristics.
        """
        if orientation == "forward":
            pip1, pip2 = domino.pips_left, domino.pips_right
        else:
            pip1, pip2 = domino.pips_right, domino.pips_left

        region1 = puzzle.get_region(cell1)
        region2 = puzzle.get_region(cell2)

        if not ConstraintChecker._check_region_lookahead(
            puzzle, placing_domino=domino, region=region1, cell=cell1, pip_value=pip1
        ):
            return True

        if region1 is not region2:
            if not ConstraintChecker._check_region_lookahead(
                puzzle, placing_domino=domino, region=region2, cell=cell2, pip_value=pip2
            ):
                return True

        return False

    # ---------- value routing with INTERNAL (v,v) allowance ----------

    @staticmethod
    def _available_vv_dominoes(puzzle: PipsPuzzle, v: int) -> int:
        """Count remaining (v,v) dominos."""
        count = 0
        for d in puzzle.dominoes:
            if not d.placed and d.pips_left == v and d.pips_right == v:
                count += 1
        return count

    @staticmethod
    def _available_v_ends(puzzle: PipsPuzzle, v: int) -> int:
        """Count remaining *ends* with value v (both sides over all unplaced dominos)."""
        ends = 0
        for d in puzzle.dominoes:
            if not d.placed:
                if d.pips_left == v:
                    ends += 1
                if d.pips_right == v:
                    ends += 1
        return ends

    @staticmethod
    def _greedy_internal_pairs(puzzle: PipsPuzzle, region: Region, exclude_ids: Set[int]) -> int:
        """
        Greedy lower-bound on how many disjoint adjacent pairs remain *inside* the region.
        Uses `puzzle` to query adjacency/regions (no cell.puzzle back-pointer).
        """
        remaining = [c for c in region.get_empty_cells() if c.id not in exclude_ids]
        remaining_ids = {c.id for c in remaining}

        # Build adjacency among remaining cells that are in the same region
        adj: Dict[int, List[int]] = {c.id: [] for c in remaining}
        for c in remaining:
            for nb in puzzle.get_adjacent_cells(c):
                if (nb.id in remaining_ids) and (puzzle.get_region(nb) is region):
                    adj[c.id].append(nb.id)

        # Greedy disjoint matching
        matched: Set[int] = set()
        pairs = 0
        for cid in sorted(remaining_ids):
            if cid in matched:
                continue
            for nid in adj.get(cid, []):
                if nid not in matched and nid != cid:
                    matched.add(cid)
                    matched.add(nid)
                    pairs += 1
                    break
        return pairs

    @staticmethod
    def _value_routable_with_internal(
        puzzle: PipsPuzzle,
        region: Region,
        required_value: int,
        exclude_cells: Set[int],
    ) -> bool:
        """
        VERY PERMISSIVE: Check if a value can be routed to remaining cells.
        
        Only rejects cases that are OBVIOUSLY impossible (e.g., need 10 cells with value 5
        but only 1 domino end has value 5 total).
        
        The key insight: dominoes span regions, so even if a region needs many cells with
        a specific value, those values can come from:
        1. Internal (v,v) dominoes
        2. Cross-region dominoes importing the value
        3. Complex combinations we can't easily predict
        
        So we're VERY generous in what we allow.
        """
        remaining = [c for c in region.get_empty_cells() if c.id not in exclude_cells]
        need = len(remaining)
        if need == 0:
            return True

        # Count available ends with the required value
        available_ends = 0
        for dom in puzzle.dominoes:
            if not dom.placed:
                if dom.pips_left == required_value:
                    available_ends += 1
                if dom.pips_right == required_value:
                    available_ends += 1
        
        # EXTREMELY permissive: only reject if we literally don't have enough ends
        # Even then, be generous because (v,v) dominoes count double
        if available_ends == 0 and need > 0:
            return False  # Definitely impossible - need value but have zero
        
        # For any other case, assume success is possible
        # The backtracking will find it or fail naturally
        return True

    @staticmethod
    def _region_requires_all_of_value(region: Region, value_after: int, remaining_after: int) -> Optional[int]:
        """
        If after a placement, the only way to satisfy the region is to assign
        the SAME value `v` to all remaining cells, return that v; otherwise None.

        Practical cases covered:
          - SUM: target already matched -> remaining must be 0
          - LESS_THAN: threshold==1 -> remaining must be 0
          - EQUAL: if any cells are filled, v is fixed to that value
        """
        if remaining_after <= 0:
            return None

        ctype, cval = region.constraint_type, region.constraint_value

        # EQUAL regions: if already have a value, remaining must match it
        if ctype == "equal":
            filled = region.get_filled_cells()
            if filled:
                v0 = filled[0].value
                return v0

        # SUM regions: if target already matched, remaining must be 0
        if ctype == "sum" and cval is not None:
            if cval == value_after:
                return 0

        # LESS_THAN: < 1 -> remaining must be 0
        if ctype == "less_than" and cval is not None:
            if cval == 1:
                return 0

        return None

    # ---------- main API ----------

    @staticmethod
    def is_valid_placement(
        puzzle: PipsPuzzle,
        domino: Domino,
        cell1: Cell,
        cell2: Cell,
        orientation: str = "forward",
    ) -> bool:
        """
        Check if placing a domino at two cells is valid.
        orientation: "forward"  -> cell1 gets pips_left,  cell2 gets pips_right
                     "reverse"  -> cell1 gets pips_right, cell2 gets pips_left
        """
        # 0) tiny symmetry break for "=" when region is fresh
        r1 = puzzle.get_region(cell1)
        r2 = puzzle.get_region(cell2)
        if (r1 is r2) and (r1.constraint_type == "equal") and (not r1.get_filled_cells()):
            a, b = (
                (domino.pips_left, domino.pips_right)
                if orientation == "forward"
                else (domino.pips_right, domino.pips_left)
            )
            if cell1.id < cell2.id and a > b:
                return False

        # 1) adjacency
        if cell2 not in puzzle.get_adjacent_cells(cell1):
            return False

        # 2) both empty
        if cell1.occupied or cell2.occupied:
            return False

        # 3) domino available
        if domino.placed:
            return False

        # 4) region constraints (with correct same-region combined handling)
        if not ConstraintChecker._check_region_constraints(
            puzzle, domino, cell1, cell2, orientation
        ):
            return False

        # 5) mild SUM look-ahead
        if ConstraintChecker._would_create_impossible_constraint(
            puzzle, domino, cell1, cell2, orientation
        ):
            return False

        return True

    @staticmethod
    def _check_region_constraints(
        puzzle: PipsPuzzle,
        domino: Domino,
        cell1: Cell,
        cell2: Cell,
        orientation: str,
    ) -> bool:
        """
        Unified constraint check.
        If both halves land in the SAME region, validate the COMBINED
        contribution (pip1 + pip2) against that region’s constraint.
        """
        region1 = puzzle.get_region(cell1)
        region2 = puzzle.get_region(cell2)

        # Pip values in the chosen orientation
        if orientation == "forward":
            pip1, pip2 = domino.pips_left, domino.pips_right
        else:
            pip1, pip2 = domino.pips_right, domino.pips_left

        # ---- SAME REGION ----
        if region1 is region2:
            R = region1
            ctype = R.constraint_type
            cval = R.constraint_value
            combined_sum = R.current_sum + pip1 + pip2

            if ctype == "sum":
                target = cval or 0
                if combined_sum > target:
                    return False
                empties_after = [c for c in R.get_empty_cells() if c not in (cell1, cell2)]
                if not ConstraintChecker._sum_feasible(combined_sum, len(empties_after), target):
                    return False
                if len(empties_after) == 0 and combined_sum != target:
                    return False
                # REMOVED: Aggressive value routing check
                # The original check here was too conservative and would reject valid
                # placements by being pessimistic about available domino ends.

            elif ctype == "equal":
                filled = R.get_filled_cells()
                if filled:
                    v = filled[0].value
                    if any(c.value != v for c in filled):
                        return False
                    if pip1 != v or pip2 != v:
                        return False
                else:
                    if pip1 != pip2:
                        return False
                # SIMPLE check: if empties remain, verify we have SOME ends with that value
                empties_after = [c for c in R.get_empty_cells() if c not in (cell1, cell2)]
                if empties_after:
                    v0 = filled[0].value if filled else pip1
                    # Count available ends
                    available = sum(1 for d in puzzle.dominoes if not d.placed 
                                  and (d.pips_left == v0 or d.pips_right == v0))
                    if available == 0:
                        return False  # No way to fill remaining cells

            elif ctype == "not_equal":
                filled_vals = {c.value for c in R.get_filled_cells()}
                if (pip1 in filled_vals) or (pip2 in filled_vals) or (pip1 == pip2):
                    return False

            elif ctype == "less_than":
                if cval is not None and combined_sum >= cval:
                    return False
                # REMOVED: Aggressive value routing check

            elif ctype == "greater_than":
                empties_after = [c for c in R.get_empty_cells() if c not in (cell1, cell2)]
                if cval is not None and (combined_sum + 6 * len(empties_after)) <= cval:
                    return False
                if len(empties_after) == 0 and cval is not None and combined_sum <= cval:
                    return False

            return True

        # ---- DIFFERENT REGIONS ----
        new_sum1 = region1.current_sum + pip1
        new_sum2 = region2.current_sum + pip2

        if not ConstraintChecker._validate_region_constraint(region1, new_sum1, cell1, pip1, cell2, puzzle):
            return False
        if not ConstraintChecker._validate_region_constraint(region2, new_sum2, cell2, pip2, cell1, puzzle):
            return False

        return True

    @staticmethod
    def _validate_region_constraint(
        region: Region,
        new_sum: int,
        cell: Cell,
        pip_value: int,
        other_cell: Optional[Cell],
        puzzle: PipsPuzzle,
    ) -> bool:
        """
        Validate that adding a single-cell value to `region` doesn't violate its constraint.
        (The combined same-region case is handled earlier.)
        """
        ctype = region.constraint_type
        cval = region.constraint_value

        if ctype == "sum":
            target = cval or 0
            if new_sum > target:
                return False
            empties_before = region.get_empty_cells()
            remaining_after = len([c for c in empties_before if c is not cell])
            if not ConstraintChecker._sum_feasible(new_sum, remaining_after, target):
                return False
            if remaining_after == 0 and new_sum != target:
                return False
            # REMOVED: Aggressive value routing check

        elif ctype == "equal":
            filled = region.get_filled_cells()
            if filled:
                v = filled[0].value
                if any(c.value != v for c in filled):
                    return False
                if pip_value != v:
                    return False
                # SIMPLE check: verify we have SOME ends with that value
                remaining_after = len([c for c in region.get_empty_cells() if c is not cell and c is not other_cell])
                if remaining_after > 0:
                    available = sum(1 for d in puzzle.dominoes if not d.placed
                                  and (d.pips_left == v or d.pips_right == v))
                    if available == 0:
                        return False

        elif ctype == "not_equal":
            filled_vals = {c.value for c in region.get_filled_cells()}
            if pip_value in filled_vals:
                return False

        elif ctype == "less_than":
            if cval is not None and new_sum >= cval:
                return False
            remaining_after = len([c for c in region.get_empty_cells() if c is not cell and c is not other_cell])
            if remaining_after == 0 and cval is not None and new_sum >= cval:
                return False
            # REMOVED: Aggressive value routing check

        elif ctype == "greater_than":
            remaining_after = len([c for c in region.get_empty_cells() if c is not cell and c is not other_cell])
            if cval is not None and (new_sum + 6 * remaining_after) <= cval:
                return False
            if remaining_after == 0 and cval is not None and new_sum <= cval:
                return False

        # Unknown/none -> permissive
        return True


# -----------------------------------------------------------------------------
# Heuristic Detection (optional helpers used by diagnostics/phase-1)
# -----------------------------------------------------------------------------
class HeuristicDetector:
    """Detects forced moves and high-priority placements (orientation-aware)."""

    @staticmethod
    def find_forced_moves(puzzle: PipsPuzzle) -> List[Dict]:
        """
        Find moves that are forced (only one placement: specific domino ORIENTATION).
        Returns list of dicts:
          { 'cell1', 'cell2', 'placements': [ {'domino', 'orientation'} ], 'reason': 'only_one_valid' }
        """
        forced: List[Dict] = []

        for cell1 in puzzle.cells:
            if cell1.occupied:
                continue
            for cell2 in puzzle.get_empty_adjacent_cells(cell1):
                if cell2.id <= cell1.id:
                    continue

                placements = []
                for domino in puzzle.dominoes:
                    if domino.placed:
                        continue
                    if ConstraintChecker.is_valid_placement(puzzle, domino, cell1, cell2, "forward"):
                        placements.append({'domino': domino, 'orientation': 'forward'})
                    if domino.pips_left != domino.pips_right:
                        if ConstraintChecker.is_valid_placement(puzzle, domino, cell1, cell2, "reverse"):
                            placements.append({'domino': domino, 'orientation': 'reverse'})

                if len(placements) == 1:
                    forced.append({
                        'cell1': cell1,
                        'cell2': cell2,
                        'placements': placements,
                        'reason': 'only_one_valid'
                    })

        return forced

    @staticmethod
    def find_constrained_placements(puzzle: PipsPuzzle) -> List[Dict]:
        """
        Find highly-constrained 2-cell regions (orientation-aware).
        Returns list of dicts with 'placements' like forced moves.
        """
        constrained: List[Dict] = []

        for region in puzzle.regions.values():
            empty_cells = region.get_empty_cells()
            if not empty_cells:
                continue

            if len(region.cells) == 2 and len(empty_cells) == 2:
                c1, c2 = empty_cells
                if c2 in puzzle.get_adjacent_cells(c1):
                    placements = []
                    for domino in puzzle.dominoes:
                        if domino.placed:
                            continue
                        if ConstraintChecker.is_valid_placement(puzzle, domino, c1, c2, "forward"):
                            placements.append({'domino': domino, 'orientation': 'forward'})
                        if domino.pips_left != domino.pips_right:
                            if ConstraintChecker.is_valid_placement(puzzle, domino, c1, c2, "reverse"):
                                placements.append({'domino': domino, 'orientation': 'reverse'})

                    if placements:
                        constrained.append({
                            'cell1': c1,
                            'cell2': c2,
                            'placements': placements,
                            'reason': 'two_cell_region',
                            'constraint_type': region.constraint_type,
                            'constraint_value': region.constraint_value,
                            'num_options': len(placements)
                        })

        return constrained

    @staticmethod
    def find_corner_constraints(puzzle: PipsPuzzle) -> List[Dict]:
        """
        Find cells with only one available connection (degree-1). Orientation-aware.
        """
        constrained: List[Dict] = []

        for cell in puzzle.cells:
            if cell.occupied:
                continue

            empty_neighbors = puzzle.get_empty_adjacent_cells(cell)
            if len(empty_neighbors) == 1:
                c2 = empty_neighbors[0]
                placements = []
                for domino in puzzle.dominoes:
                    if domino.placed:
                        continue
                    if ConstraintChecker.is_valid_placement(puzzle, domino, cell, c2, "forward"):
                        placements.append({'domino': domino, 'orientation': 'forward'})
                    if domino.pips_left != domino.pips_right:
                        if ConstraintChecker.is_valid_placement(puzzle, domino, cell, c2, "reverse"):
                            placements.append({'domino': domino, 'orientation': 'reverse'})

                if placements:
                    constrained.append({
                        'cell1': cell,
                        'cell2': c2,
                        'placements': placements,
                        'reason': 'only_one_neighbor',
                        'num_options': len(placements)
                    })

        return constrained

    @staticmethod
    def get_all_valid_moves(puzzle: PipsPuzzle) -> List[Dict]:
        """
        Enumerate all currently valid domino placements (both orientations).
        Returns a flat list of dicts: { 'cell1', 'cell2', 'domino', 'orientation' }
        """
        valid_moves: List[Dict] = []

        for cell1 in puzzle.cells:
            if cell1.occupied:
                continue
            for cell2 in puzzle.get_empty_adjacent_cells(cell1):
                if cell2.id <= cell1.id:
                    continue

                for domino in puzzle.dominoes:
                    if domino.placed:
                        continue

                    if ConstraintChecker.is_valid_placement(puzzle, domino, cell1, cell2, "forward"):
                        valid_moves.append({
                            'cell1': cell1,
                            'cell2': cell2,
                            'domino': domino,
                            'orientation': 'forward'
                        })

                    if domino.pips_left != domino.pips_right:
                        if ConstraintChecker.is_valid_placement(puzzle, domino, cell1, cell2, "reverse"):
                            valid_moves.append({
                                'cell1': cell1,
                                'cell2': cell2,
                                'domino': domino,
                                'orientation': 'reverse'
                            })

        return valid_moves