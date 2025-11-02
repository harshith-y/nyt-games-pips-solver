"""
Constraint checking and heuristic detection for Pips solver
Drop-in module with:
 - ConstraintChecker.is_valid_placement(...)
 - Mild look-ahead for SUM regions when exactly one cell would remain
 - Correct combined handling when both halves land in the SAME region
 - Orientation-aware checks for all region types
 - HeuristicDetector (optional utilities used by diagnostics / heuristics)

Compatible with:
  from puzzle import PipsPuzzle, Cell, Region, Domino
"""

from typing import List, Tuple, Optional, Dict
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
    def _would_create_impossible_constraint(
        puzzle: PipsPuzzle,
        domino: Domino,
        cell1: Cell,
        cell2: Cell,
        orientation: str,
    ) -> bool:
        """
        Lightweight look-ahead: when a SUM region would be left with exactly
        one cell after this placement, ensure at least one remaining domino
        half can supply the exact pip value needed for that last cell.

        NOTE:
          - This is a mild, SAFE early-out; it will not exclude legitimate moves
            when implemented as below.
          - It only applies to SUM regions.
        """
        # Pip values in the chosen orientation
        if orientation == "forward":
            pip1, pip2 = domino.pips_left, domino.pips_right
        else:
            pip1, pip2 = domino.pips_right, domino.pips_left

        region1 = puzzle.get_region(cell1)
        region2 = puzzle.get_region(cell2)

        # Region 1 check
        if not ConstraintChecker._check_region_lookahead(
            puzzle, placing_domino=domino, region=region1, cell=cell1, pip_value=pip1
        ):
            return True

        # Region 2 check (only if distinct)
        if region1 is not region2:
            if not ConstraintChecker._check_region_lookahead(
                puzzle, placing_domino=domino, region=region2, cell=cell2, pip_value=pip2
            ):
                return True

        return False

    @staticmethod
    def _check_region_lookahead(
        puzzle: PipsPuzzle,
        placing_domino: Domino,
        region: Region,
        cell: Cell,
        pip_value: int,
    ) -> bool:
        """
        For SUM regions only:
         - If placing `pip_value` here leaves exactly 1 empty cell in the region,
           check whether at least one remaining domino half can supply the
           precise value needed to hit the target.
        """
        if region.constraint_type != "sum":
            return True  # Only SUM regions get this look-ahead

        empties_before = region.get_empty_cells()
        # After placing into `cell`, the remaining empties are everyone except `cell`
        empties_after = [c for c in empties_before if c is not cell]

        if len(empties_after) == 1:
            new_sum = region.current_sum + pip_value
            needed_value = (region.constraint_value or 0) - new_sum

            # Needed value must be in [0..6] and achievable by at least one remaining half.
            if needed_value < 0 or needed_value > 6:
                return False

            available_values = set()
            for dom2 in puzzle.dominoes:
                if dom2.placed:
                    continue
                if dom2 is placing_domino:
                    # When the two halves of placing_domino land in DIFFERENT regions
                    # (the usual reason we're in this look-ahead),
                    # the placing_domino gives no additional half for this region.
                    # So we skip it here to avoid optimistic self-usage.
                    continue
                # Add both sides (set semantics are fine — we only need existence)
                available_values.add(dom2.pips_left)
                available_values.add(dom2.pips_right)

            if needed_value not in available_values:
                return False

        return True

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
        # 0) optional tiny symmetry break for "=" when region is fresh
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

        # 5) mild feasibility look-ahead on SUM regions
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
        Crucial: if both halves land in the SAME region, validate the COMBINED
        contribution (pip1 + pip2) against that region’s constraint.
        """
        region1 = puzzle.get_region(cell1)
        region2 = puzzle.get_region(cell2)

        # Pip values in the chosen orientation
        if orientation == "forward":
            pip1, pip2 = domino.pips_left, domino.pips_right
        else:
            pip1, pip2 = domino.pips_right, domino.pips_left

        # ---- Case A: BOTH HALVES in the SAME REGION ----
        if region1 is region2:
            R = region1
            ctype = R.constraint_type
            cval = R.constraint_value
            combined_sum = R.current_sum + pip1 + pip2

            if ctype == "sum":
                target = cval or 0
                # cannot overshoot
                if combined_sum > target:
                    return False
                # feasibility after placing both halves
                empties_before = R.get_empty_cells()
                empties_after = [c for c in empties_before if c not in (cell1, cell2)]
                if not ConstraintChecker._sum_feasible(combined_sum, len(empties_after), target):
                    return False
                # if this fills the region, must match exactly
                if len(empties_after) == 0 and combined_sum != target:
                    return False

            elif ctype == "equal":
                filled = R.get_filled_cells()
                if filled:
                    v = filled[0].value
                    if any(c.value != v for c in filled):
                        return False
                    if pip1 != v or pip2 != v:
                        return False
                else:
                    # Region is empty - both pips must be equal to maintain constraint
                    if pip1 != pip2:
                        return False

            elif ctype == "not_equal":
                # All cells distinct within region
                filled_vals = {c.value for c in R.get_filled_cells()}
                # both halves must be distinct from existing and from each other
                if (pip1 in filled_vals) or (pip2 in filled_vals) or (pip1 == pip2):
                    return False

            elif ctype == "less_than":
                if cval is not None and combined_sum >= cval:
                    return False

            elif ctype == "greater_than":
                # If combined_sum cannot possibly exceed, reject when it finishes region
                empties_before = R.get_empty_cells()
                empties_after = [c for c in empties_before if c not in (cell1, cell2)]
                if len(empties_after) == 0 and cval is not None and combined_sum <= cval:
                    return False
                # Feasibility guard
                if cval is not None and (combined_sum + 6 * len(empties_after)) <= cval:
                    return False

            # unknown/none -> permissive
            return True

        # ---- Case B: HALVES in DIFFERENT REGIONS ----
        new_sum1 = region1.current_sum + pip1
        new_sum2 = region2.current_sum + pip2

        if not ConstraintChecker._validate_region_constraint(region1, new_sum1, cell1, pip1):
            return False
        if not ConstraintChecker._validate_region_constraint(region2, new_sum2, cell2, pip2):
            return False

        return True

    @staticmethod
    def _validate_region_constraint(
        region: Region, new_sum: int, cell: Cell, pip_value: int
    ) -> bool:
        """
        Validate that adding a single-cell value to `region` doesn't violate its constraint.
        (The combined same-region case is handled earlier.)
        """
        ctype = region.constraint_type
        cval = region.constraint_value

        if ctype == "sum":
            target = cval or 0
            # cannot overshoot
            if new_sum > target:
                return False
            # feasibility guard
            empties_before = region.get_empty_cells()
            remaining_after = len([c for c in empties_before if c is not cell])
            if not ConstraintChecker._sum_feasible(new_sum, remaining_after, target):
                return False
            # if last cell, must match exactly
            if remaining_after == 0 and new_sum != target:
                return False

        elif ctype == "equal":
            filled = region.get_filled_cells()
            if filled:
                v = filled[0].value
                if any(c.value != v for c in filled):
                    return False
                if pip_value != v:
                    return False

        elif ctype == "not_equal":
            filled_vals = {c.value for c in region.get_filled_cells()}
            if pip_value in filled_vals:
                return False

        elif ctype == "less_than":
            if cval is not None and new_sum >= cval:
                return False
            empties_before = region.get_empty_cells()
            remaining_after = len([c for c in empties_before if c is not cell])
            if remaining_after == 0 and cval is not None and new_sum >= cval:
                return False

        elif ctype == "greater_than":
            empties_before = region.get_empty_cells()
            remaining_after = len([c for c in empties_before if c is not cell])
            if remaining_after == 0 and cval is not None and new_sum <= cval:
                return False
            if cval is not None and (new_sum + 6 * remaining_after) <= cval:
                return False

        else:
            # Unknown / no constraint -> permissive
            pass

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
                if cell2.id <= cell1.id:  # avoid duplicates
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
                if cell2.id <= cell1.id:  # avoid duplicates
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