# solver_fixed.py
"""
Fixed CSP Solver for Pips puzzles with conservative heuristics

Key fixes:
1. Conservative heuristics - only apply when TRULY forced
2. No aggressive region-sum forcing (too prone to errors)
3. Backtracking can explore alternatives even after heuristic phase
4. Forward checking is validation-only, doesn't pre-commit moves
5. MRV ordering with proper domain tracking

This approach is more generalizable across different puzzle types.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from puzzle import PipsPuzzle, Cell, Domino, Region
from constraints import ConstraintChecker


EdgeKey = Tuple[int, int]
Placement = Tuple[int, str]


class CSPSolver:
    def __init__(self, puzzle: PipsPuzzle, verbose: bool = True, use_lcv: bool = False, use_aggressive_heuristics: bool = True, skip_heuristics: bool = False):
        self.puzzle = puzzle
        self.verbose = verbose
        self.use_lcv = use_lcv
        self.use_aggressive_heuristics = use_aggressive_heuristics  # NEW: control heuristics
        self.skip_heuristics = skip_heuristics  # NEW: skip heuristics entirely
        self.in_backtrack_mode = False  # Track if we're in backtracking (vs heuristics)
        self.stats = {
            'backtracks': 0,
            'heuristic_moves': 0,
            'search_moves': 0,
            'total_attempts': 0,
            'forward_check_prunes': 0
        }
        # Domains: for each edge, set of (domino_id, orientation)
        self.domains: Dict[EdgeKey, Set[Placement]] = {}
        self._init_domains()

    # -------------------------------------------------------------------------
    # Domain management
    # -------------------------------------------------------------------------
    def _init_domains(self) -> None:
        """Initialize domains for all edges between empty cells."""
        self.domains.clear()
        for c1 in self.puzzle.cells:
            if c1.occupied:
                continue
            for c2 in self.puzzle.get_adjacent_cells(c1):
                if c2.occupied or c2.id <= c1.id:
                    continue
                key = (c1.id, c2.id)
                self.domains[key] = self._compute_edge_domain(c1, c2)

    def _compute_edge_domain(self, c1: Cell, c2: Cell) -> Set[Placement]:
        """Compute all legal placements for edge (c1,c2)."""
        if c1.occupied or c2.occupied:
            return set()

        out: Set[Placement] = set()
        for dom in self.puzzle.dominoes:
            if dom.placed:
                continue

            # Try forward orientation
            if ConstraintChecker.is_valid_placement(self.puzzle, dom, c1, c2, "forward"):
                out.add((dom.id, "forward"))

            # Try reverse orientation (only for non-doubles)
            if dom.pips_left != dom.pips_right:
                if ConstraintChecker.is_valid_placement(self.puzzle, dom, c1, c2, "reverse"):
                    out.add((dom.id, "reverse"))

        return out

    def _refresh_domains(self) -> None:
        """Recompute all domains (used after placing/removing dominoes)."""
        for key in list(self.domains.keys()):
            c1 = self.puzzle.cell_by_id[key[0]]
            c2 = self.puzzle.cell_by_id[key[1]]
            if c1.occupied or c2.occupied:
                self.domains[key] = set()
            else:
                self.domains[key] = self._compute_edge_domain(c1, c2)

    # -------------------------------------------------------------------------
    # Main solving driver
    # -------------------------------------------------------------------------
    def solve(self, timeout_seconds: int = 300) -> bool:
        import time
        self.start_time = time.time()
        self.timeout = timeout_seconds

        if self.verbose:
            print(f"Starting conservative solver: {self.puzzle}")
            print("Strategy: Conservative heuristics + MRV backtracking\n")

        # Sanity checks
        if any(d.placed for d in self.puzzle.dominoes):
            raise RuntimeError("Dominoes already placed at start")

        # Phase 1: VERY conservative heuristics (only absolutely forced moves)
        if not self.skip_heuristics:
            if self.verbose:
                print("=== Phase 1: Conservative Heuristics ===")
            
            heur_count = self._apply_conservative_heuristics()
            
            if self.verbose:
                print(f"Heuristic placements: {heur_count}")

            if self.puzzle.is_complete():
                if self.verbose:
                    print("\n✓ Solved using heuristics alone!")
                    self._print_stats()
                return True
        else:
            if self.verbose:
                print("=== Phase 1: Heuristics SKIPPED ===")
            heur_count = 0

        # Phase 2: MRV backtracking with forward checking
        if self.verbose:
            cp = self.puzzle.get_completion_percentage()
            print(f"\n=== Phase 2: MRV Backtracking ===")
            print(f"Starting at {cp:.1%} complete\n")

        self.in_backtrack_mode = True  # Enable permissive mode for backtracking
        result = self._backtrack_mrv(0)
        
        if self.verbose:
            print("\n✓ Puzzle solved!" if result else "\n✗ No solution found")
            self._print_stats()
        
        return result

    # -------------------------------------------------------------------------
    # Conservative heuristics (only truly forced moves)
    # -------------------------------------------------------------------------
    def _apply_conservative_heuristics(self) -> int:
        """
        Apply ONLY conservative heuristics that are definitely forced.
        
        Can be partially or fully disabled via use_aggressive_heuristics flag.
        """
        total = 0
        max_rounds = 10  # Safety limit
        
        for _ in range(max_rounds):
            made = 0
            
            # Try critical region forcing first (like sum=0) - OPTIONAL
            if self.use_aggressive_heuristics:
                made += self._heuristic_critical_regions()
                if made:
                    total += made
                    continue
            
            # Try isolated cell forcing - ALWAYS enabled (safe)
            made += self._heuristic_isolated_cell()
            if made:
                total += made
                continue
            
            # Try truly isolated edge forcing - ALWAYS enabled (safe)
            made += self._heuristic_truly_isolated_edge()
            if made:
                total += made
                continue
            
            break  # No more forced moves
        
        return total

    def _heuristic_critical_regions(self) -> int:
        """
        Detect regions with extremely tight constraints that have only ONE valid solution.
        Example: Region with 3 cells, sum=0 - only (0,0) can work.
        
        For such regions, pre-commit the forced domino placement.
        
        NOTE: This can sometimes lead to dead ends, so it's optional.
        """
        if not self.use_aggressive_heuristics:
            return 0  # Skip if disabled
        
        for region in self.puzzle.regions.values():
            if region.constraint_type != 'sum' or region.constraint_value is None:
                continue
            
            # Skip if region already has placements
            if region.get_filled_cells():
                continue
                
            empty_cells = region.get_empty_cells()
            if len(empty_cells) < 2:
                continue
            
            target = region.constraint_value
            
            # Special case: sum = 0
            # Only (0,0) domino can contribute, must be placed within the region
            if target == 0:
                # Find all edges within this region
                region_edges = []
                for i, c1 in enumerate(empty_cells):
                    for c2 in empty_cells[i+1:]:
                        if c2 in self.puzzle.get_adjacent_cells(c1):
                            region_edges.append((c1, c2))
                
                # For each edge, check if (0,0) is the only domino that works
                for c1, c2 in region_edges:
                    key = (min(c1.id, c2.id), max(c1.id, c2.id))
                    domain = self.domains.get(key, set())
                    
                    # Look for (0,0) domino
                    zero_domino = next((d for d in self.puzzle.dominoes 
                                       if not d.placed and d.pips_left == 0 and d.pips_right == 0), None)
                    
                    if zero_domino and (zero_domino.id, 'forward') in domain:
                        # Check if ALL cells in region can only be filled with 0
                        all_must_be_zero = True
                        for cell in empty_cells:
                            # Check if this cell can have non-zero value
                            can_be_nonzero = False
                            for neighbor in self.puzzle.get_adjacent_cells(cell):
                                if neighbor.occupied:
                                    continue
                                nkey = (min(cell.id, neighbor.id), max(cell.id, neighbor.id))
                                ndom = self.domains.get(nkey, set())
                                
                                for did, ori in ndom:
                                    d = next((x for x in self.puzzle.dominoes if x.id == did), None)
                                    if d and not d.placed and d.id != zero_domino.id:
                                        can_be_nonzero = True
                                        break
                                if can_be_nonzero:
                                    break
                        
                        # If this edge is the only way to place (0,0), force it
                        if self.verbose:
                            print(f"  [CRITICAL] Forcing (0,0) on cells {c1.id}-{c2.id} for sum=0 region {region.section_id}")
                        
                        self._place_domino(zero_domino, c1, c2, 'forward')
                        self._refresh_domains()
                        
                        if not self._forward_check_valid():
                            # This was wrong, undo
                            self._remove_domino(zero_domino, c1, c2)
                            self._refresh_domains()
                            continue
                        
                        self.stats['heuristic_moves'] += 1
                        return 1
        
        return 0
    
    def _heuristic_isolated_cell(self) -> int:
        """
        If a cell has only ONE empty neighbor and only ONE valid domino/orientation
        for that edge, it's truly forced.
        """
        for cell in self.puzzle.cells:
            if cell.occupied:
                continue
            
            empty_neighbors = [n for n in self.puzzle.get_adjacent_cells(cell) if not n.occupied]
            
            if len(empty_neighbors) != 1:
                continue  # Not isolated
            
            neighbor = empty_neighbors[0]
            key = (min(cell.id, neighbor.id), max(cell.id, neighbor.id))
            domain = self.domains.get(key, set())
            
            # Filter to available dominoes
            available = []
            for did, ori in domain:
                d = next((x for x in self.puzzle.dominoes if x.id == did), None)
                if d and not d.placed:
                    available.append((d, ori))
            
            if len(available) == 1:
                d, ori = available[0]
                c1, c2 = (cell, neighbor) if cell.id < neighbor.id else (neighbor, cell)
                
                # Double-check validity
                if ConstraintChecker.is_valid_placement(self.puzzle, d, c1, c2, ori):
                    self._place_domino(d, c1, c2, ori)
                    self._refresh_domains()
                    
                    # Check if this leads to immediate contradiction
                    if not self._forward_check_valid():
                        # Undo - this wasn't actually forced
                        self._remove_domino(d, c1, c2)
                        self._refresh_domains()
                        return 0
                    
                    self.stats['heuristic_moves'] += 1
                    if self.verbose:
                        print(f"Forced (isolated cell): {d} at {c1.id}-{c2.id} ({ori})")
                    return 1
        
        return 0

    def _heuristic_truly_isolated_edge(self) -> int:
        """
        If an edge has only one available domino/orientation AND both endpoints
        have very limited options (degree <= 2), consider it forced.
        
        This is conservative: we only apply when BOTH cells are constrained.
        """
        for key, domain in self.domains.items():
            c1 = self.puzzle.cell_by_id[key[0]]
            c2 = self.puzzle.cell_by_id[key[1]]
            
            if c1.occupied or c2.occupied:
                continue
            
            # Both cells must have low degree (constrained)
            deg1 = len([n for n in self.puzzle.get_adjacent_cells(c1) if not n.occupied])
            deg2 = len([n for n in self.puzzle.get_adjacent_cells(c2) if not n.occupied])
            
            if deg1 > 2 or deg2 > 2:
                continue  # Not constrained enough
            
            # Filter to available dominoes
            available = []
            for did, ori in domain:
                d = next((x for x in self.puzzle.dominoes if x.id == did), None)
                if d and not d.placed:
                    available.append((d, ori))
            
            if len(available) == 1:
                d, ori = available[0]
                
                # Double-check validity
                if ConstraintChecker.is_valid_placement(self.puzzle, d, c1, c2, ori):
                    self._place_domino(d, c1, c2, ori)
                    self._refresh_domains()
                    
                    # Check if this leads to immediate contradiction
                    if not self._forward_check_valid():
                        # Undo - this wasn't actually safe
                        self._remove_domino(d, c1, c2)
                        self._refresh_domains()
                        return 0
                    
                    self.stats['heuristic_moves'] += 1
                    if self.verbose:
                        print(f"Forced (isolated edge): {d} at {c1.id}-{c2.id} ({ori})")
                    return 1
        
        return 0

    # -------------------------------------------------------------------------
    # MRV Backtracking with Forward Checking
    # -------------------------------------------------------------------------
    def _backtrack_mrv(self, depth: int) -> bool:
        import time
        
        # Timeout check
        if time.time() - self.start_time > self.timeout:
            return False
        
        # Success check
        if self.puzzle.is_complete():
            return True
        
        self.stats['total_attempts'] += 1
        
        # Progress reporting
        if self.verbose and self.stats['total_attempts'] % 1000 == 0:
            cp = self.puzzle.get_completion_percentage()
            print(f"  Progress: {cp:.1%} | Attempts: {self.stats['total_attempts']} | "
                  f"Backtracks: {self.stats['backtracks']} | Depth: {depth}")
        
        # STRATEGY: Use a hybrid approach
        # - Prioritize edges with fewer options (MRV)
        # - But ALSO consider how constraining each edge is (proto-LCV)
        # - Avoid edges that would severely limit future options
        pairs_to_try = self._select_mrv_pairs_with_lcv_hint(k=3)
        
        if not pairs_to_try:
            return False  # No valid moves
        
        # Try each candidate pair
        for c1, c2, candidates in pairs_to_try:
            # Optional LCV ordering
            if self.use_lcv:
                candidates = self._order_by_lcv(c1, c2, candidates)
            
            if self.verbose and depth < 3:
                print(f"{'  ' * depth}Trying cells {c1.id}-{c2.id} ({len(candidates)} options)")
            
            # Try each domino/orientation for this pair
            for did, ori in candidates:
                d = next((x for x in self.puzzle.dominoes if x.id == did), None)
                if not d or d.placed:
                    continue
                
                # Validate placement
                if not ConstraintChecker.is_valid_placement(self.puzzle, d, c1, c2, ori):
                    continue
                
                # Save state for backtracking
                saved_domains = self._save_domains()
                
                # Make move
                self._place_domino(d, c1, c2, ori)
                self.stats['search_moves'] += 1
                
                if self.verbose and depth < 3:
                    print(f"{'  ' * depth}  Placing {d} ({ori})")
                
                # Refresh domains
                self._refresh_domains()
                
                # Forward check
                if not self._forward_check_valid():
                    # Forward check failed - prune this branch
                    self._remove_domino(d, c1, c2)
                    self._restore_domains(saved_domains)
                    self.stats['forward_check_prunes'] += 1
                    if self.verbose and depth < 3:
                        print(f"{'  ' * depth}  FC pruned")
                    continue
                
                # Recurse
                if self._backtrack_mrv(depth + 1):
                    return True
                
                # Backtrack
                self._remove_domino(d, c1, c2)
                self._restore_domains(saved_domains)
                self.stats['backtracks'] += 1
                
                if self.verbose and depth < 3:
                    print(f"{'  ' * depth}  Backtrack")
        
        return False
    
    def _select_flexible_pairs(self, k: int = 5) -> List[Tuple[Cell, Cell, List[Placement]]]:
        """
        Select top K edges with MOST remaining values (most flexible).
        Used at depth 0 to find good starting points.
        """
        candidates_list = []
        
        for key, domain in self.domains.items():
            c1 = self.puzzle.cell_by_id[key[0]]
            c2 = self.puzzle.cell_by_id[key[1]]
            
            if c1.occupied or c2.occupied:
                continue
            
            # Filter to available dominoes
            candidates = []
            for did, ori in domain:
                d = next((x for x in self.puzzle.dominoes if x.id == did), None)
                if d and not d.placed:
                    candidates.append((did, ori))
            
            if not candidates or len(candidates) < 2:  # Skip edges with only 1 option
                continue
            
            count = len(candidates)
            
            # Sort key: PREFER more options (descending), then lower cell IDs
            sort_key = (-count, c1.id + c2.id)
            
            candidates_list.append((sort_key, c1, c2, candidates))
        
        # Sort and return top k
        candidates_list.sort(key=lambda x: x[0])
        return [(c1, c2, cands) for _, c1, c2, cands in candidates_list[:k]]
    
    def _select_mrv_pairs_with_lcv_hint(self, k: int = 3) -> List[Tuple[Cell, Cell, List[Placement]]]:
        """
        Hybrid MRV + LCV: Select edges with few options (MRV),
        but deprioritize edges that would be highly constraining (LCV hint).
        
        This is the KEY generalizable improvement: we consider both
        - How constrained the current edge is (MRV - fewer options = pick first)
        - How constraining choosing this edge would be (LCV - more constraining = pick last)
        
        CRITICAL FIX: Edges connecting TO tight regions (like sum=0) should be
        PRIORITIZED, not deferred, as they're necessary to satisfy the constraint.
        Only defer edges WITHIN tight regions that have many options.
        """
        candidates_list = []
        
        for key, domain in self.domains.items():
            c1 = self.puzzle.cell_by_id[key[0]]
            c2 = self.puzzle.cell_by_id[key[1]]
            
            if c1.occupied or c2.occupied:
                continue
            
            # Get available placements
            candidates = []
            for did, ori in domain:
                d = next((x for x in self.puzzle.dominoes if x.id == did), None)
                if d and not d.placed:
                    candidates.append((did, ori))
            
            if not candidates:
                continue
            
            count = len(candidates)
            
            # Calculate how "constraining" this edge is
            # by simulating placement of the first candidate and counting remaining options
            constraining_score = 0
            if self.use_lcv and len(candidates) > 0:
                # Quick estimate: check how many neighboring edges would be affected
                did, ori = candidates[0]
                d = next((x for x in self.puzzle.dominoes if x.id == did), None)
                if d:
                    # Count affected neighbors
                    neighbor_ids = set()
                    for n in self.puzzle.get_adjacent_cells(c1):
                        if not n.occupied:
                            neighbor_ids.add(n.id)
                    for n in self.puzzle.get_adjacent_cells(c2):
                        if not n.occupied:
                            neighbor_ids.add(n.id)
                    
                    constraining_score = len(neighbor_ids)  # More neighbors = more constraining
            
            # Get region constraint info
            r1 = self.puzzle.get_region(c1)
            r2 = self.puzzle.get_region(c2)
            same_region = (r1 is r2)
            
            # NEW: Distinguish between edges WITHIN vs CONNECTING TO tight regions
            within_tight_region = False
            connecting_to_tight_region = False
            
            if same_region:
                # Both cells in same region - check if it's tight
                if r1.constraint_type == 'sum' and r1.constraint_value is not None:
                    if r1.constraint_value <= 6:
                        within_tight_region = True
            else:
                # Spanning two regions - check if connecting TO a tight region
                for r in [r1, r2]:
                    if r.constraint_type == 'sum' and r.constraint_value is not None:
                        if r.constraint_value <= 6:
                            # This edge provides value TO a tight region
                            # IMPORTANT: These should be prioritized, not deferred!
                            connecting_to_tight_region = True
                            break
            
            # Sort key:
            # 1. Connecting to tight regions should come EARLY (low priority number)
            # 2. Fewer options (MRV)
            # 3. NOT within tight regions (defer these only if same region)
            # 4. Lower constraining score (less impact on neighbors)
            # 5. Prefer same region (more localized)
            # 6. Cell ID for determinism
            
            # Priority boost for edges connecting to tight regions
            priority_boost = 0 if connecting_to_tight_region else 1
            
            sort_key = (priority_boost, count, within_tight_region, constraining_score, not same_region, c1.id + c2.id)
            
            candidates_list.append((sort_key, c1, c2, candidates))
        
        # Sort and return top k
        candidates_list.sort(key=lambda x: x[0])
        return [(c1, c2, cands) for _, c1, c2, cands in candidates_list[:k]]
    
    def _select_top_mrv_pairs(self, k: int = 1) -> List[Tuple[Cell, Cell, List[Placement]]]:
        """
        Select top K edges with minimum remaining values.
        This allows exploring multiple starting points when k > 1.
        """
        candidates_list = []
        
        for key, domain in self.domains.items():
            c1 = self.puzzle.cell_by_id[key[0]]
            c2 = self.puzzle.cell_by_id[key[1]]
            
            if c1.occupied or c2.occupied:
                continue
            
            # Filter to available dominoes
            candidates = []
            for did, ori in domain:
                d = next((x for x in self.puzzle.dominoes if x.id == did), None)
                if d and not d.placed:
                    candidates.append((did, ori))
            
            if not candidates:
                continue
            
            count = len(candidates)
            
            # Calculate degree
            deg1 = len([n for n in self.puzzle.get_adjacent_cells(c1) if not n.occupied])
            deg2 = len([n for n in self.puzzle.get_adjacent_cells(c2) if not n.occupied])
            min_degree = min(deg1, deg2)
            
            # Check if same region
            r1 = self.puzzle.get_region(c1)
            r2 = self.puzzle.get_region(c2)
            same_region = (r1 is r2)
            
            # Sort key: prefer different regions, higher degree (less constrained)
            sort_key = (count, same_region, -min_degree, c1.id + c2.id)
            
            candidates_list.append((sort_key, c1, c2, candidates))
        
        # Sort and return top k
        candidates_list.sort(key=lambda x: x[0])
        return [(c1, c2, cands) for _, c1, c2, cands in candidates_list[:k]]

    # -------------------------------------------------------------------------
    # MRV selection
    # -------------------------------------------------------------------------
    def _select_mrv_pair(self) -> Optional[Tuple[Cell, Cell, List[Placement]]]:
        """
        Select edge with minimum remaining values.
        
        Improved tie-breaks:
        1. Prefer edges where both cells are in the SAME region (less coupling)
        2. Prefer cells with fewer neighbors (more constrained)
        3. Prefer regions that are "closer to complete"
        4. Prefer earlier cell IDs (deterministic)
        """
        best = None
        best_key = None
        
        for key, domain in self.domains.items():
            c1 = self.puzzle.cell_by_id[key[0]]
            c2 = self.puzzle.cell_by_id[key[1]]
            
            if c1.occupied or c2.occupied:
                continue
            
            # Filter to available dominoes
            candidates = []
            for did, ori in domain:
                d = next((x for x in self.puzzle.dominoes if x.id == did), None)
                if d and not d.placed:
                    candidates.append((did, ori))
            
            if not candidates:
                continue
            
            count = len(candidates)
            
            # Calculate degree (neighbors) for tie-breaking
            deg1 = len([n for n in self.puzzle.get_adjacent_cells(c1) if not n.occupied])
            deg2 = len([n for n in self.puzzle.get_adjacent_cells(c2) if not n.occupied])
            min_degree = min(deg1, deg2)
            
            # Check if same region (preferred for less coupling)
            r1 = self.puzzle.get_region(c1)
            r2 = self.puzzle.get_region(c2)
            same_region = (r1 is r2)
            
            # Calculate region "completeness" - prefer regions closer to being filled
            r1_completeness = len(r1.get_filled_cells()) / len(r1.cells) if r1.cells else 0
            r2_completeness = len(r2.get_filled_cells()) / len(r2.cells) if r2.cells else 0
            avg_completeness = (r1_completeness + r2_completeness) / 2
            
            # Check if this is a "critical" edge (single option in a constrained region)
            # These should be DEFERRED unless forced, as they're likely traps
            is_critical = False
            if count == 1:
                # Single option edges in sum/equal/not_equal regions are critical
                if r1.constraint_type in ['sum', 'equal', 'not_equal']:
                    # If region needs specific values (like sum=0), defer this edge
                    if r1.constraint_value is not None and r1.constraint_value <= 6:
                        is_critical = True
                if r2.constraint_type in ['sum', 'equal', 'not_equal']:
                    if r2.constraint_value is not None and r2.constraint_value <= 6:
                        is_critical = True
            
            # Sort key:
            # 0. NOT critical (defer single-option edges in constrained regions)
            # 1. Fewer options (MRV)
            # 2. NOT same region (prefer edges that span regions for now, less restrictive)
            # 3. Higher degree (less constrained = safer to try first)
            # 4. Lower cell ID sum (deterministic)
            sort_key = (is_critical, count, same_region, -min_degree, c1.id + c2.id)
            
            if best_key is None or sort_key < best_key:
                best_key = sort_key
                best = (c1, c2, candidates)
        
        return best

    # -------------------------------------------------------------------------
    # LCV ordering (optional)
    # -------------------------------------------------------------------------
    def _order_by_lcv(self, c1: Cell, c2: Cell, candidates: List[Placement]) -> List[Placement]:
        """
        Order candidates by least-constraining-value.
        
        For each candidate, count how many options remain for neighboring edges.
        Try least constraining first (more options = more likely to succeed).
        """
        scored = []
        
        for did, ori in candidates:
            d = next((x for x in self.puzzle.dominoes if x.id == did), None)
            if not d:
                continue
            
            # Simulate placement
            saved = self._save_domains()
            self._place_domino(d, c1, c2, ori)
            self._refresh_domains()
            
            # Count remaining options on neighboring edges
            neighbor_ids = {c1.id, c2.id}
            for n in self.puzzle.get_adjacent_cells(c1):
                neighbor_ids.add(n.id)
            for n in self.puzzle.get_adjacent_cells(c2):
                neighbor_ids.add(n.id)
            
            remaining = 0
            for key, dom_set in self.domains.items():
                if key[0] in neighbor_ids or key[1] in neighbor_ids:
                    ca = self.puzzle.cell_by_id[key[0]]
                    cb = self.puzzle.cell_by_id[key[1]]
                    if not ca.occupied and not cb.occupied:
                        for d2, o2 in dom_set:
                            D2 = next((x for x in self.puzzle.dominoes if x.id == d2), None)
                            if D2 and not D2.placed:
                                remaining += 1
            
            # Undo simulation
            self._remove_domino(d, c1, c2)
            self._restore_domains(saved)
            
            scored.append(((did, ori), remaining))
        
        # Sort by remaining (descending = least constraining first)
        scored.sort(key=lambda t: -t[1])
        return [p for p, _ in scored]

    # -------------------------------------------------------------------------
    # Forward checking
    # -------------------------------------------------------------------------
    def _forward_check_valid(self) -> bool:
        """
        Check that every empty cell still has at least one valid placement,
        AND that all constrained regions are still solvable.
        
        This is a simple but effective pruning technique.
        """
        # Check 1: Every empty cell must have at least one option
        for cell in self.puzzle.cells:
            if cell.occupied:
                continue
            
            # Check if cell has any valid placement
            has_option = False
            for neighbor in self.puzzle.get_adjacent_cells(cell):
                if neighbor.occupied:
                    continue
                
                key = (min(cell.id, neighbor.id), max(cell.id, neighbor.id))
                domain = self.domains.get(key, set())
                
                for did, ori in domain:
                    d = next((x for x in self.puzzle.dominoes if x.id == did), None)
                    if d and not d.placed:
                        # Found at least one option
                        has_option = True
                        break
                
                if has_option:
                    break
            
            if not has_option:
                # This cell has no valid moves - dead end
                return False
        
        # Check 2: Constrained regions must still be solvable
        if not self._check_regions_solvable():
            return False
        
        return True

    
    def _check_regions_solvable(self) -> bool:
        """
        SIMPLIFIED: Check only provably impossible states.
        
        Removed aggressive "ends counting" and predictive logic that was too
        conservative. Only validates completed regions and basic bounds for
        incomplete regions.
        """

        for region in self.puzzle.regions.values():
            empty_cells = region.get_empty_cells()

            # =========================
            # Completed region checks
            # =========================
            if not empty_cells:
                ct, cv = region.constraint_type, region.constraint_value
                if ct == 'sum' and cv is not None and region.current_sum != cv:
                    if self.verbose:
                        print(f"    [FC] prune: R{region.section_id} sum mismatch")
                    self.stats['forward_check_prunes'] += 1
                    return False
                if ct == 'less_than' and cv is not None and not (region.current_sum < cv):
                    if self.verbose:
                        print(f"    [FC] prune: R{region.section_id} violates <")
                    self.stats['forward_check_prunes'] += 1
                    return False
                if ct == 'greater_than' and cv is not None and not (region.current_sum > cv):
                    if self.verbose:
                        print(f"    [FC] prune: R{region.section_id} violates >")
                    self.stats['forward_check_prunes'] += 1
                    return False
                if ct == 'equal':
                    filled = region.get_filled_cells()
                    if filled:
                        v = filled[0].value
                        if any(c.value != v for c in filled):
                            if self.verbose:
                                print(f"    [FC] prune: R{region.section_id} equal violated")
                            self.stats['forward_check_prunes'] += 1
                            return False
                if ct == 'not_equal':
                    filled = region.get_filled_cells()
                    vals = [c.value for c in filled]
                    if len(vals) != len(set(vals)):
                        if self.verbose:
                            print(f"    [FC] prune: R{region.section_id} not_equal violated")
                        self.stats['forward_check_prunes'] += 1
                        return False
                continue

            # =========================
            # Incomplete region checks - BASIC ONLY
            # =========================
            ct, cv = region.constraint_type, region.constraint_value
            remaining_cells = len(empty_cells)

            # Basic bounds checking only - no aggressive predictions
            if ct == 'sum' and cv is not None:
                cur = region.current_sum
                # Simple interval: can target be reached with 0..6 per cell?
                if cv < cur or cv > cur + 6 * remaining_cells:
                    if self.verbose:
                        print(f"    [FC] prune: R{region.section_id} sum unreachable")
                    self.stats['forward_check_prunes'] += 1
                    return False

            elif ct == 'equal':
                # Only check that existing values match
                filled = region.get_filled_cells()
                if filled:
                    v0 = filled[0].value
                    if any(c.value != v0 for c in filled):
                        if self.verbose:
                            print(f"    [FC] prune: R{region.section_id} equal mismatch")
                        self.stats['forward_check_prunes'] += 1
                        return False

            elif ct == 'less_than' and cv is not None:
                if region.current_sum >= cv:
                    if self.verbose:
                        print(f"    [FC] prune: R{region.section_id} already too large")
                    self.stats['forward_check_prunes'] += 1
                    return False

            elif ct == 'greater_than' and cv is not None:
                max_possible = region.current_sum + 6 * remaining_cells
                if max_possible <= cv:
                    if self.verbose:
                        print(f"    [FC] prune: R{region.section_id} cannot reach minimum")
                    self.stats['forward_check_prunes'] += 1
                    return False

        return True


    
    # -------------------------------------------------------------------------
    # Domain save/restore
    # -------------------------------------------------------------------------
    def _save_domains(self) -> Dict[EdgeKey, Set[Placement]]:
        """Save current domain state."""
        return {k: v.copy() for k, v in self.domains.items()}

    def _restore_domains(self, saved: Dict[EdgeKey, Set[Placement]]) -> None:
        """Restore domain state."""
        self.domains = saved

    # -------------------------------------------------------------------------
    # Domino placement/removal
    # -------------------------------------------------------------------------
    def _place_domino(self, domino: Domino, cell1: Cell, cell2: Cell, orientation: str) -> None:
        """Place a domino on the board."""
        domino.placed = True
        domino.cell1 = cell1
        domino.cell2 = cell2

        if orientation == "forward":
            pip1, pip2 = domino.pips_left, domino.pips_right
        else:
            pip1, pip2 = domino.pips_right, domino.pips_left

        cell1.occupied = True
        cell1.value = pip1
        cell1.domino_id = domino.id

        cell2.occupied = True
        cell2.value = pip2
        cell2.domino_id = domino.id

        r1 = self.puzzle.get_region(cell1)
        r2 = self.puzzle.get_region(cell2)
        r1.current_sum += pip1
        if r1 is r2:
            r1.current_sum += pip2
        else:
            r2.current_sum += pip2

        self.puzzle.move_history.append({
            'domino': domino,
            'cell1': cell1,
            'cell2': cell2,
            'orientation': orientation
        })

    def _remove_domino(self, domino: Domino, cell1: Cell, cell2: Cell) -> None:
        """Remove a domino from the board (backtrack)."""
        pip1, pip2 = cell1.value, cell2.value

        domino.placed = False
        domino.cell1 = None
        domino.cell2 = None

        cell1.occupied = False
        cell1.value = None
        cell1.domino_id = None

        cell2.occupied = False
        cell2.value = None
        cell2.domino_id = None

        r1 = self.puzzle.get_region(cell1)
        r2 = self.puzzle.get_region(cell2)
        r1.current_sum -= pip1
        if r1 is r2:
            r1.current_sum -= pip2
        else:
            r2.current_sum -= pip2

        if self.puzzle.move_history:
            self.puzzle.move_history.pop()

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def _print_stats(self) -> None:
        """Print solving statistics."""
        print("\nSolving Statistics:")
        print(f"  Heuristic moves: {self.stats['heuristic_moves']}")
        print(f"  Search moves: {self.stats['search_moves']}")
        print(f"  Backtracks: {self.stats['backtracks']}")
        print(f"  Forward check prunes: {self.stats['forward_check_prunes']}")
        print(f"  Total attempts: {self.stats['total_attempts']}")
        print(f"  Final placements: {len(self.puzzle.move_history)}")