"""
Core data structures for Pips puzzle representation with multi-board support
"""
import json
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Cell:
    """Represents a single cell in the puzzle grid"""
    id: int
    row: int
    col: int
    section: int  # Which region this cell belongs to
    bbox: List[int]
    center: List[int]
    board_index: int = 0  # Which board this cell belongs to (for multi-board puzzles)
    occupied: bool = False
    value: Optional[int] = None  # Pip value if occupied
    domino_id: Optional[int] = None  # Which domino occupies this cell
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Cell) and self.id == other.id


@dataclass
class Region:
    """Represents a region (section) with a constraint"""
    section_id: int
    cells: List[Cell] = field(default_factory=list)
    constraint_type: str = ""  # '=', '≠', '<', '>', 'sum', etc.
    constraint_value: Optional[int] = None
    current_sum: int = 0
    
    def add_cell(self, cell: Cell):
        """Add a cell to this region"""
        self.cells.append(cell)
    
    def get_empty_cells(self) -> List[Cell]:
        """Get all unoccupied cells in this region"""
        return [c for c in self.cells if not c.occupied]
    
    def get_filled_cells(self) -> List[Cell]:
        """Get all occupied cells in this region"""
        return [c for c in self.cells if c.occupied]
    
    def is_complete(self) -> bool:
        """Check if all cells in region are filled"""
        return all(c.occupied for c in self.cells)
    
    def __repr__(self):
        return f"Region(id={self.section_id}, size={len(self.cells)}, constraint={self.constraint_type}{self.constraint_value}, sum={self.current_sum})"


@dataclass
class Domino:
    """Represents a domino piece"""
    id: int
    pips_left: int
    pips_right: int
    bbox: List[int]
    placed: bool = False
    cell1: Optional[Cell] = None
    cell2: Optional[Cell] = None
    
    def as_tuple(self) -> Tuple[int, int]:
        """Return domino as (left, right) tuple"""
        return (self.pips_left, self.pips_right)
    
    def sum(self) -> int:
        """Total pips on this domino"""
        return self.pips_left + self.pips_right
    
    def is_equal(self) -> bool:
        """Check if both sides have equal pips"""
        return self.pips_left == self.pips_right
    
    def __repr__(self):
        return f"Domino({self.pips_left},{self.pips_right})"


class PipsPuzzle:
    """Main puzzle class with multi-board support"""
    
    def __init__(self, json_path: str):
        """Load puzzle from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.data = data
        all_boards = data['boards']
        
        # Filter to only valid boards (board_index > 0)
        self.boards_data = [b for b in all_boards if b.get('board_index', 0) > 0]
        self.num_boards = len(self.boards_data)
        
        print(f"Found {len(all_boards)} total boards, using {self.num_boards} valid boards (board_index > 0)")
        
        # Parse cells from ALL valid boards
        self.cells: List[Cell] = []
        self.cell_by_id: Dict[int, Cell] = {}
        self.board_cells: Dict[int, List[Cell]] = {}
        self._parse_all_cells()
        
        # Parse regions from ALL valid boards
        self.regions: Dict[int, Region] = {}
        self._parse_all_regions()
        
        # Parse dominoes (shared across all boards)
        self.dominoes: List[Domino] = []
        self.available_dominoes: Set[Tuple[int, int]] = set()
        self._parse_dominoes()
        
        # Build adjacency map (only within same board)
        self.adjacency: Dict[Cell, List[Cell]] = {}
        self._build_adjacency()
        
        # Solve tracking
        self.move_history: List[Dict] = []
    
    def _parse_all_cells(self):
        """Parse cells from ALL valid boards"""
        global_cell_id = 0
        
        for board_idx, board_data in enumerate(self.boards_data):
            board_cells = []
            original_board_index = board_data.get('board_index', board_idx + 1)
            
            print(f"  Parsing board {board_idx} (original board_index={original_board_index}): {len(board_data['cells'])} cells")
            
            for cell_data in board_data['cells']:
                cell = Cell(
                    id=global_cell_id,
                    row=cell_data['row'],
                    col=cell_data['col'],
                    section=cell_data['section'] + (board_idx * 1000),  # Offset sections per board
                    bbox=cell_data['bbox'],
                    center=cell_data['center'],
                    board_index=board_idx
                )
                self.cells.append(cell)
                self.cell_by_id[global_cell_id] = cell
                board_cells.append(cell)
                global_cell_id += 1
            
            self.board_cells[board_idx] = board_cells
    
    def _parse_all_regions(self):
        """Parse regions from ALL valid boards"""
        for board_idx, board_data in enumerate(self.boards_data):
            # Group cells by section for this board
            for cell in self.board_cells[board_idx]:
                section_id = cell.section
                if section_id not in self.regions:
                    self.regions[section_id] = Region(section_id=section_id)
                self.regions[section_id].add_cell(cell)
            
            # Parse constraints from badges
            if 'badges' in board_data:
                for badge in board_data['badges']:
                    section_id = badge['section_id'] + (board_idx * 1000)  # Match offset
                    text = badge['text']
                    
                    if section_id in self.regions:
                        region = self.regions[section_id]
                        region.constraint_type, region.constraint_value = self._parse_constraint(text)
    
    def _parse_constraint(self, text: str) -> Tuple[str, Optional[int]]:
        """Parse constraint text into type and value"""
        text = text.strip()
        
        if text == '=':
            return ('equal', None)
        elif text == '≠' or text == '!=':
            return ('not_equal', None)
        elif text.startswith('>'):
            value = int(text[1:]) if len(text) > 1 else None
            return ('greater_than', value)
        elif text.startswith('<'):
            value = int(text[1:]) if len(text) > 1 else None
            return ('less_than', value)
        elif text.isdigit():
            return ('sum', int(text))
        else:
            return ('unknown', None)
    
    def _parse_dominoes(self):
        """Parse dominoes from JSON"""
        self.dominoes = []
        self.available_dominoes = set()
        for dom_data in self.data['dominoes']:
            domino = Domino(
                id=dom_data['id'],
                pips_left=dom_data['pips_left'],
                pips_right=dom_data['pips_right'],
                bbox=dom_data['bbox']
            )
            self.dominoes.append(domino)
            self.available_dominoes.add(domino.as_tuple())

        # --- NEW: integrity checks and loud warnings ---
        want = {(d['pips_left'], d['pips_right']) for d in self.data['dominoes']}
        have = {(d.pips_left, d.pips_right) for d in self.dominoes}
        missing = want - have
        if missing:
            raise RuntimeError(f"[puzzle] Missing domino tuples after parse: {sorted(missing)}")
        # specific sanity: (0,0) should be there for this JSON
        if (0, 0) in want and (0, 0) not in have:
            raise RuntimeError("[puzzle] Double-zero (0,0) domino missing after parse.")

        # also ensure all are unplaced initially
        bad_placed = [d for d in self.dominoes if d.placed]
        if bad_placed:
            raise RuntimeError(f"[puzzle] Dominoes pre-marked placed at init: {bad_placed}")

    
    def _build_adjacency(self):
        """Build adjacency map for cells (only within same board)"""
        for cell in self.cells:
            neighbors = []
            for other in self.cells:
                if cell.id != other.id and self._are_adjacent(cell, other):
                    neighbors.append(other)
            self.adjacency[cell] = neighbors
    
    def _are_adjacent(self, cell1: Cell, cell2: Cell) -> bool:
        """Check if two cells are adjacent (horizontally or vertically) AND on same board"""
        # Cells must be on the same board
        if cell1.board_index != cell2.board_index:
            return False
        
        row_diff = abs(cell1.row - cell2.row)
        col_diff = abs(cell1.col - cell2.col)
        
        # Adjacent if exactly one unit apart in one direction
        return (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)
    
    def get_adjacent_cells(self, cell: Cell) -> List[Cell]:
        """Get all adjacent cells"""
        return self.adjacency.get(cell, [])
    
    def get_empty_adjacent_cells(self, cell: Cell) -> List[Cell]:
        """Get adjacent cells that are empty"""
        return [c for c in self.get_adjacent_cells(cell) if not c.occupied]
    
    def get_region(self, cell: Cell) -> Region:
        """Get the region a cell belongs to"""
        return self.regions[cell.section]
    
    def is_complete(self) -> bool:
        """Check if puzzle is completely solved"""
        return all(cell.occupied for cell in self.cells)
    
    def get_completion_percentage(self) -> float:
        """Get percentage of cells filled"""
        filled = sum(1 for c in self.cells if c.occupied)
        return filled / len(self.cells) if self.cells else 0.0
    
    def __repr__(self):
        return f"PipsPuzzle(boards={self.num_boards}, cells={len(self.cells)}, regions={len(self.regions)}, dominoes={len(self.dominoes)})"