import json
from typing import Dict, List
from puzzle import PipsPuzzle
from datetime import datetime


class SolutionFormatter:
    """Formats puzzle solutions for output"""
    
    @staticmethod
    def format_solution_json(puzzle: PipsPuzzle, stats: Dict) -> Dict:
        """
        Format solution as JSON
        """
        solution = {
            'puzzle_info': {
                'total_cells': len(puzzle.cells),
                'total_regions': len(puzzle.regions),
                'total_dominoes': len(puzzle.dominoes),
                'solved': puzzle.is_complete(),
                'timestamp': datetime.now().isoformat()
            },
            'solving_stats': stats,
            'placements': [],
            'region_validation': {}
        }
        
        # Add each domino placement
        for move in puzzle.move_history:
            domino = move['domino']
            cell1 = move['cell1']
            cell2 = move['cell2']
            
            placement = {
                'domino_id': domino.id,
                'pips': [domino.pips_left, domino.pips_right],
                'cells': [cell1.id, cell2.id],
                'positions': [
                    {'row': cell1.row, 'col': cell1.col},
                    {'row': cell2.row, 'col': cell2.col}
                ],
                'regions': [cell1.section, cell2.section]
            }
            solution['placements'].append(placement)
        
        # Validate regions
        for region_id, region in puzzle.regions.items():
            solution['region_validation'][region_id] = {
                'constraint_type': region.constraint_type,
                'constraint_value': region.constraint_value,
                'actual_sum': region.current_sum,
                'satisfied': SolutionFormatter._check_region_satisfied(region)
            }
        
        return solution
    
    @staticmethod
    def _check_region_satisfied(region) -> bool:
        """Check if region constraint is satisfied"""
        if region.constraint_type == 'sum':
            return region.current_sum == region.constraint_value
        elif region.constraint_type == 'equal':
            values = [c.value for c in region.get_filled_cells()]
            return len(set(values)) <= 1
        elif region.constraint_type == 'not_equal':
            values = [c.value for c in region.get_filled_cells()]
            return len(set(values)) == len(values)
        elif region.constraint_type == 'less_than':
            if region.constraint_value:
                return region.current_sum < region.constraint_value
        elif region.constraint_type == 'greater_than':
            if region.constraint_value:
                return region.current_sum > region.constraint_value
        return True
    
    @staticmethod
    def format_solution_human_readable(puzzle: PipsPuzzle) -> str:
        """
        Format solution as human-readable text
        """
        lines = []
        lines.append("=" * 60)
        lines.append("PIPS PUZZLE SOLUTION")
        lines.append("=" * 60)
        lines.append(f"\nPuzzle has {len(puzzle.cells)} cells, {len(puzzle.regions)} regions")
        lines.append(f"Placed {len(puzzle.move_history)} dominoes\n")
        
        lines.append("DOMINO PLACEMENTS:")
        lines.append("-" * 60)
        
        for i, move in enumerate(puzzle.move_history, 1):
            domino = move['domino']
            cell1 = move['cell1']
            cell2 = move['cell2']
            
            orientation = "horizontal" if cell1.row == cell2.row else "vertical"
            
            lines.append(
                f"{i:2d}. Domino ({domino.pips_left},{domino.pips_right}) "
                f"→ Cells {cell1.id:2d}-{cell2.id:2d} "
                f"at ({cell1.row},{cell1.col})-({cell2.row},{cell2.col}) "
                f"[{orientation}]"
            )
        
        lines.append("\n" + "=" * 60)
        lines.append("REGION VALIDATION:")
        lines.append("-" * 60)
        
        for region_id, region in sorted(puzzle.regions.items()):
            satisfied = "✓" if SolutionFormatter._check_region_satisfied(region) else "✗"
            
            constraint_str = f"{region.constraint_type}"
            if region.constraint_value is not None:
                constraint_str += f" {region.constraint_value}"
            
            lines.append(
                f"Region {region_id:2d}: {constraint_str:15s} "
                f"→ Sum: {region.current_sum:3d} {satisfied}"
            )
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_grid_visualization(puzzle: PipsPuzzle) -> str:
        """
        Create a text-based grid visualization.
        For multi-board puzzles, shows each board separately.
        """
        if not puzzle.cells:
            return "Empty puzzle"
        
        lines = []
        lines.append("\nGRID VISUALIZATION:")
        
        # Handle multi-board puzzles
        if puzzle.num_boards > 1:
            for board_idx in range(puzzle.num_boards):
                board_cells = puzzle.board_cells.get(board_idx, [])
                if not board_cells:
                    continue
                
                lines.append(f"\n--- Board {board_idx + 1} ---")
                
                # Find grid dimensions for this board
                max_row = max(c.row for c in board_cells)
                max_col = max(c.col for c in board_cells)
                
                # Create grid
                grid = [[' ' for _ in range(max_col + 1)] for _ in range(max_row + 1)]
                
                # Fill in values
                for cell in board_cells:
                    if cell.occupied:
                        grid[cell.row][cell.col] = str(cell.value) if cell.value is not None else '?'
                    else:
                        grid[cell.row][cell.col] = '·'
                
                # Format grid
                lines.append("-" * (max_col * 2 + 3))
                for row in grid:
                    lines.append("  " + " ".join(row))
                lines.append("-" * (max_col * 2 + 3))
        
        else:
            # Single board puzzle
            max_row = max(c.row for c in puzzle.cells)
            max_col = max(c.col for c in puzzle.cells)
            
            # Create grid
            grid = [[' ' for _ in range(max_col + 1)] for _ in range(max_row + 1)]
            
            # Fill in values
            for cell in puzzle.cells:
                if cell.occupied:
                    grid[cell.row][cell.col] = str(cell.value) if cell.value is not None else '?'
                else:
                    grid[cell.row][cell.col] = '·'
            
            # Format grid
            lines.append("-" * (max_col * 2 + 3))
            for row in grid:
                lines.append("  " + " ".join(row))
            lines.append("-" * (max_col * 2 + 3))
        
        return "\n".join(lines)
    
    @staticmethod
    def save_solution(puzzle: PipsPuzzle, stats: Dict, output_path: str):
        """
        Save solution to JSON file
        """
        solution = SolutionFormatter.format_solution_json(puzzle, stats)
        
        with open(output_path, 'w') as f:
            json.dump(solution, f, indent=2)
        
        print(f"\n✓ Solution saved to: {output_path}")
    
    @staticmethod
    def save_human_readable(puzzle: PipsPuzzle, output_path: str):
        """
        Save human-readable solution to text file
        """
        text = SolutionFormatter.format_solution_human_readable(puzzle)
        text += "\n\n" + SolutionFormatter.format_grid_visualization(puzzle)
        
        with open(output_path, 'w') as f:
            f.write(text)
        
        print(f"✓ Human-readable solution saved to: {output_path}")