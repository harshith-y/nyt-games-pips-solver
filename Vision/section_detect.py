# Vision/section_detect.py
# Assign cells to sections based on dominant color and connectivity
# Handles duplicate colors by treating disconnected regions as separate sections

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Dict, Set
from collections import deque

# Import from cell_grid
from Vision.cell_grid import Cell, GridResult


def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    h = hex_str.strip().lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to LAB color space for perceptual distance."""
    if rgb.ndim == 1:
        rgb = rgb.reshape(1, 1, 3)
    bgr = rgb[:, :, ::-1] if rgb.ndim == 3 else np.array([[[rgb[2], rgb[1], rgb[0]]]])
    lab = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).astype(np.float32)[0]


def _delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Calculate perceptual color distance (Delta E) in LAB space."""
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def _get_dominant_color(roi: np.ndarray, palette_hex: List[str], 
                        ignore_white: bool = True) -> Tuple[int, int, int]:
    """
    Extract the dominant color from a cell ROI.
    Returns the palette color that best matches the most common color in the ROI.
    
    Args:
        roi: RGB image region (H, W, 3)
        palette_hex: List of hex colors to match against
        ignore_white: If True, exclude near-white pixels
    
    Returns:
        RGB tuple of the dominant palette color
    """
    # Convert palette to RGB and LAB
    palette_rgb = [_hex_to_rgb(h) for h in palette_hex]
    palette_lab = [_rgb_to_lab(np.array(rgb)) for rgb in palette_rgb]
    
    # Flatten pixels
    pixels = roi.reshape(-1, 3)
    
    # Filter out white/near-white pixels if requested
    if ignore_white:
        brightness = pixels.mean(axis=1)
        mask = brightness < 240  # Not bright white
        pixels = pixels[mask]
    
    if len(pixels) == 0:
        # Fallback: return first palette color
        return palette_rgb[0]
    
    # Find MODE (most common color) by quantizing and counting
    # Quantize to reduce color space: RGB(195,162,191) → bin(195//8, 162//8, 191//8)
    quantized = (pixels // 8).astype(np.uint8)
    
    # Find most common quantized color
    unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
    most_common_idx = np.argmax(counts)
    mode_color_quantized = unique_colors[most_common_idx]
    
    # Get actual pixels that map to this quantized bin
    matches = np.all(quantized == mode_color_quantized, axis=1)
    mode_pixels = pixels[matches]
    
    # Use mean of this bin as representative color (more stable than single pixel)
    mode_color = mode_pixels.mean(axis=0).astype(np.uint8)
    mode_lab = _rgb_to_lab(mode_color)
    
    # Find closest palette color
    min_distance = float('inf')
    best_color = palette_rgb[0]
    
    for rgb, lab in zip(palette_rgb, palette_lab):
        dist = _delta_e(mode_lab, lab)
        if dist < min_distance:
            min_distance = dist
            best_color = rgb
    
    return best_color


def _build_adjacency_graph(cells: List[Cell]) -> Dict[int, Set[int]]:
    """
    Build adjacency graph: which cells are neighbors?
    Two cells are adjacent if they share an edge (touch in 4-connectivity).
    
    Returns:
        Dict mapping cell.id -> set of adjacent cell.id's
    """
    adjacency = {cell.id: set() for cell in cells}
    
    # Build spatial index for efficiency
    cell_by_pos = {}
    for cell in cells:
        cell_by_pos[(cell.row, cell.col)] = cell.id
    
    # Check 4-connectivity (up, down, left, right)
    for cell in cells:
        r, c = cell.row, cell.col
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_pos = (r + dr, c + dc)
            if neighbor_pos in cell_by_pos:
                neighbor_id = cell_by_pos[neighbor_pos]
                adjacency[cell.id].add(neighbor_id)
    
    return adjacency


def _connected_components_with_tolerance(cells: List[Cell], 
                                        adjacency: Dict[int, Set[int]],
                                        color_map: Dict[int, Tuple[int, int, int]],
                                        tolerance: float = 15.0) -> Dict[int, int]:
    """
    Perform connected components analysis based on color and adjacency.
    Cells with similar colors (within tolerance) that are connected belong to the same section.
    
    Args:
        tolerance: Max Delta E distance to consider colors "same" (default: 15.0)
    
    Returns:
        Dict mapping cell.id -> section_id
    """
    section_assignment = {}
    visited = set()
    section_id = 0
    
    # Convert all colors to LAB for comparison
    color_lab_map = {cid: _rgb_to_lab(np.array(rgb)) for cid, rgb in color_map.items()}
    
    for cell in cells:
        if cell.id in visited:
            continue
        
        # BFS to find all connected cells with similar color
        queue = deque([cell.id])
        visited.add(cell.id)
        component = []
        cell_color_lab = color_lab_map[cell.id]
        
        while queue:
            current_id = queue.popleft()
            component.append(current_id)
            
            # Check neighbors
            for neighbor_id in adjacency[current_id]:
                if neighbor_id not in visited:
                    # Check if colors are similar (within tolerance)
                    neighbor_color_lab = color_lab_map[neighbor_id]
                    color_distance = _delta_e(cell_color_lab, neighbor_color_lab)
                    
                    if color_distance <= tolerance:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)
        
        # Assign this component to a section
        for cid in component:
            section_assignment[cid] = section_id
        
        section_id += 1
    
    return section_assignment


def assign_sections(grid: GridResult, 
                   board_rgb: np.ndarray,
                   palette_hex: List[str],
                   color_tolerance: float = 15.0,
                   debug: bool = False) -> GridResult:
    """
    Assign each cell to a section based on dominant color and connectivity.
    
    Args:
        grid: GridResult with cells detected
        board_rgb: RGB image of the board (H, W, 3)
        palette_hex: List of section colors in hex format
        color_tolerance: Max Delta E distance to consider colors "same" (default: 15.0)
        debug: If True, print debug information
    
    Returns:
        Updated GridResult with cell.section populated
    """
    if not grid.cells:
        print("WARNING: No cells to assign sections to")
        return grid
    
    # Step 1: Extract dominant color for each cell
    color_map = {}  # cell.id -> (R, G, B)
    
    for cell in grid.cells:
        x, y, w, h = cell.bbox
        roi = board_rgb[y:y+h, x:x+w]
        dominant_color = _get_dominant_color(roi, palette_hex)
        color_map[cell.id] = dominant_color
    
    if debug:
        print(f"\n{'='*70}")
        print(f"DEBUG: Section Detection")
        print(f"{'='*70}")
        print(f"  Extracted colors for {len(color_map)} cells")
        
        # Count cells per color
        color_counts = {}
        for color in color_map.values():
            color_counts[color] = color_counts.get(color, 0) + 1
        print(f"  Color distribution:")
        for color, count in sorted(color_counts.items(), key=lambda x: -x[1]):
            print(f"    RGB{color}: {count} cells")
    
    # Step 2: Build adjacency graph
    adjacency = _build_adjacency_graph(grid.cells)
    
    if debug:
        avg_neighbors = sum(len(neighbors) for neighbors in adjacency.values()) / len(adjacency)
        print(f"  Built adjacency graph: avg {avg_neighbors:.1f} neighbors per cell")
    
    # Step 3: Connected components based on color + adjacency
    section_assignment = _connected_components_with_tolerance(
        grid.cells, adjacency, color_map, tolerance=color_tolerance
    )
    
    # Step 4: Update cells with section assignments
    for cell in grid.cells:
        cell.section = section_assignment[cell.id]
    
    if debug:
        num_sections = len(set(section_assignment.values()))
        print(f"  Identified {num_sections} distinct sections")
        
        # Show section sizes
        section_sizes = {}
        for section_id in section_assignment.values():
            section_sizes[section_id] = section_sizes.get(section_id, 0) + 1
        
        print(f"  Section sizes:")
        for section_id, size in sorted(section_sizes.items(), key=lambda x: -x[1])[:10]:
            print(f"    Section {section_id}: {size} cells")
        print(f"{'='*70}\n")
    
    # Update debug info
    grid.debug['num_sections'] = len(set(section_assignment.values()))
    
    return grid


def visualize_sections(board_rgb: np.ndarray, 
                      grid: GridResult,
                      alpha: float = 0.4) -> np.ndarray:
    """
    Create visualization overlay showing detected sections with different colors.
    
    Args:
        board_rgb: Original board image (H, W, 3)
        grid: GridResult with section assignments
        alpha: Transparency of overlay (0=transparent, 1=opaque)
    
    Returns:
        BGR image with section overlay
    """
    overlay = board_rgb.copy()
    
    # Generate distinct colors for each section
    num_sections = len(set(cell.section for cell in grid.cells if cell.section is not None))
    
    # Use HSV to generate visually distinct colors
    section_colors = {}
    for i, section_id in enumerate(sorted(set(cell.section for cell in grid.cells if cell.section is not None))):
        hue = int((i * 180 / num_sections) % 180)
        section_colors[section_id] = cv2.cvtColor(
            np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2RGB
        )[0, 0]
    
    # Draw cells colored by section
    for cell in grid.cells:
        if cell.section is None:
            continue
        
        x, y, w, h = cell.bbox
        color = section_colors[cell.section]
        
        # Draw filled rectangle with transparency
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color.tolist(), -1)
    
    # Blend with original
    result = cv2.addWeighted(overlay, alpha, board_rgb, 1-alpha, 0)
    
    # Draw section IDs
    for cell in grid.cells:
        if cell.section is None:
            continue
        cx, cy = cell.center
        cv2.putText(result, str(cell.section), (cx-10, cy+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)