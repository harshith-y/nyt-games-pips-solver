import cv2
import numpy as np
from typing import Tuple, List


def count_pips_on_half(half_image: np.ndarray, debug: bool = False) -> int:
    """
    Count pips (dots) on one half of a domino.
    
    Args:
        half_image: BGR image of half a domino
        debug: If True, return debug visualization
        
    Returns:
        Number of pips (0-6)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(half_image, cv2.COLOR_BGR2GRAY)
    
    # Pips are dark circles on light background
    # Use adaptive threshold to handle varying lighting
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Clean up noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size and circularity
    h, w = half_image.shape[:2]
    min_area = (h * w) * 0.005  # Pip must be at least 0.5% of half area
    max_area = (h * w) * 0.15   # Pip can't be more than 15% of half area
    
    pips = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area or area > max_area:
            continue
        
        # Check circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Pips should be reasonably circular (>0.4)
        if circularity > 0.4:
            pips.append(contour)
    
    # Limit to maximum 6 pips (standard domino)
    pip_count = min(len(pips), 6)
    
    if debug:
        # Create visualization
        vis = half_image.copy()
        cv2.drawContours(vis, pips, -1, (0, 255, 0), 2)
        for contour in pips:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
        return pip_count, vis, binary
    
    return pip_count


def detect_pips_on_domino(domino_image: np.ndarray, 
                          bbox: Tuple[int, int, int, int] = None,
                          debug: bool = False) -> Tuple[int, int]:
    """
    Detect pips on both halves of a domino.
    
    Args:
        domino_image: BGR image of a domino piece
        bbox: Optional (x, y, w, h) bounding box (for debugging)
        debug: If True, save debug visualizations
        
    Returns:
        (left_pips, right_pips) - pip counts for each half
    """
    h, w = domino_image.shape[:2]
    
    # Determine orientation and split
    if w > h:
        # Horizontal domino - split vertically
        mid = w // 2
        left_half = domino_image[:, :mid]
        right_half = domino_image[:, mid:]
    else:
        # Vertical domino - split horizontally
        mid = h // 2
        left_half = domino_image[:mid, :]  # Top half (called "left")
        right_half = domino_image[mid:, :]  # Bottom half (called "right")
    
    # Count pips on each half
    if debug:
        left_count, left_vis, left_binary = count_pips_on_half(left_half, debug=True)
        right_count, right_vis, right_binary = count_pips_on_half(right_half, debug=True)
        return left_count, right_count, (left_vis, right_vis, left_binary, right_binary)
    else:
        left_count = count_pips_on_half(left_half, debug=False)
        right_count = count_pips_on_half(right_half, debug=False)
        return left_count, right_count


def detect_all_pips(image_path: str, 
                   domino_boxes: List[Tuple[int, int, int, int]],
                   debug: bool = False) -> List[Tuple[int, int]]:
    """
    Detect pips on all dominoes in an image.
    
    Args:
        image_path: Path to the full screenshot
        domino_boxes: List of (x, y, w, h) bounding boxes for dominoes
        debug: If True, save debug visualizations
        
    Returns:
        List of (left_pips, right_pips) for each domino
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    pip_counts = []
    
    for i, (x, y, w, h) in enumerate(domino_boxes):
        # Extract domino
        domino_crop = img[y:y+h, x:x+w]
        
        # Detect pips
        if debug:
            left_pips, right_pips, debug_imgs = detect_pips_on_domino(
                domino_crop, bbox=(x, y, w, h), debug=True
            )
            left_vis, right_vis, left_binary, right_binary = debug_imgs
            
            # Save debug images
            cv2.imwrite(f'/tmp/domino_{i}_left_vis.png', left_vis)
            cv2.imwrite(f'/tmp/domino_{i}_right_vis.png', right_vis)
            cv2.imwrite(f'/tmp/domino_{i}_left_binary.png', left_binary)
            cv2.imwrite(f'/tmp/domino_{i}_right_binary.png', right_binary)
        else:
            left_pips, right_pips = detect_pips_on_domino(domino_crop)
        
        pip_counts.append((left_pips, right_pips))
    
    return pip_counts


def visualize_pip_counts(image_path: str,
                        domino_boxes: List[Tuple[int, int, int, int]],
                        pip_counts: List[Tuple[int, int]]) -> np.ndarray:
    """
    Create visualization showing domino boxes with pip counts labeled.
    
    Args:
        image_path: Path to the full screenshot
        domino_boxes: List of (x, y, w, h) for each domino
        pip_counts: List of (left_pips, right_pips) for each domino
        
    Returns:
        BGR image with annotations
    """
    img = cv2.imread(image_path)
    vis = img.copy()
    
    for i, ((x, y, w, h), (left, right)) in enumerate(zip(domino_boxes, pip_counts)):
        # Draw bounding box
        color = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        
        # Determine orientation
        if w > h:
            # Horizontal - show pips side by side
            left_pos = (x + w//4 - 10, y + h//2 + 5)
            right_pos = (x + 3*w//4 - 10, y + h//2 + 5)
        else:
            # Vertical - show pips top and bottom
            left_pos = (x + w//2 - 10, y + h//4 + 5)
            right_pos = (x + w//2 - 10, y + 3*h//4 + 5)
        
        # Draw pip counts
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        cv2.putText(vis, str(left), left_pos, font, font_scale, (255, 255, 255), thickness+2)
        cv2.putText(vis, str(left), left_pos, font, font_scale, (0, 0, 255), thickness)
        
        cv2.putText(vis, str(right), right_pos, font, font_scale, (255, 255, 255), thickness+2)
        cv2.putText(vis, str(right), right_pos, font, font_scale, (0, 0, 255), thickness)
        
        # Draw divider line
        if w > h:
            mid_x = x + w//2
            cv2.line(vis, (mid_x, y), (mid_x, y+h), (0, 255, 255), 1)
        else:
            mid_y = y + h//2
            cv2.line(vis, (x, mid_y), (x+w, mid_y), (0, 255, 255), 1)
    
    return vis