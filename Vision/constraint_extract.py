# constraint_extract.py
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from Vision.cell_grid import GridResult
import easyocr
from PIL import Image


@dataclass
class Badge:
    """Represents a detected constraint badge"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]  # (x, y)
    color_rgb: Tuple[int, int, int]  # Detected badge color
    color_hex: str  # Badge color as hex
    section_color: str  # Mapped section color (hex)
    section_id: Optional[int] = None  # Which section this badge belongs to
    text: Optional[str] = None  # OCR result (will be added in step 7)
    roi: Optional[np.ndarray] = None  # Badge image region


# Badge-to-Section color mapping
BADGE_TO_SECTION_MAP = {
    "#9251ca": "#c3a2bf",  # Purple badge → Purple section
    "#d15609": "#e9bd8c",  # Orange badge → Peach section
    "#464fb1": "#b2a5bf",  # Blue badge → Lilac section
    "#db137a": "#e89fae",  # Magenta badge → Pink section
    "#008293": "#9dbfc1",  # Teal badge → Teal-gray section
    "#547601": "#b6b18a",  # Olive badge → Olive section
    # Note: Beige (#e1cbc5) has no badge - unconstrained cells
}


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex string"""
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex string to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _color_distance(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """Calculate Euclidean distance between two RGB colors"""
    # Convert to float to avoid overflow
    r1, g1, b1 = float(rgb1[0]), float(rgb1[1]), float(rgb1[2])
    r2, g2, b2 = float(rgb2[0]), float(rgb2[1]), float(rgb2[2])
    return np.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)


def _get_dominant_color(roi: np.ndarray) -> Tuple[int, int, int]:
    """
    Extract dominant color from badge ROI using mode (most common color).
    Quantizes color space to find most frequent color bin.
    """
    if roi.size == 0:
        return (0, 0, 0)
    
    # Flatten pixels
    pixels = roi.reshape(-1, 3)
    
    # Quantize to reduce color space (group similar colors)
    quantized = (pixels // 16).astype(np.uint8)  # Bin size of 16
    
    # Find most common quantized color
    unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
    most_common_idx = np.argmax(counts)
    mode_color_quantized = unique_colors[most_common_idx]
    
    # Get actual pixels in this bin
    matches = np.all(quantized == mode_color_quantized, axis=1)
    mode_pixels = pixels[matches]
    
    # Use mean of this bin as representative color
    mode_color = mode_pixels.mean(axis=0).astype(np.uint8)
    
    return tuple(mode_color)


def _preprocess_for_ocr(badge_roi: np.ndarray) -> np.ndarray:
    """
    Preprocess badge image for OCR.
    Goal: BLACK text on WHITE background, high contrast.
    
    Args:
        badge_roi: Badge image region (RGB)
    
    Returns:
        Binary image (grayscale, pure black/white with BLACK text on WHITE bg)
    """
    # 1. Convert to grayscale
    if len(badge_roi.shape) == 3:
        gray = cv2.cvtColor(badge_roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = badge_roi.copy()
    
    # 2. Resize significantly if too small (OCR likes 50-70px text height)
    h, w = gray.shape
    if h < 50:
        scale = 60 / h
        new_w = int(w * scale)
        new_h = 60
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 3. Apply bilateral filter to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 4. Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. CRITICAL: ALWAYS ensure BLACK text on WHITE background
    # Check what the text color is by looking at a small region in the center
    h, w = binary.shape
    center_h = slice(h//3, 2*h//3)
    center_w = slice(w//3, 2*w//3)
    center = binary[center_h, center_w]
    
    # Get mean value of center region (where text likely is)
    center_mean = center.mean()
    
    # If center is bright (>127), text is white -> need to invert
    if center_mean > 127:
        binary = cv2.bitwise_not(binary)
    
    # 6. Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary


def ocr_badge_text_easyocr(badge_roi: np.ndarray, reader, debug: bool = False, 
                           badge_id: int = 0, debug_dir: str = "data/debug") -> str:
    """
    Extract text from badge using EasyOCR.
    
    Args:
        badge_roi: Badge image region (RGB numpy array)
        reader: EasyOCR reader instance
        debug: If True, print OCR results and save debug images
        badge_id: Badge number for debug file naming
        debug_dir: Directory to save debug images
    
    Returns:
        Detected text (e.g., ">4", "12", "=")
    """
    if badge_roi is None or badge_roi.size == 0:
        return ""
    
    # Step 1: Extract center region (text is in middle of badge)
    h, w = badge_roi.shape[:2]
    if h < 10 or w < 10:  # Too small
        return ""
    
    # Extract inner region (avoid badge borders)
    pad_h = h // 6
    pad_w = w // 6
    text_region = badge_roi[pad_h:h-pad_h, pad_w:w-pad_w]
    
    # Step 2: Try MULTIPLE approaches with EasyOCR
    all_results = []
    
    # APPROACH 1: Use ORIGINAL colored badge (no preprocessing)
    # Upscale for better detection
    h_orig, w_orig = text_region.shape[:2]
    if h_orig < 80:
        scale = 120 / h_orig  # Even bigger
        new_w = int(w_orig * scale)
        upscaled_color = cv2.resize(text_region, (new_w, 120), interpolation=cv2.INTER_CUBIC)
    else:
        upscaled_color = text_region
    
    try:
        # Try with allowlist
        results1 = reader.readtext(
            upscaled_color,
            detail=1,
            paragraph=False,
            allowlist='0123456789<>=÷',
            text_threshold=0.4,  # Lower threshold to detect more
        )
        all_results.extend([(r[1], r[2], 'color+allowlist') for r in results1])
        
        # Try WITHOUT allowlist (might catch = and 0 better)
        results2 = reader.readtext(
            upscaled_color,
            detail=1,
            paragraph=False,
            text_threshold=0.4,
        )
        all_results.extend([(r[1], r[2], 'color+no_allowlist') for r in results2])
        
    except Exception as e:
        if debug:
            print(f"    Approach 1 (color) failed: {e}")
    
    # APPROACH 2: Use preprocessed black/white
    processed = _preprocess_for_ocr(text_region)
    h, w = processed.shape
    if h < 80:
        scale = 120 / h
        new_w = int(w * scale)
        processed = cv2.resize(processed, (new_w, 120), interpolation=cv2.INTER_CUBIC)
    
    # Convert to RGB for EasyOCR
    if len(processed.shape) == 2:
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    else:
        processed_rgb = processed
    
    try:
        # Try preprocessed with allowlist
        results3 = reader.readtext(
            processed_rgb,
            detail=1,
            paragraph=False,
            allowlist='0123456789<>=÷',
            text_threshold=0.4,
        )
        all_results.extend([(r[1], r[2], 'preprocessed+allowlist') for r in results3])
        
        # Try preprocessed without allowlist
        results4 = reader.readtext(
            processed_rgb,
            detail=1,
            paragraph=False,
            text_threshold=0.4,
        )
        all_results.extend([(r[1], r[2], 'preprocessed+no_allowlist') for r in results4])
        
    except Exception as e:
        if debug:
            print(f"    Approach 2 (preprocessed) failed: {e}")
    
    # APPROACH 3: Try inverted preprocessed (white text on black)
    try:
        inverted = cv2.bitwise_not(processed)
        inverted_rgb = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
        
        results5 = reader.readtext(
            inverted_rgb,
            detail=1,
            paragraph=False,
            allowlist='0123456789<>=÷',
            text_threshold=0.4,
        )
        all_results.extend([(r[1], r[2], 'inverted+allowlist') for r in results5])
        
    except Exception as e:
        if debug:
            print(f"    Approach 3 (inverted) failed: {e}")
    
    # Step 3: Pick best result
    if debug:
        print(f"    EasyOCR found {len(all_results)} total results:")
        for text, conf, approach in sorted(all_results, key=lambda x: x[1], reverse=True)[:5]:
            print(f"      '{text}' (conf={conf:.2f}, {approach})")
    
    if all_results:
        # Get result with highest confidence
        best = max(all_results, key=lambda x: x[1])
        text = best[0]
        confidence = best[1]
    else:
        text = ""
        confidence = 0.0
    
    # SPECIAL HANDLING: If no results, try even more aggressive settings
    if not text and debug:
        print(f"    No text detected! Trying ultra-aggressive settings...")
    
    if not text:
        # Last resort: Try with VERY low threshold and no allowlist
        try:
            # Try the inverted image with minimal restrictions
            inverted = cv2.bitwise_not(processed)
            inverted_rgb = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
            
            emergency_results = reader.readtext(
                inverted_rgb,
                detail=1,
                paragraph=False,
                text_threshold=0.1,  # VERY low threshold
                low_text=0.1,  # Low text score threshold
                link_threshold=0.1,  # Low link threshold
            )
            
            if emergency_results and debug:
                print(f"    Emergency detection found {len(emergency_results)} results:")
                for bbox, t, c in emergency_results:
                    print(f"      '{t}' (conf={c:.2f})")
            
            if emergency_results:
                best_emergency = max(emergency_results, key=lambda x: x[2])
                text = best_emergency[1]
                confidence = best_emergency[2]
                if debug:
                    print(f"    Using emergency result: '{text}' (conf={confidence:.2f})")
        except Exception as e:
            if debug:
                print(f"    Emergency detection failed: {e}")
    
    # Step 4: Clean up result
    text = text.strip()
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    
    # Common OCR mistakes
    text = text.replace("O", "0")
    text = text.replace("o", "0")
    text = text.replace("l", "1")
    text = text.replace("I", "1")
    text = text.replace("S", "5")
    text = text.replace("|", "1")
    
    # Special case: "<" is often misread "4" (the angled part without crossbar)
    # Replace standalone "<" with "4" if it appears alone
    if text.strip() == "<":
        text = "4"
        if debug:
            print(f"    Note: '<' symbol detected, likely misread '4'")
    
    # Special case: ">" might also be misread "4" in some fonts
    # (less common, but possible if image is mirrored or rotated)
    if text.strip() == ">":
        text = "4"
        if debug:
            print(f"    Note: '>' symbol detected, likely misread '4'")
    
    # Special case: "?" or "???" means unrecognized/low quality
    # Keep as-is for now (could be legitimate uncertainty marker)
    if "?" in text and debug:
        print(f"    Warning: Text contains '?' - OCR uncertain or low quality")
    
    # Special case: "not equals" symbol often misread as numbers like "72", "44", "7/2"
    # Heuristic: Any number > 50 is likely OCR error for "≠"
    # (Valid constraints are 0-40 max, since multi-cell sums rarely exceed that)
    if text.replace(">", "").replace("<", "").isdigit():
        num = int(text.replace(">", "").replace("<", ""))
        if num > 50:
            text = "≠"
            if debug:
                print(f"    Note: Number {num} too large for valid constraint, assuming '≠'")
    
    # Also catch common OCR variations of "≠"
    # "44" = two horizontal lines seen as two 4's
    # "72" = slash + lines misread
    # "7/2", "Z2", etc. = partial recognition of the slash
    if text in ["44", "7/2", "7Z", "7z", "Z2", "7l", "7I", "/=", "4/4"]:
        text = "≠"
        if debug:
            print(f"    Note: Detected common '≠' OCR variation '{text}', correcting")
    
    # Special case: "7" might be "≠" if the badge has strong horizontal line patterns
    # The "≠" symbol has TWO prominent horizontal lines
    # A real "7" has one angled line and one horizontal line
    if text == "7":
        # Analyze the preprocessed image for horizontal line dominance
        h, w = processed.shape if len(processed.shape) == 2 else processed.shape[:2]
        
        if h > 20 and w > 20:
            # Sample three horizontal strips (top third, middle, bottom third)
            top_strip = processed[int(h*0.25):int(h*0.35), :]
            mid_strip = processed[int(h*0.45):int(h*0.55), :]
            bot_strip = processed[int(h*0.65):int(h*0.75), :]
            
            # Count bright pixels (white = foreground) in each strip
            top_bright = np.mean(top_strip > 127)
            mid_bright = np.mean(mid_strip > 127)
            bot_bright = np.mean(bot_strip > 127)
            
            # "≠" has bright pixels in top AND bottom (two horizontal lines)
            # "7" has bright pixels mainly in top (one horizontal line)
            has_top_line = top_bright > 0.3
            has_bot_line = bot_bright > 0.3
            
            # If BOTH top and bottom have strong horizontal presence, it's likely "≠"
            if has_top_line and has_bot_line:
                text = "≠"
                if debug:
                    print(f"    Note: '7' has two horizontal lines (top={top_bright:.2f}, bot={bot_bright:.2f}), likely '≠'")
    
    # Special case: "4" might be "≠" if the badge has strong horizontal line patterns
    # The "≠" symbol has TWO prominent horizontal lines that are SYMMETRIC
    # A real "4" has one strong crossbar in middle, with minimal top/bottom content
    if text == "4":
        # Analyze the preprocessed image for horizontal line dominance
        h, w = processed.shape if len(processed.shape) == 2 else processed.shape[:2]
        
        if h > 20 and w > 20:
            # Sample FIVE horizontal strips for more precision
            top_strip = processed[int(h*0.20):int(h*0.30), :]      # Top region
            upper_mid = processed[int(h*0.35):int(h*0.45), :]      # Upper middle
            mid_strip = processed[int(h*0.45):int(h*0.55), :]      # True middle (crossbar)
            lower_mid = processed[int(h*0.55):int(h*0.65), :]      # Lower middle
            bot_strip = processed[int(h*0.70):int(h*0.80), :]      # Bottom region
            
            # Count bright pixels (white = foreground) in each strip
            top_bright = np.mean(top_strip > 127)
            upper_mid_bright = np.mean(upper_mid > 127)
            mid_bright = np.mean(mid_strip > 127)
            lower_mid_bright = np.mean(lower_mid > 127)
            bot_bright = np.mean(bot_strip > 127)
            
            # NEW: Sample vertical strips to detect vertical strokes in "4"
            left_strip = processed[:, 0:int(w*0.3)]  # Left vertical
            right_strip = processed[:, int(w*0.7):]  # Right vertical
            
            left_vert_bright = np.mean(left_strip > 127)
            right_vert_bright = np.mean(right_strip > 127)
            
            # Key differences between "4" and "≠":
            # REAL "4":
            #   - Middle (crossbar) is STRONGEST region (>55%)
            #   - Top/bottom are WEAK (<42%)
            #   - Has LEFT vertical stroke (>35% brightness)
            #   - Has RIGHT vertical stroke (>25% brightness)
            #   - Clear peak in the middle
            # 
            # "≠" SYMBOL:
            #   - Top and/or bottom are STRONG (>45%)
            #   - Top and bottom are REASONABLY SYMMETRIC (diff <20%)
            #   - Middle is WEAK (<35%)
            #   - Vertical brightness from diagonal only (not structural verticals)
            #   - Extremes dominate over middle
            
            # Condition 1: At least ONE extreme is very strong (relaxed from both)
            has_very_strong_top = top_bright > 0.45
            has_very_strong_bot = bot_bright > 0.45
            has_strong_extreme = has_very_strong_top or has_very_strong_bot
            
            # Condition 2: Extremes reasonably symmetric (relaxed: 20% from 10%)
            reasonably_similar_ends = abs(top_bright - bot_bright) < 0.20
            
            # Condition 3: Middle MUCH weaker (keep at 0.08)
            middle_much_weaker = mid_bright < min(top_bright, bot_bright) - 0.08
            
            # Condition 4: NO strong middle peak
            middle_not_dominant = mid_bright < max(top_bright, bot_bright, upper_mid_bright, lower_mid_bright)
            
            # Condition 5: Horizontal line continuity
            top_line_continuous = np.std(np.mean(top_strip > 127, axis=0)) < 0.3
            bot_line_continuous = np.std(np.mean(bot_strip > 127, axis=0)) < 0.3
            
            # Condition 6: Check if verticals are TRUE structural verticals (not just diagonal)
            # Real "4" has BOTH strong left (>35%) AND right (>25%)
            # "≠" may have brightness from diagonal (<35% on sides), but not true structural verticals
            has_structural_verticals = (left_vert_bright > 0.35) and (right_vert_bright > 0.25)
            no_structural_verticals = not has_structural_verticals
            
            # Condition 7: Middle is WEAK (not just weaker)
            middle_is_weak = mid_bright < 0.35
            
            # Condition 8: At least one extreme DOMINATES middle significantly
            # (relaxed: only need ONE to dominate, not both)
            at_least_one_dominates = (top_bright > mid_bright + 0.15) or (bot_bright > mid_bright + 0.15)
            
            # ALL 8 conditions must be true for "≠"
            # BUT: Add bypass for VERY clear "≠" patterns (extremely weak middle + strong domination)
            # This catches "≠" where diagonal creates vertical brightness
            extremely_weak_middle = mid_bright < 0.22
            very_strong_domination = (top_bright > mid_bright + 0.25) or (bot_bright > mid_bright + 0.25)
            
            # Bypass path: If middle is EXTREMELY weak and extremes STRONGLY dominate, it's clearly "≠"
            if extremely_weak_middle and very_strong_domination and has_strong_extreme and reasonably_similar_ends:
                text = "≠"
                if debug:
                    print(f"    Note: '4' has EXTREMELY weak middle pattern (mid={mid_bright:.2f}, top={top_bright:.2f}, bot={bot_bright:.2f}) → '≠'")
            # Full check path: All conditions including vertical check
            elif (has_strong_extreme and reasonably_similar_ends and 
                middle_much_weaker and middle_not_dominant and 
                top_line_continuous and bot_line_continuous and 
                no_structural_verticals and middle_is_weak and 
                at_least_one_dominates):
                text = "≠"
                if debug:
                    print(f"    Note: '4' matches '≠' pattern: top={top_bright:.2f}, mid={mid_bright:.2f}, bot={bot_bright:.2f}, vert_L={left_vert_bright:.2f}, vert_R={right_vert_bright:.2f}")
            elif debug and (has_very_strong_top or has_very_strong_bot or top_bright > 0.35 or bot_bright > 0.35):
                # Debug: show why we didn't convert
                print(f"    Note: '4' analysis: top={top_bright:.2f}, mid={mid_bright:.2f}, bot={bot_bright:.2f}")
                print(f"          Vertical: left={left_vert_bright:.2f}, right={right_vert_bright:.2f}")
                print(f"          Bypass check: extreme_weak={extremely_weak_middle} (mid<0.22), very_strong_dom={very_strong_domination}")
                print(f"          Full check: strong_extreme={has_strong_extreme}, sym<0.20={reasonably_similar_ends}, mid_weak={middle_is_weak}")
                print(f"          no_struct_vert={no_structural_verticals} (need L>0.35 AND R>0.25), dominates={at_least_one_dominates}")
                print(f"          Kept as '4' (not all conditions met)")

    if debug:
        print(f"    FINAL Result: '{text}' (confidence: {confidence:.2f})")

    return text

def detect_badges(board_rgb: np.ndarray,
                  saturation_threshold: int = 120,  # Raised to 120 for better separation
                  min_area: int = None,  # Will be calculated adaptively
                  max_area: int = None,  # Will be calculated adaptively
                  aspect_ratio_range: Tuple[float, float] = (0.6, 1.5),  # Wider range (was 0.7-1.3)
                  debug: bool = False) -> List[Badge]:
    """
    Detect constraint badges in the board image (Steps 1-4).
    
    Args:
        board_rgb: RGB image of the board
        saturation_threshold: Min saturation to consider as badge (0-255)
        min_area: Minimum badge area in pixels (auto-calculated if None)
        max_area: Maximum badge area in pixels (auto-calculated if None)
        aspect_ratio_range: (min, max) aspect ratio for badge bounding box
        debug: If True, print debug information
    
    Returns:
        List of detected Badge objects
    """
    # Calculate adaptive area thresholds based on board size
    H, W = board_rgb.shape[:2]
    board_area = H * W
    
    # Badges are typically 0.2% to 4% of board area (widened range)
    if min_area is None:
        min_area = int(board_area * 0.002)  # Lowered from 0.003 to 0.002
    if max_area is None:
        max_area = int(board_area * 0.1)   # Increased from 0.03 to 0.04
    
    # Step 1: Convert to HSV and filter by saturation
    hsv = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    
    # Create binary mask: high saturation = badges
    mask = (saturation > saturation_threshold).astype(np.uint8) * 255
    
    if debug:
        print(f"\n{'='*70}")
        print(f"DEBUG: Badge Detection")
        print(f"{'='*70}")
        print(f"  Board size: {W}x{H} ({board_area} pixels)")
        print(f"  Adaptive area range: {min_area} - {max_area} pixels")
        print(f"  Saturation threshold: {saturation_threshold}")
        print(f"  Mask pixels above threshold: {np.sum(mask > 0)}")
    
    # Step 2: Find contours (blob detection)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        print(f"  Total contours found: {len(contours)}")
        # Show area distribution
        areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)
        print(f"  Top 10 contour areas: {[int(a) for a in areas[:10]]}")
    
    # Step 3: Filter by size and shape
    badges = []
    min_aspect, max_aspect = aspect_ratio_range
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip if too small or too large
        if area < min_area or area > max_area:
            continue
        
        # Check aspect ratio (roughly square)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            continue
        
        # Step 4: Extract color from badge
        badge_roi = board_rgb[y:y+h, x:x+w]
        dominant_color = _get_dominant_color(badge_roi)
        color_hex = _rgb_to_hex(dominant_color)
        
        center = (x + w // 2, y + h // 2)
        
        badge = Badge(
            bbox=(x, y, w, h),
            center=center,
            color_rgb=dominant_color,
            color_hex=color_hex,
            section_color="",  # Will be filled in step 5
            roi=badge_roi
        )
        
        badges.append(badge)
        
        if debug:
            print(f"  Badge {len(badges)}: area={area:.0f}, bbox=({x},{y},{w},{h}), "
                  f"aspect={aspect_ratio:.2f}, color={color_hex}")
    
    if debug:
        print(f"  Filtered badges: {len(badges)}")
        print(f"{'='*70}\n")
    
    return badges


def match_badge_to_section_color(badge: Badge, tolerance: float = 50.0) -> str:
    """
    Match detected badge color to known badge colors (Step 5).
    
    Args:
        badge: Badge object with color_rgb
        tolerance: Max RGB distance to accept match
    
    Returns:
        Section color (hex) that this badge corresponds to
    """
    min_distance = float('inf')
    best_section_color = None
    
    for badge_hex, section_hex in BADGE_TO_SECTION_MAP.items():
        badge_ref_rgb = _hex_to_rgb(badge_hex)
        distance = _color_distance(badge.color_rgb, badge_ref_rgb)
        
        if distance < min_distance:
            min_distance = distance
            best_section_color = section_hex
    
    # If no good match found, return empty string
    if min_distance > tolerance:
        return ""
    
    return best_section_color


def assign_badge_to_section(badge: Badge,
                            grid: GridResult,
                            board_rgb: np.ndarray,
                            already_assigned_sections: set,
                            diagonal_distance: int = 50) -> Optional[int]:
    """
    Find which section this badge belongs to (Step 6).
    Badges are placed diagonally down-right from their section.
    So we search diagonally up-left (northwest) from the badge.
    
    Args:
        badge: Badge object
        grid: GridResult with cells and section assignments
        board_rgb: RGB image (not used in this approach)
        already_assigned_sections: Set of section IDs that already have badges
        diagonal_distance: How far diagonally to search (pixels)
    
    Returns:
        Section ID, or None if no match found
    """
    badge_x, badge_y = badge.center
    
    # Look diagonally northwest (up and left)
    # Search point is approximately diagonal_distance pixels in both directions
    search_x = badge_x - diagonal_distance
    search_y = badge_y - diagonal_distance
    
    # Find which cell is at or near this search point
    min_dist = float('inf')
    closest_cell = None
    
    for cell in grid.cells:
        cx, cy = cell.center
        
        # Calculate distance from search point to this cell
        dist = np.sqrt((search_x - cx)**2 + (search_y - cy)**2)
        
        if dist < min_dist:
            min_dist = dist
            closest_cell = cell
    
    # Return the section of the closest cell to our northwest search point
    if closest_cell is not None:
        return closest_cell.section
    
    return None


def extract_constraints(grid: GridResult,
                       board_rgb: np.ndarray,
                       debug: bool = False,
                       debug_dir: str = "data/debug",
                       reader=None) -> GridResult:
    """
    Main function: Detect badges and assign to sections (Steps 1-7).
    Now uses EasyOCR for text extraction!
    
    Args:
        grid: GridResult with cells and sections already assigned
        board_rgb: RGB image of the board
        debug: If True, print debug information
        debug_dir: Directory to save debug images
        reader: EasyOCR reader instance (will create if None)
    
    Returns:
        Updated GridResult with badges (including OCR text)
    """
    # Initialize EasyOCR reader if not provided
    if reader is None:
        if debug:
            print(f"\n{'='*70}")
            print(f"Initializing EasyOCR (first run downloads models ~100MB)...")
            print(f"{'='*70}")
        reader = easyocr.Reader(['en'], gpu=False)
    
    # Step 1-4: Detect badges
    badges = detect_badges(board_rgb, debug=debug)
    
    # Step 5: Match badge colors to section colors
    for badge in badges:
        badge.section_color = match_badge_to_section_color(badge)
    
    # Step 6: Assign badges to sections
    # Track which sections already have badges (each section can only have ONE badge)
    already_assigned_sections = set()
    
    for badge in badges:
        badge.section_id = assign_badge_to_section(
            badge, grid, board_rgb, already_assigned_sections
        )
        if badge.section_id is not None:
            already_assigned_sections.add(badge.section_id)
    
    # Step 7: OCR the badge text with EasyOCR
    if debug:
        print(f"\n{'='*70}")
        print(f"DEBUG: OCR Badge Text (EasyOCR)")
        print(f"{'='*70}")
    
    for i, badge in enumerate(badges):
        if debug:
            print(f"  Badge {i+1}:")
        badge.text = ocr_badge_text_easyocr(badge.roi, reader, debug=debug, 
                                           badge_id=i+1, debug_dir=debug_dir)
        if debug and not badge.text:
            print(f"    FINAL Result: (empty)")
    
    if debug:
        print(f"{'='*70}\n")
        print(f"  Debug images saved to {debug_dir}/badge_*.png")
    
    if debug:
        print(f"\n{'='*70}")
        print(f"DEBUG: Badge-to-Section Assignment Summary")
        print(f"{'='*70}")
        for i, badge in enumerate(badges):
            print(f"  Badge {i+1}:")
            print(f"    Position: {badge.center}")
            print(f"    Detected color: {badge.color_hex}")
            print(f"    Mapped to section color: {badge.section_color}")
            print(f"    Assigned to section ID: {badge.section_id}")
            print(f"    OCR Text: '{badge.text}'")
        print(f"{'='*70}\n")
    
    # Store badges in grid debug info
    if not hasattr(grid, 'badges'):
        grid.badges = badges
    else:
        grid.badges = badges
    
    return grid


def visualize_badges(board_rgb: np.ndarray,
                     grid: GridResult,
                     alpha: float = 0.6) -> np.ndarray:
    """
    Visualize detected badges on the board image.
    Now includes OCR text!
    
    Args:
        board_rgb: Original RGB board image
        grid: GridResult with badges stored
        alpha: Transparency for overlays
    
    Returns:
        BGR image with badge visualizations
    """
    # Convert to BGR for OpenCV
    vis = cv2.cvtColor(board_rgb.copy(), cv2.COLOR_RGB2BGR)
    
    if not hasattr(grid, 'badges'):
        return vis
    
    # Draw each badge
    for badge in grid.badges:
        x, y, w, h = badge.bbox
        
        # Draw bounding box
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 255), 2)  # Yellow box
        
        # Draw center point
        cv2.circle(vis, badge.center, 5, (0, 0, 255), -1)  # Red dot
        
        # Draw section ID and OCR text
        if badge.section_id is not None:
            label = f"S{badge.section_id}"
            if badge.text:
                label += f": {badge.text}"
            
            # Draw text with background for readability
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis, (x, y-text_h-10), (x+text_w+5, y), (0, 0, 0), -1)
            cv2.putText(vis, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 2)
    
    return vis