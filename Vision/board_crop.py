# Vision/board_cropy.py
import cv2
import numpy as np
from typing import List, Tuple, Literal

PAD_FRAC = 0  # padding around crops

def _clip(v, lo, hi): return max(lo, min(hi, v))

def _white_mask_lab(bgr: np.ndarray) -> np.ndarray:
    """Very robust white mask using LAB (white ~ high L, a≈128, b≈128)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # thresholds tuned for screenshots (bright backgrounds, pastel boards)
    white = (L >= 215) & (np.abs(A.astype(np.int16) - 128) <= 10) & (np.abs(B.astype(np.int16) - 128) <= 12)
    return white.astype(np.uint8) * 255

def _candidate_boxes(bgr: np.ndarray,
                     min_area_ratio: float,
                     center_band: float,
                     aspect_max: float) -> List[Tuple[int,int,int,int]]:
    H, W = bgr.shape[:2]
    # Non-white regions → components
    non_white = cv2.bitwise_not(_white_mask_lab(bgr))
    # Morphology to fuse puzzle tiles and badges
    non_white = cv2.morphologyEx(non_white, cv2.MORPH_CLOSE, np.ones((17,17), np.uint8))
    non_white = cv2.morphologyEx(non_white, cv2.MORPH_OPEN,  np.ones((7,7),  np.uint8))
    contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total = H * W
    cx_img, cy_img = W*0.5, H*0.5
    band_w, band_h = W*(center_band/2), H*(center_band/2)

    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if (w*h)/total < min_area_ratio:
            continue
        ar = max(w,h) / max(1, min(w,h))
        if ar > aspect_max:  # filters out long trays/footers
            continue
        cx, cy = x + w/2, y + h/2
        if not (abs(cx - cx_img) <= band_w and abs(cy - cy_img) <= band_h):
            continue
        boxes.append((x,y,w,h))
    return boxes

def _grid_score(bgr: np.ndarray, box: Tuple[int,int,int,int]) -> float:
    """
    Score how 'grid-like' a box is:
      - Canny edges
      - HoughLinesP to count near-vertical & near-horizontal segments
      - score ~ (#vert * #horiz) normalized by area
    """
    x,y,w,h = box
    roi = bgr[y:y+h, x:x+w]
    if roi.size == 0: return 0.0

    # Downscale for speed, keep aspect
    scale = 800 / max(roi.shape[0], roi.shape[1])
    if scale < 1.0:
        roi_small = cv2.resize(roi, (int(roi.shape[1]*scale), int(roi.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    else:
        roi_small = roi

    gray = cv2.cvtColor(roi_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 40, 120)

    # Slight dilation to bridge dotted borders
    edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)

    # Line detection
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,
                            minLineLength=max(10, int(0.06*max(edges.shape))),
                            maxLineGap=int(0.03*max(edges.shape)))
    if lines is None:
        return 0.0

    v_cnt = 0
    h_cnt = 0
    for l in lines[:,0,:]:
        x1,y1,x2,y2 = l
        dx, dy = x2-x1, y2-y1
        ang = np.degrees(np.arctan2(dy, dx))
        ang = (ang + 180) % 180  # 0..180
        if min(abs(ang-0), abs(ang-180)) <= 10: h_cnt += 1     # near horizontal
        if abs(ang-90) <= 10:                                  # near vertical
            v_cnt += 1

    area_norm = (w*h) / 1e6  # soft normalization, avoids bias to huge boxes
    # Require both directions (a grid needs both)
    if v_cnt == 0 or h_cnt == 0:
        return 0.0

    return (v_cnt * h_cnt) / max(1e-3, area_norm)

def crop_puzzle_boards(
    image_path: str,
    *,
    mode: Literal["separate","union"] = "separate",
    min_area_ratio: float = 0.006,   # permissive enough for small phone boards
    center_band: float = 0.92,       # allow boards offset vertically/horizontally
    aspect_max: float = 3.8,         # filter out trays/footers
    keep_top_k: int = 4,             # up to 4 boards if ever present
    score_threshold: float = 8.0     # minimal grid score to accept as a board
) -> Tuple[List[np.ndarray], List[Tuple[int,int,int,int]]]:
    """
    Grid-aware multi-board cropping for NYT Pips screenshots.
    Returns list of RGB crops and their boxes in original coords.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    H, W = bgr.shape[:2]

    # 1) Find plausible non-white components near the center
    candidates = _candidate_boxes(bgr, min_area_ratio, center_band, aspect_max)
    if not candidates:
        # Fallback: loosen center/area a bit and retry once
        candidates = _candidate_boxes(bgr, min_area_ratio*0.6, 0.98, aspect_max+0.5)
    if not candidates:
        raise ValueError("No board-like regions found.")

    # 2) Score each by grid-ness
    scored = [(box, _grid_score(bgr, box)) for box in candidates]
    # Keep only those with meaningful grid score
    keep = [(box, sc) for (box, sc) in scored if sc >= score_threshold]
    if not keep:
        # Fallback: take top 2 by score anyway (handles very faint grids)
        keep = sorted(scored, key=lambda t: t[1], reverse=True)[:2]

    # Sort top candidates by score (desc), then by y,x for stable naming
    keep = sorted(keep, key=lambda t: (-t[1], t[0][1], t[0][0]))[:keep_top_k]

    # 3) Build crops
    def pad_box(x,y,w,h):
        pad = int(round(PAD_FRAC * max(W, H)))
        x1 = _clip(x - pad, 0, W-1)
        y1 = _clip(y - pad, 0, H-1)
        x2 = _clip(x + w + pad, 0, W)
        y2 = _clip(y + h + pad, 0, H)
        return x1,y1,x2,y2

    boxes = [b for (b, _) in keep]
    if mode == "union" and len(boxes) > 1:
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0]+b[2] for b in boxes)
        y2 = max(b[1]+b[3] for b in boxes)
        x1,y1,x2,y2 = pad_box(x1,y1,x2-x1,y2-y1)
        crop = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        return [crop], [(x1,y1,x2-x1,y2-y1)]

    crops = []
    out_boxes: List[Tuple[int,int,int,int]] = []
    for (x,y,w,h) in boxes:
        x1,y1,x2,y2 = pad_box(x,y,w,h)
        crops.append(cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        out_boxes.append((x1,y1,x2-x1,y2-y1))
    return crops, out_boxes


def detect_dominoes(
    image_path: str,
    *,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.02,   # Lowered from 0.05 to exclude puzzle boards better
    aspect_min: float = 1.2,
    aspect_max: float = 6.0,
    vertical_gap_threshold: float = 0.15,
    min_cluster_size: int = 2,
    exclude_top_fraction: float = 0.15
) -> List[Tuple[int,int,int,int]]:
    """
    Detect domino pieces using adaptive region finding.
    Works for both mobile and desktop layouts without hardcoded positions.
    
    Strategy:
    1. Find ALL non-white rectangular regions
    2. Filter for domino-shaped regions (elongated rectangles)
    3. Cluster spatially to separate domino tray from puzzle boards
    4. Return domino boxes sorted by position
    
    Args:
        image_path: Path to screenshot
        min_area_ratio: Minimum area relative to total (default: 0.001)
        max_area_ratio: Maximum area relative to total (default: 0.02, excludes boards)
        aspect_min: Min aspect ratio for dominoes (default: 1.2)
        aspect_max: Max aspect ratio for dominoes (default: 6.0)
        vertical_gap_threshold: Gap to split clusters, fraction of height (default: 0.15)
        min_cluster_size: Min dominoes in valid cluster (default: 2)
        exclude_top_fraction: Exclude regions in top fraction of image (default: 0.15)
    
    Returns:
        List of (x,y,w,h) boxes for each domino, sorted by position
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    H, W = bgr.shape[:2]
    
    # Find non-white regions (reuse same method as board detection)
    non_white = cv2.bitwise_not(_white_mask_lab(bgr))
    
    # Lighter morphology to preserve small dominoes
    # Using 3x3 close and 2x2 open instead of 7x7/3x3
    non_white = cv2.morphologyEx(non_white, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    non_white = cv2.morphologyEx(non_white, cv2.MORPH_OPEN,  np.ones((2,2), np.uint8))
    
    contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for domino-shaped candidates
    total_area = H * W
    candidates = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        area_ratio = area / total_area
        
        # Filter by area (not too small, not too large)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        
        # Filter out UI elements at top of screen
        vertical_center = (y + h / 2) / H
        if vertical_center < exclude_top_fraction:
            continue
        
        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio < aspect_min or aspect_ratio > aspect_max:
            continue
        
        vertical_pos = (y + h / 2) / H
        
        candidates.append({
            'bbox': (x, y, w, h),
            'aspect': aspect_ratio,
            'vertical_pos': vertical_pos
        })
    
    if not candidates:
        return []
    
    # Cluster candidates spatially by vertical position
    candidates_sorted = sorted(candidates, key=lambda c: c['vertical_pos'])
    clusters = []
    current_cluster = [candidates_sorted[0]]
    gap_threshold_px = H * vertical_gap_threshold
    
    for i in range(1, len(candidates_sorted)):
        prev = current_cluster[-1]
        curr = candidates_sorted[i]
        
        prev_bottom = prev['bbox'][1] + prev['bbox'][3]
        curr_top = curr['bbox'][1]
        vertical_gap = curr_top - prev_bottom
        
        if vertical_gap > gap_threshold_px:
            clusters.append(current_cluster)
            current_cluster = [curr]
        else:
            current_cluster.append(curr)
    
    clusters.append(current_cluster)
    
    # Find cluster that looks most like domino tray
    best_cluster = None
    best_score = -1
    
    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            continue
        
        avg_vertical_pos = sum(c['vertical_pos'] for c in cluster) / len(cluster)
        count = len(cluster)
        aspects = [c['aspect'] for c in cluster]
        aspect_std = np.std(aspects) if len(aspects) > 1 else 0
        aspect_mean = np.mean(aspects)
        
        aspect_score = 1.0 if 1.5 <= aspect_mean <= 4.0 else 0.5
        consistency_score = 1.0 / (1.0 + aspect_std)
        
        cluster_score = (
            avg_vertical_pos * 0.35 +
            min(count / 15.0, 1.0) * 0.35 +
            consistency_score * 0.15 +
            aspect_score * 0.15
        )
        
        if cluster_score > best_score:
            best_score = cluster_score
            best_cluster = cluster
    
    if best_cluster is None:
        return []
    
    # Extract and sort domino boxes
    domino_boxes = [c['bbox'] for c in best_cluster]
    domino_boxes.sort(key=lambda b: (b[1] // 40, b[0]))  # Sort by row, then column
    
    return domino_boxes