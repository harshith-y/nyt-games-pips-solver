# Vision/cell_grid.py
# Robust cell lattice detection for NYT "Pips" - ADAPTIVE VERSION
# Automatically adapts to different screen sizes (phone, tablet, desktop, high-DPI)

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

# --------------------------- Models ---------------------------

@dataclass
class Cell:
    id: int
    row: int
    col: int
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    center: Tuple[int, int]
    section: Optional[int] = None

@dataclass
class GridResult:
    pitch: float
    offset: Tuple[int, int]           # (dx, dy)
    cells: List[Cell]
    debug: Dict[str, float]

@dataclass
class CellDetectConfig:
    # ADAPTIVE PITCH DETECTION - relative to image size
    # These ratios work better for phone screenshots
    pitch_min_ratio: float = 0.10     # min pitch = 10% of smaller dimension (was 8%)
    pitch_max_ratio: float = 0.35     # max pitch = 35% of smaller dimension (was 25%)
    
    # Absolute bounds as safety rails
    absolute_pitch_min: int = 50      # never go below 50px (was 40px)
    absolute_pitch_max: int = 500     # never go above 500px (was 400px)
    
    # Practical bounds for typical Pips boards (these override adaptive if tighter)
    # Based on empirical observation: cells are typically 90-200px across devices
    practical_pitch_min: int = 90     # HARD LIMIT: smallest cell size (excludes sub-cell artifacts like 66px, 72px)
    practical_pitch_max: int = 200    # HARD LIMIT: typical largest cell size

    # Core mask erosion (px; removes border bleed)
    core_erode_px: int = 2

    # Inner validation window
    core_frac: float = 0.60

    # Acceptance thresholds
    core_palette_thresh: float = 0.30
    overlap_thresh: float = 0.65
    void_max: float = 0.10

    # Phase search (fine)
    fine_phase_steps: Tuple[int, ...] = (4, 2, 1)

    # Placeable mask extras
    hsv_s_min: int = 22
    hsv_v_min: int = 115
    use_hsv_fallback_in_placeable: bool = True

    # Palette ΔE thresholds
    delta_e_core: float = 20.0
    delta_e_placeable: float = 22.0

# --------------------------- Palette helpers ---------------------------

def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.strip().lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _palette_to_lab(hex_list: List[str]) -> np.ndarray:
    if not hex_list:
        return np.zeros((0, 3), np.float32)
    rgb = np.array([_hex_to_rgb(h) for h in hex_list], dtype=np.uint8).reshape(-1, 1, 3)
    bgr = rgb[:, :, ::-1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return lab

def _rgb_to_lab_img(img_rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

def _delta_e_min(lab_img: np.ndarray, lab_palette: np.ndarray) -> np.ndarray:
    if lab_palette.size == 0:
        return np.full(lab_img.shape[:2], np.inf, dtype=np.float32)
    H, W, _ = lab_img.shape
    P = lab_palette.shape[0]
    img = lab_img.reshape(-1, 1, 3).astype(np.float32)
    pal = lab_palette.reshape(1, P, 3).astype(np.float32)
    de2 = np.sum((img - pal) ** 2, axis=2)
    de_min = np.sqrt(de2.min(axis=1)).reshape(H, W).astype(np.float32)
    return de_min

# --------------------------- Masks ---------------------------

def _core_mask(img_rgb: np.ndarray,
               lab_inc: np.ndarray,
               lab_exc: np.ndarray,
               cfg: CellDetectConfig) -> np.ndarray:
    lab = _rgb_to_lab_img(img_rgb)
    de_inc = _delta_e_min(lab, lab_inc)
    m = (de_inc <= cfg.delta_e_core).astype(np.uint8)
    if lab_exc.size:
        de_exc = _delta_e_min(lab, lab_exc)
        m = (m & (de_exc > cfg.delta_e_core).astype(np.uint8)).astype(np.uint8)
    if cfg.core_erode_px > 0:
        k = max(1, int(cfg.core_erode_px))
        m = cv2.erode(m, np.ones((k, k), np.uint8), iterations=1)
    return m

def _placeable_mask(img_rgb: np.ndarray,
                    lab_inc: np.ndarray,
                    lab_exc: np.ndarray,
                    cfg: CellDetectConfig) -> np.ndarray:
    lab = _rgb_to_lab_img(img_rgb)
    de_inc = _delta_e_min(lab, lab_inc)
    pal_hit = (de_inc <= cfg.delta_e_placeable).astype(np.uint8) * 255
    if lab_exc.size:
        de_exc = _delta_e_min(lab, lab_exc)
        pal_hit = pal_hit & (de_exc > cfg.delta_e_placeable).astype(np.uint8) * 255
    if cfg.use_hsv_fallback_in_placeable:
        hsv = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        s_ok = (hsv[:, :, 1] >= cfg.hsv_s_min).astype(np.uint8) * 255
        v_ok = (hsv[:, :, 2] >= cfg.hsv_v_min).astype(np.uint8) * 255
        pal_hit = cv2.bitwise_or(pal_hit, cv2.bitwise_and(s_ok, v_ok))
    m = cv2.morphologyEx(pal_hit, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((7, 7),  np.uint8))
    return (m > 0).astype(np.uint8)

def _void_mask_white(img_rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    white = (L >= 215) & (np.abs(A.astype(np.int16) - 128) <= 10) & (np.abs(B.astype(np.int16) - 128) <= 12)
    return white.astype(np.uint8)

# --------------------------- Adaptive pitch bounds ---------------------------

def _compute_adaptive_pitch_bounds(img_shape: Tuple[int, int], cfg: CellDetectConfig) -> Tuple[int, int]:
    """
    Compute adaptive pitch bounds based on image dimensions.
    Returns (pitch_min, pitch_max) appropriate for this image size.
    """
    H, W = img_shape[:2]
    min_dim = min(H, W)
    
    # Calculate relative bounds
    pitch_min = int(round(min_dim * cfg.pitch_min_ratio))
    pitch_max = int(round(min_dim * cfg.pitch_max_ratio))
    
    # Apply absolute safety bounds
    pitch_min = max(cfg.absolute_pitch_min, pitch_min)
    pitch_max = min(cfg.absolute_pitch_max, pitch_max)
    
    # Apply practical bounds (these are tighter, based on typical Pips cells)
    # Use the tighter of adaptive vs practical bounds
    pitch_min = max(pitch_min, cfg.practical_pitch_min)
    pitch_max = min(pitch_max, cfg.practical_pitch_max)
    
    # Ensure min < max
    if pitch_min >= pitch_max:
        pitch_max = pitch_min + 20
    
    return pitch_min, pitch_max

# --------------------------- Border-based pitch (de-dashed) ---------------------------

def _border_edge_map(img_rgb: np.ndarray, placeable01: np.ndarray) -> np.ndarray:
    """
    Internal grid/border edges, outer frame suppressed, dashes bridged → solid lines.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 40, 120)

    inner = cv2.erode((placeable01 * 255).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    grad  = cv2.morphologyEx(inner, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))

    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8))

    edges = cv2.bitwise_and(canny, grad)

    edges = cv2.blur(edges, (9, 1))
    edges = cv2.blur(edges, (1, 9))
    return edges

def _autocorr_first_peak(v: np.ndarray, lo_frac=0.08, hi_frac=0.5, min_abs=None) -> float:
    """
    Find first autocorrelation peak, ignoring tiny periods (like dash spacing).
    lo_frac=0.08 means ignore periods < 8% of image dimension (filters out dashes).
    min_abs allows setting an absolute minimum search position.
    """
    v = v.astype(np.float32)
    v = v - v.mean()
    ac = np.correlate(v, v, mode="full")[len(v)-1:]
    ac[:8] = 0
    # CRITICAL: Start search at 8% to skip dash spacing (~20-80px on typical boards)
    lo = max(15, int(len(v)*lo_frac))
    if min_abs is not None:
        lo = max(lo, min_abs)
    hi = max(lo+1, int(len(v)*hi_frac))
    if hi <= lo:
        return 0.0
    k = int(np.argmax(ac[lo:hi]) + lo)
    return float(k)

def _autocorr_multiple_scales(v: np.ndarray) -> List[float]:
    """
    Try autocorr at multiple scales to find both small patterns (dashes) and large patterns (cells).
    Returns list of candidate periods found at different scales.
    """
    candidates = []
    
    # Scale 1: Skip tiny patterns (find cells directly)
    p1 = _autocorr_first_peak(v, lo_frac=0.08, hi_frac=0.5)
    if p1 > 0:
        candidates.append(p1)
    
    # Scale 2: Larger patterns only (for very large cells)
    p2 = _autocorr_first_peak(v, lo_frac=0.15, hi_frac=0.5)
    if p2 > 0 and abs(p2 - p1) > 10:  # Different from p1
        candidates.append(p2)
    
    # Scale 3: Medium patterns (backup)
    p3 = _autocorr_first_peak(v, lo_frac=0.05, hi_frac=0.5)
    if p3 > 0 and all(abs(p3 - c) > 10 for c in candidates):
        candidates.append(p3)
    
    return candidates

def _candidate_pitches_from_borders(borders: np.ndarray, pitch_min: int, pitch_max: int) -> List[int]:
    """
    Generate candidate pitches from border autocorrelation.
    Now uses adaptive bounds instead of hardcoded values.
    """
    vx = borders.sum(axis=0).astype(np.float32)
    vy = borders.sum(axis=1).astype(np.float32)

    # Try multiple scales to find both dashes and cells
    px_candidates = _autocorr_multiple_scales(vx)
    py_candidates = _autocorr_multiple_scales(vy)

    # DEBUG: Print what autocorrelation found
    print(f"\n{'='*70}")
    print(f"DEBUG: Autocorrelation Results (Multi-Scale)")
    print(f"{'='*70}")
    print(f"  X-axis peaks: {[f'{p:.1f}px' for p in px_candidates]}")
    print(f"  Y-axis peaks: {[f'{p:.1f}px' for p in py_candidates]}")
    print(f"  Pitch bounds: {pitch_min}-{pitch_max}px")

    # Combine all peaks as seeds
    seeds = [p for p in (px_candidates + py_candidates) if p > 0]
    
    if not seeds:
        H, W = borders.shape
        rough = max(pitch_min, min(pitch_max, max(H, W)//6))
        seeds = [rough]
        print(f"  No peaks found, using fallback: {rough}px")

    print(f"  All seeds for multiplication: {[f'{s:.1f}px' for s in seeds]}")

    cands: List[int] = []
    for p in seeds:
        # EXPANDED multipliers to cover cases where autocorr finds dash spacing
        # Includes up to 15.0x for fine dash spacing (6-7 dashes per cell edge)
        for mul in (0.5, 2/3, 3/4, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0):
            cp = int(round(p * mul))
            if pitch_min <= cp <= pitch_max:
                cands.append(cp)

    cands = sorted(list({int(c) for c in cands if c >= pitch_min}))
    
    # SAFETY: If we have very few candidates, add evenly-spaced samples across the range
    # This ensures we explore the full pitch range even if autocorr fails
    if len(cands) < 8:
        print(f"  WARNING: Only {len(cands)} candidates, adding range samples...")
        step = (pitch_max - pitch_min) // 8
        for i in range(9):
            sample = pitch_min + i * step
            if pitch_min <= sample <= pitch_max:
                cands.append(sample)
        cands = sorted(list(set(cands)))
    
    print(f"  Generated {len(cands)} candidates: {cands[:20]}{'...' if len(cands) > 20 else ''}")
    print(f"{'='*70}\n")
    
    return cands[:25]  # Increased to allow more exploration

# --------------------------- Validation & phase search ---------------------------

def _valid_cells_for_pitch(board_rgb: np.ndarray,
                           pitch: int,
                           dx: int,
                           dy: int,
                           core01: np.ndarray,
                           placeable01: np.ndarray,
                           void01: np.ndarray,
                           cfg: CellDetectConfig) -> Tuple[int, List[Cell], Dict[str, float]]:
    H, W = board_rgb.shape[:2]
    s = int(pitch)
    inner = int(round((1.0 - cfg.core_frac) * 0.5 * s))

    cells: List[Cell] = []
    gid = 0
    valid = 0

    rows = int((H - dy) // s) + 1
    cols = int((W - dx) // s) + 1

    for r in range(rows):
        for c in range(cols):
            x = dx + c*s
            y = dy + r*s
            if x+s > W or y+s > H:
                continue

            x1i, x2i = x+inner, x+s-inner
            y1i, y2i = y+inner, y+s-inner
            if x2i <= x1i or y2i <= y1i:
                continue

            core_win = core01[y1i:y2i, x1i:x2i]
            plc_win  = placeable01[y:y+s, x:x+s]
            void_win = void01[y1i:y2i, x1i:x2i]

            if core_win.size == 0:
                continue

            core_ratio = float(core_win.mean())
            overlap    = float(plc_win.mean()) if plc_win.size else 0.0
            void_ratio = float(void_win.mean()) if void_win.size else 0.0

            ok = (core_ratio >= cfg.core_palette_thresh) and (overlap >= cfg.overlap_thresh) and (void_ratio <= cfg.void_max)
            if ok:
                cells.append(Cell(
                    id=gid, row=r, col=c,
                    bbox=(int(x), int(y), int(s), int(s)),
                    center=(int(x + s//2), int(y + s//2)),
                    section=None
                ))
                gid += 1
                valid += 1

    dbg = {"rows": rows, "cols": cols, "found": len(cells)}
    return valid, cells, dbg

def _phase_search(board_rgb: np.ndarray,
                  pitch: int,
                  core01: np.ndarray,
                  placeable01: np.ndarray,
                  void01: np.ndarray,
                  cfg: CellDetectConfig) -> Tuple[Tuple[int,int], List[Cell], Dict[str,float]]:
    p = int(pitch)
    quarters = [0, p//4, p//2, (3*p)//4]

    best = None
    for dy0 in quarters:
        for dx0 in quarters:
            v, cells, _ = _valid_cells_for_pitch(board_rgb, p, dx0, dy0, core01, placeable01, void01, cfg)
            sc = v * p
            pack = (sc, dx0, dy0, cells)
            if (best is None) or (sc > best[0]):
                best = pack

    _, dx, dy, _cells = best
    best_f = best

    for step in cfg.fine_phase_steps:
        cand = []
        for ddy in range(-3, 4, step):
            for ddx in range(-3, 4, step):
                dx1 = (dx + ddx) % p
                dy1 = (dy + ddy) % p
                v, cells, _ = _valid_cells_for_pitch(board_rgb, p, dx1, dy1, core01, placeable01, void01, cfg)
                sc = v * p
                cand.append((sc, dx1, dy1, cells))
        best_f = max(cand + [best_f], key=lambda t: t[0])
        dx, dy = best_f[1], best_f[2]

    return (best_f[1], best_f[2]), best_f[3], {"phase_dx": best_f[1], "phase_dy": best_f[2], "phase_score": best_f[0]}

# --------------------------- Public API ---------------------------

def detect_cells(
    board_rgb: np.ndarray,
    include_palette_hex: List[str],
    cfg: Optional[CellDetectConfig] = None,
    exclude_palette_hex: Optional[List[str]] = None
) -> GridResult:
    """
    Detect placeable cells using a manual palette (ΔE in LAB) and border-based pitch.
    NOW WITH ADAPTIVE PITCH DETECTION - automatically adapts to different screen sizes.
    """
    if cfg is None:
        cfg = CellDetectConfig()

    # ADAPTIVE: Compute pitch bounds based on image size
    pitch_min, pitch_max = _compute_adaptive_pitch_bounds(board_rgb.shape, cfg)

    # Palettes
    lab_inc = _palette_to_lab(include_palette_hex)
    lab_exc = _palette_to_lab(exclude_palette_hex or [])

    # Masks
    core01       = _core_mask(board_rgb, lab_inc, lab_exc, cfg)
    placeable01  = _placeable_mask(board_rgb, lab_inc, lab_exc, cfg)
    void01       = _void_mask_white(board_rgb)

    # Pitch from de-dashed border edges (now with adaptive bounds)
    borders = _border_edge_map(board_rgb, placeable01)
    cand_pitches = _candidate_pitches_from_borders(borders, pitch_min, pitch_max)
    
    # Extract autocorrelation peaks for scoring bonus
    # These are the "real" periodicities detected in the image
    H, W = board_rgb.shape[:2]
    col_sum = np.sum(borders, axis=0, dtype=np.float32)
    row_sum = np.sum(borders, axis=1, dtype=np.float32)
    
    peaks_x = _autocorr_multiple_scales(col_sum)
    peaks_y = _autocorr_multiple_scales(row_sum)
    
    # CRITICAL FIX: Expand bounds if autocorr found strong peaks outside the range
    # This handles cases where adaptive bounds were too conservative
    # BUT: Only expand for reasonable peaks (not image-width artifacts)
    # AND: Never exceed practical_pitch_max (hard limit)
    all_peaks = peaks_x + peaks_y
    if all_peaks:
        max_peak = max(all_peaks)
        # Only expand if peak is reasonable relative to image dimensions
        # A cell pitch should be < 50% of the smaller dimension
        max_reasonable_pitch = min(H, W) * 0.5
        
        # CRITICAL: Never expand beyond practical_pitch_max (hard limit)
        if max_peak > pitch_max and max_peak <= max_reasonable_pitch and max_peak <= cfg.practical_pitch_max:
            print(f"  WARNING: Autocorr found peak {max_peak:.0f}px > pitch_max {pitch_max}px")
            print(f"  Expanding pitch_max to {int(max_peak * 1.1)}px")
            pitch_max = int(min(max_peak * 1.1, cfg.practical_pitch_max))
            # Regenerate candidates with expanded bounds
            cand_pitches = _candidate_pitches_from_borders(borders, pitch_min, pitch_max)
        elif max_peak > cfg.practical_pitch_max:
            print(f"  NOTE: Ignoring peak {max_peak:.0f}px (exceeds practical_pitch_max {cfg.practical_pitch_max}px)")
        elif max_peak > max_reasonable_pitch:
            print(f"  NOTE: Ignoring unreasonable peak {max_peak:.0f}px (> 50% of min dimension)")
    
    autocorr_peaks = [p for p in peaks_x + peaks_y if pitch_min <= p <= pitch_max]

    # Evaluate each candidate: phase search + valid cells; pick best
    best_pack = None
    pitch_scores = []  # For debugging
    
    # Calculate middle of pitch range as reference point
    pitch_mid = (pitch_min + pitch_max) / 2
    
    for p in cand_pitches:
        (dx, dy), cells, phase_dbg = _phase_search(board_rgb, p, core01, placeable01, void01, cfg)
        
        cell_count = len(cells)
        base_score = cell_count * p
        
        # UNIVERSAL SCORING WITH PITCH PREFERENCE:
        # Penalize pitches significantly below OR above the middle - likely artifacts
        # Small pitches = sub-cell features, Large pitches = multi-cell spacing
        
        if p < pitch_mid * 0.70:  # More than 30% below middle
            # Very likely sub-cells - very strong penalty
            penalty = 0.35
        elif p < pitch_mid * 0.85:  # 15-30% below middle
            # Likely too small - strong penalty
            penalty = 0.65
        elif p < pitch_mid * 0.95:  # 5-15% below middle
            # Slightly small - mild penalty
            penalty = 0.85
        elif p > pitch_mid * 1.30:  # More than 30% above middle
            # Likely multi-cell spacing - strong penalty
            penalty = 0.60
        elif p > pitch_mid * 1.15:  # 20-30% above middle
            # Possibly too large - mild penalty
            penalty = 0.85
        else:
            # Reasonable range - no penalty
            penalty = 1.0
        
        score = base_score * penalty
        
        # NEW: Boost score if pitch is close to an autocorrelation peak
        # This helps distinguish true cell pitch from sub-cellular features
        peak_bonus = 1.0
        peak_annotation = ""
        if autocorr_peaks:
            # Calculate distance to nearest peak
            min_distance_to_peak = min(abs(p - peak) for peak in autocorr_peaks)
            
            # Tight tolerance - only boost very close matches
            tolerance_tight = 8   # Within 8px = very strong match
            tolerance_loose = 15  # Within 15px = likely related
            
            if min_distance_to_peak < tolerance_tight:
                peak_bonus = 1.6  # Moderate boost for exact matches
                peak_annotation = "×1.6 PEAK"
            elif min_distance_to_peak < tolerance_loose:
                peak_bonus = 1.2  # Small boost for nearby
                peak_annotation = "×1.2 peak"
        else:
            # NO autocorr peaks found - use upper-half bias as tiebreaker
            # This helps when autocorr fails: prefer larger pitches (less likely to be sub-cells)
            if p > pitch_mid:
                peak_bonus = 1.1  # Small bias toward upper half
                peak_annotation = "×1.1 upper"
        
        score = score * peak_bonus
        
        pack = (score, p, dx, dy, cells, phase_dbg)
        pitch_scores.append((p, cell_count, base_score, score, penalty, peak_bonus, peak_annotation))
        if (best_pack is None) or (score > best_pack[0]):
            best_pack = pack
    
    # DEBUG: Show all pitch evaluations
    print(f"\n{'='*70}")
    print(f"DEBUG: Pitch Evaluation Results")
    print(f"{'='*70}")
    print(f"  Pitch range: {pitch_min}-{pitch_max}px (mid: {pitch_mid:.0f}px)")
    print(f"  Autocorr peaks in range: {[f'{p:.0f}px' for p in autocorr_peaks] if autocorr_peaks else 'none'}")
    print(f"  Lower penalties: <{pitch_mid*0.70:.0f}px (×0.35), <{pitch_mid*0.85:.0f}px (×0.65), <{pitch_mid*0.95:.0f}px (×0.85)")
    print(f"  Upper penalties: >{pitch_mid*1.30:.0f}px (×0.60), >{pitch_mid*1.20:.0f}px (×0.85)")
    print(f"  Evaluated {len(pitch_scores)} candidate pitches:")
    for pitch_val, cell_count, base, final, pen, pk_bonus, pk_ann in sorted(pitch_scores, key=lambda x: -x[3])[:10]:
        marker = " ← WINNER" if pitch_val == best_pack[1] else ""
        annotations = []
        if pen < 0.99:
            annotations.append(f"×{pen:.2f}")
        if pk_ann:
            annotations.append(pk_ann)
        ann_str = f" ({', '.join(annotations)})" if annotations else ""
        print(f"    Pitch {pitch_val:3d}px: {cell_count:2d} cells, base {base:5d}, final {int(final):5d}{ann_str}{marker}")
    print(f"{'='*70}\n")

    score, pitch, dx, dy, cells, phase_dbg = best_pack

    dbg = {
        "chosen_pitch": int(pitch),
        "adaptive_pitch_min": int(pitch_min),
        "adaptive_pitch_max": int(pitch_max),
        "offset_dx": int(dx),
        "offset_dy": int(dy),
        "cells": len(cells),
        "phase_score": phase_dbg.get("phase_score", 0),
        "H": int(board_rgb.shape[0]), 
        "W": int(board_rgb.shape[1]),
        **asdict(cfg)
    }

    return GridResult(
        pitch=float(pitch),
        offset=(int(dx), int(dy)),
        cells=cells,
        debug=dbg
    )