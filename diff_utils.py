import cv2
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class DiffConfig:
    structure_threshold: int = 200  # Pixels darker than this are "Structure"
    noise_kernel_size: int = 5      # For morphological cleanup
    iou_threshold: float = 0.3      # Keeping purely new items (30% overlap with change mask)
    # Note: Wall overlap might need a different heuristic than openings

def compute_change_mask(original_img: np.ndarray, target_img: np.ndarray, cfg: DiffConfig = DiffConfig()) -> np.ndarray:
    """
    Computes binary masks of "Added" and "Removed" Structure.
    Returns (added_mask, removed_mask).
    """
    # Ensure disjoint sizes don't crash us (resize target to original if needed, or vice versa)
    h, w = original_img.shape[:2]
    target_resized = cv2.resize(target_img, (w, h))

    # Convert to Grayscale
    orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    targ_gray = cv2.cvtColor(target_resized, cv2.COLOR_BGR2GRAY)

    # Define Structure (Dark pixels)
    # We want things that ARE structure in Target but were NOT structure in Original.
    # Structure Mask: 255 where dark, 0 where light
    is_structure_orig = (orig_gray < cfg.structure_threshold).astype(np.uint8) * 255
    is_structure_targ = (targ_gray < cfg.structure_threshold).astype(np.uint8) * 255

    # "New" = In Target AND NOT In Original
    # cv2.subtract handles saturation (0-255 clipped at 0), so (1 - 1 = 0), (1 - 0 = 1), (0 - 1 = 0)
    # We want: Target=1 (255), Original=0 (0) -> Result=255.
    added_structure = cv2.subtract(is_structure_targ, is_structure_orig)

    # Cleanup Noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.noise_kernel_size, cfg.noise_kernel_size))
    # Open: Erode then Dilate (Standard noise removal)
    cleaned = cv2.morphologyEx(added_structure, cv2.MORPH_OPEN, kernel)
    # Dilate slightly to catch edges of walls that might have shifted slightly/anti-aliasing
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    # Removed Structure (Old=Dark, New=Light)
    removed_structure = cv2.subtract(is_structure_orig, is_structure_targ)
    removed_cleaned = cv2.morphologyEx(removed_structure, cv2.MORPH_OPEN, kernel)
    removed_cleaned = cv2.dilate(removed_cleaned, kernel, iterations=1)

    return cleaned, removed_cleaned

def is_rect_in_mask(x: float, y: float, w: float, h: float, mask: np.ndarray, scale: float) -> bool:
    """
    Checks if a rectangle (furniture/room) significantly overlaps with the mask.
    Coordinates are in METERS. Scale is Meters-Per-Pixel.
    """
    h_img, w_img = mask.shape[:2]
    
    # Convert to pixels (Center X/Y)
    cx_px = x / scale
    cy_px = h_img - (y / scale) # Inverted Y for CV2 (Top-Left) vs CAD (Bottom-Left) usually? 
    # WAIT: mask2dsl usually uses standard image coords (0,0 top-left).
    # If the input JSON coordinates are from mask2dsl output, they might already be in a specific system.
    # Standard mask2dsl output:
    # Walls: Start/End in Metres (Y inverted typically if coming from pixel mapping).
    # Let's assume standard image top-left origin for logic, but need to check mask2dsl coordinate system.
    # mask2dsl.py: Line 219: return (pt_m[0] / cfg.meters_per_pixel, float(h_img) - pt_m[1] / cfg.meters_per_pixel)
    # So Y is inverted (Cartesian/World up).
    
    px_x = int(x / scale)
    px_y = int(h_img - (y / scale))
    px_w = int(w / scale)
    px_l = int(h / scale) # Length/Height

    # Bounding Box (Top-Left in Image Coords)
    # Furniture (x,y) is usually center? mask2dsl Furniture class says x,y. Let's assume center for now or check.
    # mask2dsl logic for furniture isn't fully shown in the snippet, but typical usage:
    # If x,y is center:
    x1 = max(0, int(px_x - px_w // 2))
    y1 = max(0, int(px_y - px_l // 2))
    x2 = min(w_img, int(px_x + px_w // 2))
    y2 = min(h_img, int(px_y + px_l // 2))
    
    if x2 <= x1 or y2 <= y1: return False
    
    # Extract ROI
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0: return False
    
    # Check overlap percentage
    non_zero = cv2.countNonZero(roi)
    area = roi.size
    ratio = non_zero / float(area)
    
    return ratio > 0.3 # Threshold

def filter_dsl(dsl_data: Dict[str, Any], added_mask: np.ndarray, removed_mask: np.ndarray, meters_per_pixel: float) -> Dict[str, Any]:
    """
    Filters the full DSL JSON, keeping only entities intersecting the change_mask.
    """
    # 1. Furniture
    # 2. Walls
    # 3. Openings (Nested in Walls usually, or linked by ID)
    
    h_img, w_img = added_mask.shape[:2]
    
    filtered = {
        "walls": [],
        "furniture": []
    }
    
    # Helper to map World (Meters) to Pixel (Image)
    def world_to_px(wx, wy):
        px = int(wx / meters_per_pixel)
        py = int(h_img - (wy / meters_per_pixel))
        return px, py

    # --- WALLS & OPENINGS ---
    for wall in dsl_data.get("walls", []):
        w_start = wall["start"]
        w_end = wall["end"]
        
        # Check Wall Overlap (Added Structure)
        # Using simplified "start/end" from JSON (which might be dict or list based on mask2dsl version)
        # mask2dsl output JSON uses dict: "start": {"x":..., "y":...}
        sx = w_start["x"] if isinstance(w_start, dict) else w_start[0]
        sy = w_start["y"] if isinstance(w_start, dict) else w_start[1]
        ex = w_end["x"] if isinstance(w_end, dict) else w_end[0]
        ey = w_end["y"] if isinstance(w_end, dict) else w_end[1]
        
        # Wall Mask Check
        wall_mask = np.zeros_like(added_mask)
        p1 = world_to_px(sx, sy)
        p2 = world_to_px(ex, ey)
        thickness = wall.get("thickness", 0.2)
        th_px = max(1, int(thickness / meters_per_pixel))
        
        cv2.line(wall_mask, p1, p2, 255, thickness=th_px)
        overlap = cv2.bitwise_and(wall_mask, added_mask)
        wall_px = cv2.countNonZero(wall_mask)
        
        wall_is_new = (wall_px > 0) and (cv2.countNonZero(overlap) / wall_px > 0.3) # 30% overlap
        
        # Process Openings (Nested)
        kept_openings = []
        original_openings = wall.get("openings", [])
        
        # Vector direction for locating openings
        vx, vy = ex - sx, ey - sy
        len_m = np.hypot(vx, vy)
        dx, dy = (vx/len_m, vy/len_m) if len_m > 1e-4 else (0,0)

        for opening in original_openings:
            # Check if Opening is New (Removed Structure)
            # Locate opening center
            t = opening.get("t", 0.0) # opening["t"] in JSON
            # mask2dsl emits "t", not "at".
            
            cx_m = sx + dx * t
            cy_m = sy + dy * t
            
            c_px = world_to_px(cx_m, cy_m)
            
            # ROI Check in REMOVED mask (Void)
            box_r = 5
            x1 = max(0, c_px[0]-box_r); x2 = min(w_img, c_px[0]+box_r)
            y1 = max(0, c_px[1]-box_r); y2 = min(h_img, c_px[1]+box_r)
            
            roi = removed_mask[y1:y2, x1:x2]
            opening_is_new_void = cv2.countNonZero(roi) > 0
            
            # Condition to keep opening:
            # 1. Wall is New (keep all openings in it)
            # 2. Opening is explicitly New Void (added to old wall)
            if wall_is_new or opening_is_new_void:
                kept_openings.append(opening)
        
        # Condition to keep wall:
        # 1. Wall is New
        # 2. Wall has New Openings (we need the wall shell to host them)
        if wall_is_new or len(kept_openings) > 0:
            # Deep copy to avoid mutating original if passed again
            new_wall = wall.copy()
            new_wall["openings"] = kept_openings
            filtered["walls"].append(new_wall)

    # --- FURNITURE ---
    for furn in dsl_data.get("furniture", []):
        fx = furn["x"]
        fy = furn["y"]
        fw = furn["width"]
        fl = furn["length"]
        
        if is_rect_in_mask(fx, fy, fw, fl, added_mask, meters_per_pixel):
            filtered["furniture"].append(furn)

    return filtered
