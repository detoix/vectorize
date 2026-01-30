import argparse
import os
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import networkx as nx
# SAFETY: Preserving original imports for reference/revert capability
# from skimage.morphology import skeletonize
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, substring
# SAFETY: Preserving original imports for reference/revert capability
# from scipy.spatial import cKDTree
# from scipy.ndimage import map_coordinates

# --- REPLACEMENT ALGORITHMS (Vercel Optimization) ---

def skeletonize_custom(image):
    """
    Zhang-Suen thinning algorithm implementation to replace skimage.morphology.skeletonize.
    Operates on a binary image (0/1 or 0/255). Returns a binary mask (0/1) of the skeleton.
    """
    # Ensure image is binary uint8 (0 or 1)
    img = image.astype(np.uint8)
    if img.max() > 1:
        img = (img > 0).astype(np.uint8)
    
    # Pre-allocate for speed
    skel = np.zeros(img.shape, np.uint8)
    curr = img.copy()
    
    # Kernel for neighbor encoding (P2..P9 weights)
    # P9 P2 P3
    # P8    P4
    # P7 P6 P5
    kernel = np.array([[128, 1, 2], [64, 0, 4], [32, 16, 8]], dtype=np.float32)
    
    # Pre-compute LUTs for Zhang-Suen
    # Note: We can cache this or compute once. Since this function is called once per request, computing is fast enough.
    lut_s1 = np.array([False]*256)
    lut_s2 = np.array([False]*256)
    
    for i in range(256):
        # Decode neighbors [P2, P3, ..., P9]
        # P2=bit0, P3=bit1...
        neighbors = [(i >> k) & 1 for k in range(8)]
        # neighbors indices: 0->P2, 1->P3, ... 7->P9
        
        # B: Number of non-zero neighbors
        B = sum(neighbors)
        if B < 2 or B > 6: continue
        
        # A: Number of 0->1 transitions in sequence P2,P3,P4,P5,P6,P7,P8,P9,P2
        seq = neighbors + [neighbors[0]]
        A = 0
        for k in range(8):
            if seq[k] == 0 and seq[k+1] == 1:
                A += 1
        if A != 1: continue
        
        P2, P3, P4, P5, P6, P7, P8, P9 = neighbors
        
        # Step 1: P2*P4*P6=0 && P4*P6*P8=0
        if (P2 * P4 * P6 == 0) and (P4 * P6 * P8 == 0):
            lut_s1[i] = True
            
        # Step 2: P2*P4*P8=0 && P2*P6*P8=0
        if (P2 * P4 * P8 == 0) and (P2 * P6 * P8 == 0):
            lut_s2[i] = True
    
    # Iterative Thinning
    skel = img.copy()
    while True:
        # Step 1
        neighbors_weighted = cv2.filter2D(skel, cv2.CV_32F, kernel)
        neighbors_int = neighbors_weighted.astype(np.uint8)
        
        # Identify pixels to delete
        # Since we only care about FG pixels, mask with skel
        to_check = (skel > 0)
        should_delete = lut_s1[neighbors_int] & to_check
        
        if not np.any(should_delete):
            break
            
        skel[should_delete] = 0
        
        # Step 2
        neighbors_weighted = cv2.filter2D(skel, cv2.CV_32F, kernel)
        neighbors_int = neighbors_weighted.astype(np.uint8)
        
        to_check = (skel > 0)
        should_delete = lut_s2[neighbors_int] & to_check
        
        if not np.any(should_delete):
            break
            
        skel[should_delete] = 0

    return skel

class SimpleKDTree:
    """
    Replacement for scipy.spatial.cKDTree.
    Uses brute-force for small N or simple spatial hashing?
    For wall endpoints (N < 1000), brute force is likely faster than building a tree in pure Python.
    """
    def __init__(self, data):
        self.data = np.asarray(data)

    def query(self, x, k=1):
        # x can be a single point or array of points
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :] # (1, D)
            return_single = True
        else:
            return_single = False
            
        # Brute force: dist matrix
        # data: (N, D), x: (M, D)
        # diff: (M, N, D) -> uses too much memory if M,N large.
        # Loop over M is safer.
        dists = []
        indices = []
        
        for pt in x:
            # pt: (D,)
            # d: (N,)
            diff = self.data - pt
            d_sq = np.sum(diff**2, axis=1)
            # Find k smallest
            if k == 1:
                idx = np.argmin(d_sq)
                dists.append(np.sqrt(d_sq[idx]))
                indices.append(idx)
            else:
                # limited support for k>1 if needed
                part_idx = np.argpartition(d_sq, k)[:k]
                # sort locally
                part_d_sq = d_sq[part_idx]
                sort_order = np.argsort(part_d_sq)
                final_idx = part_idx[sort_order]
                final_d = np.sqrt(part_d_sq[sort_order])
                dists.append(final_d)
                indices.append(final_idx)
                
        if return_single:
            return dists[0], indices[0]
        return np.array(dists), np.array(indices)
        
    def query_pairs(self, r):
        """
        Find all pairs with distance <= r.
        Returns set of (i, j) with i < j.
        """
        # Brute force pairs. (N*(N-1))/2
        pairs = set()
        n = len(self.data)
        r_sq = r * r
        for i in range(n):
            for j in range(i + 1, n):
                d_sq = np.sum((self.data[i] - self.data[j])**2)
                if d_sq <= r_sq:
                    pairs.add((i, j))
        return pairs

def map_coordinates_custom(input_array, coordinates, order=1, mode='constant', cval=0.0):
    """
    Replacement for scipy.ndimage.map_coordinates.
    Supports only order=1 (linear) and simple 2D arrays.
    coordinates: shape (2, N) -> (y, x)
    """
    h, w = input_array.shape[:2]
    ys = coordinates[0, :]
    xs = coordinates[1, :]
    
    # We can use cv2.remap. 
    # remap expects maps of shape (H_out, W_out, 1). 
    # Here we have unstructured points. 
    # So we construct a 1xN map?
    
    map_x = xs.astype(np.float32).reshape(1, -1)
    map_y = ys.astype(np.float32).reshape(1, -1)
    
    # borderMode mapping
    b_mode = cv2.BORDER_CONSTANT
    if mode == 'nearest': b_mode = cv2.BORDER_REPLICATE # approx
    # scipy 'constant' -> cv2 BORDER_CONSTANT
    
    # interp
    interp = cv2.INTER_LINEAR
    if order == 0: interp = cv2.INTER_NEAREST
    
    # cv2.remap(src, map1, map2, interpolation, borderMode, borderValue)
    out = cv2.remap(input_array, map_x, map_y, interp, borderMode=b_mode, borderValue=cval)
    # out shape will be (1, N)
    return out.reshape(-1)


def _resample_thickness_from_dist_map(
    walls: List["Wall"],
    dist_map: np.ndarray,
    h_img: int,
    cfg: "Config",
) -> None:
    """
    Sample thickness once after axis_offset is known.

    Rationale: For walls with significant axis_offset, the snapped wall axis can run near the edge of
    the wall mask, producing artificially small distance-transform values. Sampling along the
    "original/baseline" axis (snapped + n_left * axis_offset) matches the wall mask centerline better.
    """

    def to_px_float(pt_m: Tuple[float, float]) -> Tuple[float, float]:
        return (pt_m[0] / float(cfg.meters_per_pixel), float(h_img) - pt_m[1] / float(cfg.meters_per_pixel))

    for w in walls:
        dx = w.end[0] - w.start[0]
        dy = w.end[1] - w.start[1]
        length_m = float(math.hypot(dx, dy))
        if length_m < 1e-9:
            continue

        # Left-hand normal.
        nx = -dy / length_m
        ny = dx / length_m

        # Sample along baseline/original axis (visualized dashed line).
        p1_m = (w.start[0] + nx * w.axis_offset, w.start[1] + ny * w.axis_offset)
        p2_m = (w.end[0] + nx * w.axis_offset, w.end[1] + ny * w.axis_offset)
        p1_px = to_px_float(p1_m)
        p2_px = to_px_float(p2_m)

        length_px = float(math.hypot(p2_px[0] - p1_px[0], p2_px[1] - p1_px[1]))
        if length_px < 1e-6:
            continue

        num_samples = max(5, int(length_px))
        ts = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)

        xs = p1_px[0] + (p2_px[0] - p1_px[0]) * ts
        ys = p1_px[1] + (p2_px[1] - p1_px[1]) * ts

        # Bilinear sample distance transform at subpixel coordinates.
        coords = np.vstack([ys, xs])
        vals_px = map_coordinates_custom(dist_map, coords, order=1, mode="constant", cval=0.0).astype(np.float32)

        valid = vals_px > 1.0
        if not np.any(valid):
            continue

        # Thickness samples store half-thickness in px, and s in meters from wall.start along the wall.
        thickness_samples: List[Tuple[float, float]] = []
        for t, v in zip(ts.tolist(), vals_px.tolist()):
            if v <= 1.0:
                continue
            thickness_samples.append((float(t) * length_m, float(v)))

        if not thickness_samples:
            continue

        # Recompute wall thickness from samples.
        vals = [v for (_s, v) in thickness_samples if v > 1.0]
        median_dist = float(np.median(vals)) if vals else 0.0
        thickness_px = median_dist * 2.0
        thickness_m = thickness_px * float(cfg.meters_per_pixel)
        if cfg.thickness_rounding_m and cfg.thickness_rounding_m > 0:
            thickness_m = round(thickness_m / cfg.thickness_rounding_m) * cfg.thickness_rounding_m

        w.thickness_samples = thickness_samples
        w.thickness = float(thickness_m)

# --- CONFIGURATION ---
@dataclass
class Config:
    meters_per_pixel: float = 0.02
    wall_height: float = 2.7
    default_thickness: float = 0.20 # Used if sampling fails
    
    # Tolerances & Thresholds (Meters)
    gravity_snap_axis_snap_tol_m: float = 0.20
    consolidate_axis_cluster_tol_m: float = 0.15
    collinear_tol_deg: float = 15.0 
    ortho_tol_deg: float = 15.0     
    
    min_spur_length_m: float = 0.20
    spur_alignment_power: float = 2.0 
    extension_max_m: float = 0.20
    
    axis_offset_min_m: float = 0.06 
    axis_offset_min_overlap_m: float = 0.10
    axis_offset_sigma_m: float = 0.10 
    orthogonal_junction_snap_m: float = 0.2
    
    opening_bridge_m: float = 0.06
    epsilon_rdp_m: float = 0.06
    
    long_opening_m: float = 2.5
    door_circularity_thresh: float = 0.25
    door_swing_perp_factor: float = 1.5
    door_swing_perp_min_m: float = 0.16
    
    default_door_height_m: float = 2.1
    default_window_height_m: float = 1.2
    default_window_sill_m: float = 0.9
    
    opening_wall_max_dist_m: float = 1.2
    opening_search_dilate_m: float = 0.30
    corner_gap_deviation_deg: float = 30.0
    opening_active_face_eps_m: float = 0.08
    opening_active_face_max_pts: int = 5000
    thickness_rounding_m: float = 0.01

    # Extracted Hardcoded Values (Meters)
    corner_cut_min_len_m: float = 0.24
    corner_cut_max_len_m: float = 5.0
    min_opening_area_sq_m: float = 0.004 # ~10px at 0.02
    
    t_junction_split_tol_m: float = 0.2
    t_junction_end_margin_m: float = 0.2
    t_junction_min_seg_len_m: float = 0.05
    
    merge_thickness_tol_m: float = 0.05
    merge_collinear_dist_m: float = 0.1
    merge_near_joint_dist_m: float = 0.2

# --- DATA STRUCTURES ---
@dataclass
class Wall:
    id: str
    start: Tuple[float, float] # (x, y) in World Meters
    end: Tuple[float, float]
    thickness: float = 0.2
    height: float = 2.7
    axis_offset: float = 0.0
    px_geom: Optional[LineString] = None # Original pixel geometry
    # Thickness samples: (s_m, half_thickness_px_from_dist_transform)
    # s_m is distance from wall.start along the wall, in meters.
    thickness_samples: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class Opening:
    id: str
    type: str # "door" or "window"
    wall_id: str
    at: float # Distance from start
    width: float
    height: float = 2.1
    sill: float = 0.0
    px_center: Optional[Tuple[int, int]] = None

@dataclass
class Furniture:
    id: str
    type: str
    x: float
    y: float
    width: float
    length: float
    rotation: float

# --- PHASE 0: IO & MASKS ---
def get_opening_jamb_context(cnt, mask_wall, cfg: Config):
    """
    Returns key information about the two biggest nearby wall components (jambs) around an opening contour.
    Used both for robust bridging and later for opening classification (avoid duplicated logic).
    """
    single_mask = np.zeros_like(mask_wall)
    cv2.drawContours(single_mask, [cnt], -1, 255, -1)
    search_zone_px = int(round(cfg.opening_search_dilate_m / cfg.meters_per_pixel))
    search_zone = cv2.dilate(
        single_mask,
        np.ones((search_zone_px, search_zone_px), np.uint8),
    )
    nearby_walls = cv2.bitwise_and(mask_wall, search_zone)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(nearby_walls)
    if num_labels < 3:
        return None

    sorted_indices = np.argsort(stats[1:, 4])[::-1] + 1
    jamb_a_idx = sorted_indices[0]
    jamb_b_idx = sorted_indices[1]

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center_opening = (cx, cy)
    else:
        center_opening = (int(np.mean(cnt[:, 0, 0])), int(np.mean(cnt[:, 0, 1])))

    # Stable representatives for junction/corner logic.
    pt_a_centroid = (int(centroids[jamb_a_idx][0]), int(centroids[jamb_a_idx][1]))
    pt_b_centroid = (int(centroids[jamb_b_idx][0]), int(centroids[jamb_b_idx][1]))

    def component_boundary_points(component_idx: int) -> np.ndarray:
        component_mask = (labels == component_idx).astype(np.uint8) * 255
        if int(component_mask.max()) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        kernel = np.ones((3, 3), np.uint8)
        boundary = cv2.morphologyEx(component_mask, cv2.MORPH_GRADIENT, kernel)
        ys, xs = np.nonzero(boundary)
        if xs.size == 0:
            ys, xs = np.nonzero(component_mask)
        if xs.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        return np.column_stack([xs, ys]).astype(np.float32)

    def subsample_points(points: np.ndarray) -> np.ndarray:
        if points.shape[0] <= cfg.opening_active_face_max_pts:
            return points
        step = int(math.ceil(points.shape[0] / float(cfg.opening_active_face_max_pts)))
        return points[::step]

    pts_a = subsample_points(component_boundary_points(jamb_a_idx))
    pts_b = subsample_points(component_boundary_points(jamb_b_idx))

    # "Face" representatives near the actual gap (used for drawing bridges).
    pt_a_face: Optional[Tuple[int, int]] = None
    pt_b_face: Optional[Tuple[int, int]] = None
    if pts_a.shape[0] > 0 and pts_b.shape[0] > 0:
        # SAFETY: Using custom KDTree for Vercel optimization
        # tree_a = cKDTree(pts_a)
        tree_a = SimpleKDTree(pts_a)
        # tree_b = cKDTree(pts_b)
        tree_b = SimpleKDTree(pts_b)

        dists_a_to_b, _ = tree_b.query(pts_a, k=1)
        dists_b_to_a, _ = tree_a.query(pts_b, k=1)
        d_min = float(min(float(np.min(dists_a_to_b)), float(np.min(dists_b_to_a))))

        eps = float(cfg.opening_active_face_eps_m / cfg.meters_per_pixel)
        active_a = pts_a[dists_a_to_b <= d_min + eps]
        active_b = pts_b[dists_b_to_a <= d_min + eps]

        if active_a.shape[0] > 0:
            ca = active_a.mean(axis=0)
            _, idx = tree_a.query(ca, k=1)
            ax, ay = pts_a[int(idx)]
            pt_a_face = (int(ax), int(ay))
        if active_b.shape[0] > 0:
            cb = active_b.mean(axis=0)
            _, idx = tree_b.query(cb, k=1)
            bx, by = pts_b[int(idx)]
            pt_b_face = (int(bx), int(by))

    if pt_a_face is None:
        pt_a_face = pt_a_centroid
    if pt_b_face is None:
        pt_b_face = pt_b_centroid

    return {
        "labels": labels,
        "jamb_a_idx": jamb_a_idx,
        "jamb_b_idx": jamb_b_idx,
        # Back-compat keys (used by older call sites): prefer "face" points.
        "pt_a": pt_a_face,
        "pt_b": pt_b_face,
        # Explicit keys for stable-vs-face usage.
        "pt_a_centroid": pt_a_centroid,
        "pt_b_centroid": pt_b_centroid,
        "pt_a_face": pt_a_face,
        "pt_b_face": pt_b_face,
        "center_opening": center_opening,
    }


def min_axis_deviation_deg(pt_a: Tuple[float, float], pt_b: Tuple[float, float]) -> float:
    dx = pt_b[0] - pt_a[0]
    dy = pt_b[1] - pt_a[1]
    angle_deg = math.degrees(math.atan2(dy, dx)) % 360

    dist_to_0 = min(abs(angle_deg - 0), abs(angle_deg - 360))
    dist_to_90 = abs(angle_deg - 90)
    dist_to_180 = abs(angle_deg - 180)
    dist_to_270 = abs(angle_deg - 270)

    return min(dist_to_0, dist_to_90, dist_to_180, dist_to_270)


def compute_corner_intersection(
    labels: np.ndarray,
    jamb_a_idx: int,
    jamb_b_idx: int,
    pt_a: Tuple[int, int],
    pt_b: Tuple[int, int],
    center_opening: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    # Candidates
    candidates: List[Tuple[int, int]] = []

    # 1) Axis-aligned candidates.
    candidates.append((pt_a[0], pt_b[1]))
    candidates.append((pt_b[0], pt_a[1]))

    # 2) FitLine intersection candidate.
    mask_a = (labels == jamb_a_idx).astype(np.uint8)
    mask_b = (labels == jamb_b_idx).astype(np.uint8)
    pts_a = cv2.findNonZero(mask_a)
    pts_b = cv2.findNonZero(mask_b)

    if pts_a is not None and len(pts_a) > 5 and pts_b is not None and len(pts_b) > 5:
        [vx_a, vy_a, x_a, y_a] = cv2.fitLine(pts_a, cv2.DIST_L2, 0, 0.01, 0.01)
        [vx_b, vy_b, x_b, y_b] = cv2.fitLine(pts_b, cv2.DIST_L2, 0, 0.01, 0.01)

        det = -vx_a[0] * vy_b[0] + vy_a[0] * vx_b[0]
        if abs(det) > 0.1:
            dx = x_b[0] - x_a[0]
            dy = y_b[0] - y_a[0]
            t = (dx * (-vy_b[0]) - (-vx_b[0]) * dy) / det
            ix = x_a[0] + t * vx_a[0]
            iy = y_a[0] + t * vy_a[0]
            candidates.append((int(round(ix)), int(round(iy))))

    # Select best candidate based on proximity to opening center (with sanity guard).
    best_cand: Optional[Tuple[int, int]] = None
    min_dist = float("inf")

    dist_ab = math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])
    if dist_ab <= 1e-6:
        return None

    for c in candidates:
        d = math.hypot(c[0] - center_opening[0], c[1] - center_opening[1])

        dist_to_a = math.hypot(c[0] - pt_a[0], c[1] - pt_a[1])
        dist_to_b = math.hypot(c[0] - pt_b[0], c[1] - pt_b[1])
        if dist_to_a < dist_ab * 2.5 and dist_to_b < dist_ab * 2.5:
            if d < min_dist:
                min_dist = d
                best_cand = c

    if best_cand and min_dist < dist_ab:
        return best_cand
    return None


def cut_corner_opening_in_mask(
    mask_open: np.ndarray,
    cnt: np.ndarray,
    intersection: Tuple[int, int],
    pt_a: Tuple[int, int],
    pt_b: Tuple[int, int],
    cfg: Config
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Draw a 45-degree cut line through `intersection` to split a single corner-opening blob into 2.
    Picks the +45 or -45 diagonal that best separates the two jamb components (pt_a/pt_b).
    Returns the (p1, p2) endpoints that were used.
    """
    (ix, iy) = intersection

    rect = cv2.minAreaRect(cnt)
    rect_w_px, rect_h_px = rect[1]
    max_dim = float(max(rect_w_px, rect_h_px))
    min_dim = float(min(rect_w_px, rect_h_px))

    # Long enough to traverse the opening blob, plus a little margin.
    min_len_px = cfg.corner_cut_min_len_m / cfg.meters_per_pixel
    max_len_px = cfg.corner_cut_max_len_m / cfg.meters_per_pixel
    half_len = int(round(max(min_len_px, min(max_len_px, 0.9 * max_dim))))

    def signed_side(normal: Tuple[int, int], p: Tuple[int, int]) -> float:
        return float((p[0] - ix) * normal[0] + (p[1] - iy) * normal[1])

    # For slope +1 line: normal (1,-1). For slope -1 line: normal (1,1).
    n_pos = (1, -1)
    n_neg = (1, 1)

    sa_pos = signed_side(n_pos, pt_a)
    sb_pos = signed_side(n_pos, pt_b)
    sa_neg = signed_side(n_neg, pt_a)
    sb_neg = signed_side(n_neg, pt_b)

    separates_pos = (sa_pos == 0.0) or (sb_pos == 0.0) or (sa_pos * sb_pos < 0.0)
    separates_neg = (sa_neg == 0.0) or (sb_neg == 0.0) or (sa_neg * sb_neg < 0.0)

    if separates_pos and not separates_neg:
        dir_x, dir_y = 1.0, 1.0
    elif separates_neg and not separates_pos:
        dir_x, dir_y = 1.0, -1.0
    else:
        score_pos = abs(sa_pos - sb_pos)
        score_neg = abs(sa_neg - sb_neg)
        if score_pos >= score_neg:
            dir_x, dir_y = 1.0, 1.0
        else:
            dir_x, dir_y = 1.0, -1.0

    p1 = (int(round(ix + half_len * dir_x)), int(round(iy + half_len * dir_y)))
    p2 = (int(round(ix - half_len * dir_x)), int(round(iy - half_len * dir_y)))

    h, w = mask_open.shape[:2]
    p1 = (max(0, min(w - 1, p1[0])), max(0, min(h - 1, p1[1])))
    p2 = (max(0, min(w - 1, p2[0])), max(0, min(h - 1, p2[1])))

    # Cut with black (0) on a 1-channel mask.
    cv2.line(mask_open, p1, p2, 0, thickness=2, lineType=cv2.LINE_8)
    return p1, p2


def split_corner_openings_in_mask(mask_open: np.ndarray, mask_wall: np.ndarray, debug_dir: Optional[str], cfg: Config) -> np.ndarray:
    """
    Corner openings can get rasterized as a single connected component (L-shaped).
    This pass "cuts" those blobs with a 45-degree line so later `findContours` sees two openings.
    """
    mask_out = mask_open.copy()
    contours, _ = cv2.findContours(mask_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_img = None
    if debug_dir:
        debug_img = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)

    min_area_px = cfg.min_opening_area_sq_m / (cfg.meters_per_pixel ** 2)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue

        ctx = get_opening_jamb_context(cnt, mask_wall, cfg)
        if ctx is None:
            continue

        pt_a_centroid = ctx["pt_a_centroid"]
        pt_b_centroid = ctx["pt_b_centroid"]
        center_opening = ctx["center_opening"]

        # Only split skew/corner gaps.
        is_skew_gap = min_axis_deviation_deg(pt_a_centroid, pt_b_centroid) > cfg.corner_gap_deviation_deg
        if not is_skew_gap:
            continue

        intersection = compute_corner_intersection(
            labels=ctx["labels"],
            jamb_a_idx=ctx["jamb_a_idx"],
            jamb_b_idx=ctx["jamb_b_idx"],
            pt_a=pt_a_centroid,
            pt_b=pt_b_centroid,
            center_opening=center_opening,
        )
        if not intersection:
            continue

        p1, p2 = cut_corner_opening_in_mask(mask_out, cnt, intersection, pt_a_centroid, pt_b_centroid, cfg)
        if debug_img is not None:
            cv2.circle(debug_img, intersection, 3, (255, 0, 0), -1)
            cv2.line(debug_img, p1, p2, (0, 255, 255), 2, lineType=cv2.LINE_8)

    if debug_dir and debug_img is not None:
        cv2.imwrite(f"{debug_dir}/00_openings_split.png", debug_img)

    return mask_out


def create_robust_union_mask(mask_wall, mask_open, debug_dir, cfg: Config):
    """
    Bridges wall gaps ONLY where a blue opening exists.
    """
    union_mask = mask_wall.copy()
    debug_img = cv2.cvtColor(mask_wall, cv2.COLOR_GRAY2BGR)
    
    contours, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area_px = cfg.min_opening_area_sq_m / (cfg.meters_per_pixel ** 2)
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < min_area_px: continue

        ctx = get_opening_jamb_context(cnt, mask_wall, cfg)
        if ctx is None:
            continue

        labels = ctx["labels"]
        jamb_a_idx = ctx["jamb_a_idx"]
        jamb_b_idx = ctx["jamb_b_idx"]
        pt_a_face = ctx["pt_a_face"]
        pt_b_face = ctx["pt_b_face"]
        pt_a_centroid = ctx["pt_a_centroid"]
        pt_b_centroid = ctx["pt_b_centroid"]
        center_opening = ctx["center_opening"]

        is_corner = False
        intersection = None

        # If the angle is more than the threshold off-axis, it's a diagonal (corner) gap.
        is_skew_gap = min_axis_deviation_deg(pt_a_centroid, pt_b_centroid) > cfg.corner_gap_deviation_deg

        if is_skew_gap:
            intersection = compute_corner_intersection(
                labels=labels,
                jamb_a_idx=jamb_a_idx,
                jamb_b_idx=jamb_b_idx,
                pt_a=pt_a_centroid,
                pt_b=pt_b_centroid,
                center_opening=center_opening,
            )
            if intersection:
                is_corner = True

        if is_corner and intersection:
            cv2.line(union_mask, pt_a_centroid, intersection, 255, thickness=4)
            cv2.line(union_mask, intersection, pt_b_centroid, 255, thickness=4)

            cv2.line(debug_img, pt_a_centroid, intersection, (0, 255, 0), 2)
            cv2.line(debug_img, intersection, pt_b_centroid, (0, 255, 0), 2)
            cv2.circle(debug_img, intersection, 3, (255, 0, 0), -1) # Mark the corner

        else:
            cv2.line(union_mask, pt_a_face, pt_b_face, 255, thickness=4)
            cv2.line(debug_img, pt_a_face, pt_b_face, (0, 0, 255), 2)
            
    if debug_dir:
        cv2.imwrite(f"{debug_dir}/00_robust_bridges.png", debug_img)
        
    return union_mask

def load_and_preprocess(path: str, debug_dir: str, cfg: Config):
    print(f"Phase 0: Loading {path}...")
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(f"Could not load {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Walls are now BLACKish. Background is WHITE.
    mask_wall = cv2.inRange(img_rgb, np.array([0, 0, 0]), np.array([80, 80, 80]))
    # Robust opening threshold: loosened HSV range to handle variance.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([100, 30, 50], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    mask_open_raw = cv2.inRange(hsv, lower, upper)
    # Fill tiny gaps inside swings caused by threshold holes.
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_open_raw = cv2.morphologyEx(mask_open_raw, cv2.MORPH_CLOSE, kernel_clean)
    
    bridge_px = int(round(cfg.opening_bridge_m / cfg.meters_per_pixel))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_px, bridge_px))
    mask_open_dilated = cv2.dilate(mask_open_raw, kernel, iterations=1)
    
    mask_union = create_robust_union_mask(mask_wall, mask_open_dilated, debug_dir, cfg)
    mask_union = cv2.morphologyEx(mask_union, cv2.MORPH_CLOSE, kernel)

    # Compute Distance Transform on SOLID WALLS only (not the bridge)
    dist_map = cv2.distanceTransform(mask_wall, cv2.DIST_L2, 5)

    mask_open = split_corner_openings_in_mask(mask_open_raw, mask_wall, debug_dir, cfg)

    if debug_dir:
        cv2.imwrite(f"{debug_dir}/00_openings_raw.png", mask_open_raw)
        cv2.imwrite(f"{debug_dir}/00_openings_dilated.png", mask_open_dilated)
        # Visualize dilated openings over wall mask (walls=white, dilated openings=blue).
        overlay = np.zeros((mask_wall.shape[0], mask_wall.shape[1], 3), dtype=np.uint8)
        overlay[mask_wall > 0] = (255, 255, 255)
        overlay[mask_open_dilated > 0] = (255, 0, 0)
        cv2.imwrite(f"{debug_dir}/00_openings_dilated_over_walls.png", overlay)
        cv2.imwrite(f"{debug_dir}/00_union_mask.png", mask_union)
        cv2.imwrite(f"{debug_dir}/00_dist_map.png", (dist_map/dist_map.max()*255).astype(np.uint8))
        cv2.imwrite(f"{debug_dir}/00_openings_mask.png", mask_open)

    return mask_union, mask_wall, mask_open, dist_map, img_rgb.shape[:2], img_rgb

# --- PHASE 1: SKELETON ---
def build_skeleton_graph(mask_union, debug_dir, cfg):
    print("Phase 1: Skeletonizing...")
    binary = mask_union > 127
    # SAFETY: Using custom Zhang-Suen implementation
    # skel = skeletonize(binary)
    skel = skeletonize_custom(binary)
    skel_uint8 = (skel * 255).astype(np.uint8)
    if debug_dir: cv2.imwrite(f"{debug_dir}/01_skeleton.png", skel_uint8)
        
    y_idxs, x_idxs = np.nonzero(skel)
    pixels = list(zip(x_idxs, y_idxs))
    pixel_set = set(pixels)
    G = nx.Graph()
    
    for x, y in pixels:
        G.add_node((x, y))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx_pt = (x + dx, y + dy)
                if nx_pt in pixel_set:
                    dist = math.sqrt(dx*dx + dy*dy)
                    G.add_edge((x, y), nx_pt, weight=dist)

    def spur_alignment_score(path: List[Tuple[int, int]]) -> float:
        """
        Returns ~1.0 for axis-aligned (H/V) spurs, ~0.707 for diagonals.

        Uses only the start and end of the spur (net direction), not per-step directions.
        """
        if len(path) < 2:
            return 1.0
        dx = float(path[-1][0] - path[0][0])
        dy = float(path[-1][1] - path[0][1])
        norm = math.hypot(dx, dy)
        if norm <= 1e-9:
            return 1.0
        ux = abs(dx) / norm
        uy = abs(dy) / norm
        return max(ux, uy)

    # --- STEP 1: EXTEND ENDS (Before Pruning) ---
    # We extend ALL ends (including noise) to the wall boundary.
    # This gives short valid walls a chance to survive pruning if extension makes them long enough.
    
    # Identify ends
    initial_ends = [n for n in G.nodes() if G.degree(n) == 1]
    
    for end_node in initial_ends:
        if G.degree(end_node) != 1: continue # Safety if graph changed? (unlikely here)
        
        nbrs = list(G.neighbors(end_node))
        if not nbrs: continue
        prev_node = nbrs[0]
        
        # Direction: Prev -> End
        dx = float(end_node[0] - prev_node[0])
        dy = float(end_node[1] - prev_node[1])
        norm = math.hypot(dx, dy)
        if norm < 1e-9: continue
        
        ux, uy = dx/norm, dy/norm
        
        # Raycast
        curr_x, curr_y = float(end_node[0]), float(end_node[1])
        
        extended_pixels = []
        max_dist = float(cfg.extension_max_m / cfg.meters_per_pixel)
        dist_accum = 0.0
        
        while dist_accum < max_dist:
            curr_x += ux
            curr_y += uy
            dist_accum += 1.0
            
            ix, iy = int(round(curr_x)), int(round(curr_y))
            
            # Bounds check
            if ix < 0 or iy < 0 or ix >= mask_union.shape[1] or iy >= mask_union.shape[0]:
                break
                
            # Content check (0=background)
            if mask_union[iy, ix] == 0:
                break
            
            # Add intermediate pixels to graph to maintain connectivity geometry?
            if (ix, iy) not in G: 
                extended_pixels.append((ix, iy))
        
        # If we extended anything
        if extended_pixels:
            prev_p = end_node
            for p in extended_pixels:
                G.add_node(p)
                dist = math.hypot(p[0]-prev_p[0], p[1]-prev_p[1])
                G.add_edge(prev_p, p, weight=dist)
                prev_p = p

    # --- STEP 2: SYMMETRIC PRUNING (After Extension) ---
    # Now prune spurs. Since valid ends are extended to the wall face, they should be long enough to survive.
    # Short noise spurs that didn't extend much (or were very short to begin with) will be pruned.
    
    while True:
        pruned_any = False
        
        # Identify all potential spur tips (degree 1)
        deg1_nodes = [n for n in G.nodes() if G.degree(n) == 1]
        
        # Map Junction -> List of (spur_tip, entire_path, effective_length)
        junction_spurs = defaultdict(list)
        
        for start in deg1_nodes:
            if not G.has_node(start): continue # Already removed in this pass?

            path = [start]
            length = 0.0
            prev = None
            curr = start
            alignment = 1.0
            effective_length = 0.0
            
            # Walk up the spur
            while True:
                nbrs = list(G.neighbors(curr))
                next_candidates = [n for n in nbrs if n != prev]
                if not next_candidates:
                    # Isolated line or loop? Stop.
                    break

                nxt = next_candidates[0]
                length += float(G[curr][nxt].get("weight", 1.0))
                path.append(nxt)
                prev, curr = curr, nxt

                alignment = spur_alignment_score(path)
                effective_length = length * (alignment ** float(cfg.spur_alignment_power))
                
                # Check if we hit a true junction (degree > 2) OR if path is too long
                min_spur_len_px = float(cfg.min_spur_length_m / cfg.meters_per_pixel)
                if G.degree(curr) > 2 or (effective_length + 1e-9) >= min_spur_len_px:
                    break

            # If meaningful spur attached to a junction
            if (effective_length + 1e-9) < min_spur_len_px and G.degree(curr) >= 3:
                junction_spurs[curr].append({
                    'tip': start,
                    'path': path, # distinct nodes from tip to junction (curr is last, not pruned)
                    'len': effective_length
                })

        # Process junctions
        for junction, spurs in junction_spurs.items():
            # If a junction has spurs, we prune them. 
            
            for spur_info in spurs:
                # Prune nodes in path UP TO (but not including) the junction
                nodes_to_remove = spur_info['path'][:-1]
                for p in nodes_to_remove:
                    if G.has_node(p):
                        G.remove_node(p)
                        pruned_any = True

        if not pruned_any:
            break

    if debug_dir:
        # Create a blank black image
        pruned_img = np.zeros_like(skel_uint8)
        
        # Draw edges
        for u, v in G.edges():
            pt1 = (int(u[0]), int(u[1])) # Note: u is (x,y)
            pt2 = (int(v[0]), int(v[1]))
            cv2.line(pruned_img, pt1, pt2, 255, 1)
            
        cv2.imwrite(f"{debug_dir}/01_skeleton_pruned.png", pruned_img)
        
    return G

# --- PHASE 2: VECTORIZATION ---
def graph_to_vectors(G, debug_dir, cfg):
    print("Phase 2: Vectorizing...")
    key_nodes = [n for n in G.nodes() if G.degree(n) != 2]
    if not key_nodes and len(G) > 0: key_nodes = [list(G.nodes())[0]]

    vectors = [] 
    visited_edges = set()

    for node in key_nodes:
        for neighbor in G.neighbors(node):
            if tuple(sorted((node, neighbor))) in visited_edges: continue
            
            path = [node, neighbor]
            curr = neighbor
            prev = node
            visited_edges.add(tuple(sorted((prev, curr))))
            
            while G.degree(curr) == 2:
                nbrs = list(G.neighbors(curr))
                next_node = nbrs[0] if nbrs[0] != prev else nbrs[1]
                path.append(next_node)
                visited_edges.add(tuple(sorted((curr, next_node))))
                prev, curr = curr, next_node
                if curr in key_nodes: break
            
            line = LineString(path)
            epsilon_px = cfg.epsilon_rdp_m / cfg.meters_per_pixel
            simplified = line.simplify(epsilon_px, preserve_topology=True)
            if simplified.length > 2:
                if simplified.geom_type == 'MultiLineString':
                    for g in simplified.geoms: vectors.append(g)
                else:
                    vectors.append(simplified)
    return vectors

# --- PHASE 3: GRAVITY SNAPPING ---
def gravity_snap(vectors, debug_dir, cfg):
    print("Phase 3: Gravity Snapping...")
    walls = []
    for v in vectors:
        coords = list(v.coords)
        for i in range(len(coords)-1):
            p1, p2 = coords[i], coords[i+1]
            walls.append({
                'start': list(p1), 
                'end': list(p2),
                'geom': LineString([p1, p2]),
                'orient': 'O',
                'len': Point(p1).distance(Point(p2))
            })

    for w in walls:
        dx = w['end'][0] - w['start'][0]
        dy = w['end'][1] - w['start'][1]
        angle = math.degrees(math.atan2(dy, dx)) % 360
        
        if abs(angle - 0) < cfg.ortho_tol_deg or abs(angle - 180) < cfg.ortho_tol_deg or abs(angle - 360) < cfg.ortho_tol_deg:
            avg_y = (w['start'][1] + w['end'][1]) / 2
            w['start'][1] = avg_y; w['end'][1] = avg_y; w['orient'] = 'H'
        elif abs(angle - 90) < cfg.ortho_tol_deg or abs(angle - 270) < cfg.ortho_tol_deg:
            avg_x = (w['start'][0] + w['end'][0]) / 2
            w['start'][0] = avg_x; w['end'][0] = avg_x; w['orient'] = 'V'

    # Snap Horizontals
    h_walls = [w for w in walls if w['orient'] == 'H']
    h_walls.sort(key=lambda x: x['len'], reverse=True)
    axes_y = []
    for w in h_walls:
        y = w['start'][1]
        snapped = False
        snap_tol_px = cfg.gravity_snap_axis_snap_tol_m / cfg.meters_per_pixel
        for axis in axes_y:
            if abs(axis['val'] - y) < snap_tol_px:
                w['start'][1] = axis['val']; w['end'][1] = axis['val']; snapped = True; break
        if not snapped: axes_y.append({'val': y})

    # Snap Verticals
    v_walls = [w for w in walls if w['orient'] == 'V']
    v_walls.sort(key=lambda x: x['len'], reverse=True)
    axes_x = []
    for w in v_walls:
        x = w['start'][0]
        snapped = False
        snap_tol_px = cfg.gravity_snap_axis_snap_tol_m / cfg.meters_per_pixel
        for axis in axes_x:
            if abs(axis['val'] - x) < snap_tol_px:
                w['start'][0] = axis['val']; w['end'][0] = axis['val']; snapped = True; break
        if not snapped: axes_x.append({'val': x})

    # IMPORTANT: start/end may have been modified (ortho + axis snap). Keep geometry consistent so
    # downstream sampling (distance transform) runs along the snapped segment, not the original.
    for w in walls:
        p1 = (float(w['start'][0]), float(w['start'][1]))
        p2 = (float(w['end'][0]), float(w['end'][1]))
        w['geom'] = LineString([p1, p2])
        w['len'] = float(Point(p1).distance(Point(p2)))

    return walls

# --- PHASE 4: TOPOLOGY RECONSTRUCTION ---
def rebuild_topology(walls, cfg):
    print("Phase 4: Rebuilding Topology...")
    points = []
    for w in walls: points.extend([w['start'], w['end']])
    if not points: return []

    if not points: return []

    # SAFETY: Using custom KDTree for Vercel optimization
    # tree = cKDTree(points)
    tree = SimpleKDTree(points)
    snap_tol_px = cfg.gravity_snap_axis_snap_tol_m / cfg.meters_per_pixel
    pairs = tree.query_pairs(snap_tol_px)
    pt_graph = nx.Graph()
    for i in range(len(points)): pt_graph.add_node(i)
    pt_graph.add_edges_from(list(pairs))
    clusters = list(nx.connected_components(pt_graph))
    
    pt_map = {}
    for cluster in clusters:
        cluster_pts = [points[i] for i in cluster]
        cx = sum(p[0] for p in cluster_pts) / len(cluster_pts)
        cy = sum(p[1] for p in cluster_pts) / len(cluster_pts)
        for i in cluster: pt_map[i] = (cx, cy)
            
    final_walls = []
    for i, w in enumerate(walls):
        s_idx, e_idx = i*2, i*2 + 1
        ns, ne = pt_map[s_idx], pt_map[e_idx]
        if math.hypot(ns[0]-ne[0], ns[1]-ne[1]) < 1.0: continue
            
        final_walls.append({
            'start': ns, 'end': ne,
            'px_geom': LineString([ns, ne]),
            'px_len': float(Point(ns).distance(Point(ne))),
        })
    return final_walls

# --- PHASE 5: ATTRIBUTES ---
def process_attributes(walls, dist_map, h_img, cfg):
    print("Phase 5: Attributes & Scale...")
    wall_objs = []
    
    for i, w in enumerate(walls):
        tmp_id = f"tmp_{i}"
        # NOTE: Wall thickness is sampled later (after axis_offset is known), to avoid sampling
        # on a snapped axis that can run along the wall mask edge.
        thickness_m = float(cfg.default_thickness)
        if cfg.thickness_rounding_m and cfg.thickness_rounding_m > 0:
            thickness_m = round(thickness_m / cfg.thickness_rounding_m) * cfg.thickness_rounding_m
        
        # Coordinate Transform
        sx, sy = w['start']; ex, ey = w['end']
        wx1 = sx * cfg.meters_per_pixel
        wy1 = (h_img - sy) * cfg.meters_per_pixel
        wx2 = ex * cfg.meters_per_pixel
        wy2 = (h_img - ey) * cfg.meters_per_pixel
        
        if wx1 > wx2 or (abs(wx1 - wx2) < 1e-4 and wy1 > wy2):
            wx1, wy1, wx2, wy2 = wx2, wy2, wx1, wy1
            
        thickness_samples_m: List[Tuple[float, float]] = []

        wall_objs.append(Wall(
            id=tmp_id, # Temp ID
            start=(wx1, wy1), end=(wx2, wy2),
            thickness=thickness_m, height=cfg.wall_height,
            axis_offset=0.0,
            px_geom=w['px_geom'],
            thickness_samples=thickness_samples_m,
        ))
    return wall_objs

# --- PHASE 5.5: CONSOLIDATION ---
def consolidate_walls(walls: List[Wall], cfg):
    """
    Merges collinear walls UNLESS they meet at a T-junction (Degree > 2).
    """
    print("Phase 5.5: Topology-Aware Consolidation...")
    
    # Pre-filter: Remove duplicate and heavily overlapping segments
    filtered_walls = []
    for i, w1 in enumerate(walls):
        is_duplicate = False
        for j, w2 in enumerate(walls):
            if i >= j: continue  # Skip self and already compared pairs
            
            # Check if they're on the same line (collinear and overlapping)
            dx1, dy1 = abs(w1.start[0] - w1.end[0]), abs(w1.start[1] - w1.end[1])
            dx2, dy2 = abs(w2.start[0] - w2.end[0]), abs(w2.start[1] - w2.end[1])
            
            is_w1_vert = dy1 > dx1
            is_w2_vert = dy2 > dx2
            
            if is_w1_vert != is_w2_vert: continue  # Different orientations
            
            # Check if on same axis (within tolerance)
            if is_w1_vert:
                axis_diff = abs(w1.start[0] - w2.start[0])
                if axis_diff > float(cfg.consolidate_axis_cluster_tol_m): continue
                
                # Check overlap on Y axis
                w1_min_y, w1_max_y = min(w1.start[1], w1.end[1]), max(w1.start[1], w1.end[1])
                w2_min_y, w2_max_y = min(w2.start[1], w2.end[1]), max(w2.start[1], w2.end[1])
                
                overlap = min(w1_max_y, w2_max_y) - max(w1_min_y, w2_min_y)
                w1_len = w1_max_y - w1_min_y
                
                # If w1 is >80% contained in w2, mark as duplicate
                if overlap > 0 and overlap / w1_len > 0.8:
                    is_duplicate = True
                    break
            else:  # Horizontal
                axis_diff = abs(w1.start[1] - w2.start[1])
                if axis_diff > float(cfg.consolidate_axis_cluster_tol_m): continue
                
                # Check overlap on X axis
                w1_min_x, w1_max_x = min(w1.start[0], w1.end[0]), max(w1.start[0], w1.end[0])
                w2_min_x, w2_max_x = min(w2.start[0], w2.end[0]), max(w2.start[0], w2.end[0])
                
                overlap = min(w1_max_x, w2_max_x) - max(w1_min_x, w2_min_x)
                w1_len = w1_max_x - w1_min_x
                
                # If w1 is >80% contained in w2, mark as duplicate
                if overlap > 0 and overlap / w1_len > 0.8:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered_walls.append(w1)
    
    print(f"  Filtered {len(walls) - len(filtered_walls)} duplicate/overlapping segments")
    walls = filtered_walls
    
    # Map node degrees
    node_counts = defaultdict(int)
    def get_key(pt): return (round(pt[0], 3), round(pt[1], 3))
    
    for w in walls:
        node_counts[get_key(w.start)] += 1
        node_counts[get_key(w.end)] += 1
            
    # Grouping Helper
    def cluster_and_merge(walls_subset, is_horiz):
        # 1. Cluster by primary axis (Y for horiz, X for vert)
        # Tolerance: meters, to catch slightly jagged walls
        TOL = float(cfg.consolidate_axis_cluster_tol_m)
        groups = [] # List of {'val': float, 'walls': []}
        
        idx = 1 if is_horiz else 0 # Primary axis index
        
        for w in walls_subset:
            pos = (w.start[idx] + w.end[idx]) / 2
            
            best_g, min_diff = None, float('inf')
            for g in groups:
                diff = abs(g['val'] - pos)
                if diff < TOL and diff < min_diff:
                    min_diff = diff
                    best_g = g
            
            if best_g:
                best_g['walls'].append(w)
                n = len(best_g['walls'])
                best_g['val'] = (best_g['val'] * (n-1) + pos) / n
            else:
                groups.append({'val': pos, 'walls': [w]})
        
        # 2. Merge within groups
        merged_results = []
        for g in groups:
            # Align them first! Force exact alignment to pivot for cleaner DSL
            pivot = g['val']
            for w in g['walls']:
                if is_horiz: 
                    w.start = (w.start[0], pivot)
                    w.end = (w.end[0], pivot)
                    if w.start[0] > w.end[0]: w.start, w.end = w.end, w.start
                else: 
                    w.start = (pivot, w.start[1])
                    w.end = (pivot, w.end[1])
                    if w.start[1] > w.end[1]: w.start, w.end = w.end, w.start
            
            merged_results.extend(merge_group(g['walls'], is_horiz))
        return merged_results

    h_input = []
    v_input = []

    for w in walls:
        dx, dy = abs(w.start[0] - w.end[0]), abs(w.start[1] - w.end[1])
        if dx > dy: h_input.append(w)
        else: v_input.append(w)
        
    final_walls = []
    
    def merge_group(group, is_horiz):
        idx = 0 if is_horiz else 1 # Sort/Gap axis
        
        # Pre-filter redundant walls WITHIN this group
        # Since they are all snapped to same pivot, we only need to check the 'range' axis (idx)
        cleaned_group = []
        for i, w1 in enumerate(group):
            is_redundant = False
            w1_min = min(w1.start[idx], w1.end[idx])
            w1_max = max(w1.start[idx], w1.end[idx])
            w1_len = w1_max - w1_min
            
            for j, w2 in enumerate(group):
                if i == j: continue
                w2_min = min(w2.start[idx], w2.end[idx])
                w2_max = max(w2.start[idx], w2.end[idx])
                w2_len = w2_max - w2_min
                
                # If w1 is contained in w2 (and w1 is smaller or equal length)
                # If equal length, use index to break tie (only remove one)
                if w1_len > w2_len: continue
                if abs(w1_len - w2_len) < 0.001 and i < j: continue # Tie breaker
                
                if w1_min >= w2_min - 0.01 and w1_max <= w2_max + 0.01:
                    is_redundant = True
                    break
            
            if not is_redundant:
                cleaned_group.append(w1)
        
        group = cleaned_group
        
        group.sort(key=lambda w: w.start[idx])
        merged = []
        if not group: return []
        
        current = group[0]
        for next_w in group[1:]:
            # Gap check
            c_end_val = current.end[idx]
            n_start_val = next_w.start[idx]
            gap = n_start_val - c_end_val
            
            # Simple 'touching' logic
            # Also check if they are "mergable" (similar thickness)
            
            # 1. Thickness match?
            # Allow merging if thickness is very close
            same_thick = abs(current.thickness - next_w.thickness) < cfg.merge_thickness_tol_m
            
            # Check T-Junction: Look for any OTHER wall near the join point
            # Ignore very short walls (<0.5m) as they are likely artifacts
            # Ignore collinear walls on the same axis (they're part of the same wall we're consolidating!)
            is_simple_joint = True
            c_end = current.end
            num_near = 0
            join_pt = c_end
            
            # Determine current wall's orientation
            curr_dx = abs(current.end[0] - current.start[0])
            curr_dy = abs(current.end[1] - current.start[1])
            curr_is_vert = curr_dy > curr_dx
            
            for check_w in walls:
                if check_w is current or check_w is next_w: continue
                
                # Calculate wall length and skip short artifacts
                check_len = math.hypot(check_w.end[0]-check_w.start[0], check_w.end[1]-check_w.start[1])
                if check_len < 0.5: continue  # Skip short artifact walls
                
                
                # Ignore collinear walls on the same axis (they're part of the same wall chain!)
                dx2, dy2 = check_w.end[0]-check_w.start[0], check_w.end[1]-check_w.start[1]
                
                # Determine orientation
                is_w2_vert = abs(dy2) > abs(dx2)
                
                if curr_is_vert == is_w2_vert:
                    if curr_is_vert:
                        if abs(current.start[0] - check_w.start[0]) < cfg.merge_collinear_dist_m: continue
                    else:
                        if abs(current.start[1] - check_w.start[1]) < cfg.merge_collinear_dist_m: continue

                d_s = math.hypot(check_w.start[0]-join_pt[0], check_w.start[1]-join_pt[1])
                d_e = math.hypot(check_w.end[0]-join_pt[0], check_w.end[1]-join_pt[1])
                if d_s < cfg.merge_near_joint_dist_m or d_e < cfg.merge_near_joint_dist_m: 
                    num_near += 1
            
            if num_near > 0: is_simple_joint = False
            
            # Thickness policy:
            # For collinear segments we merge whenever geometry/topology allows, and keep the thicker value.
            # This treats thickness differences as sampling noise (or conservative "max thickness wins").
            if gap < cfg.merge_near_joint_dist_m and is_simple_joint:
                # Merge
                current.thickness = max(current.thickness, next_w.thickness)
                if next_w.thickness_samples:
                    if is_horiz:
                        offset = float(next_w.start[0] - current.start[0])
                    else:
                        offset = float(next_w.start[1] - current.start[1])
                    if offset < 0:
                        offset = -offset
                    current.thickness_samples.extend([(s + offset, v) for (s, v) in next_w.thickness_samples])
                if is_horiz: current.end = (next_w.end[0], current.end[1])
                else: current.end = (current.end[0], next_w.end[1])
            else:
                 merged.append(current)
                 current = next_w
        merged.append(current)
        return merged

    final_walls.extend(cluster_and_merge(h_input, True))
    final_walls.extend(cluster_and_merge(v_input, False))
    
    # Post-consolidation: Remove short walls that are entirely contained in longer walls
    cleaned_walls = []
    for i, w1 in enumerate(final_walls):
        w1_len = math.hypot(w1.end[0]-w1.start[0], w1.end[1]-w1.start[1])
        
        is_redundant = False
        for j, w2 in enumerate(final_walls):
            if i == j: continue
            
            w2_len = math.hypot(w2.end[0]-w2.start[0], w2.end[1]-w2.start[1])
            
            # Only check if w1 is shorter than w2
            if w1_len >= w2_len: continue
            
            # Check if they're collinear
            dx1, dy1 = abs(w1.end[0] - w1.start[0]), abs(w1.end[1] - w1.start[1])
            dx2, dy2 = abs(w2.end[0] - w2.start[0]), abs(w2.end[1] - w2.start[1])
            
            is_w1_vert = dy1 > dx1
            is_w2_vert = dy2 > dx2
            
            if is_w1_vert != is_w2_vert: continue
            
            # Check if on same axis
            if is_w1_vert:
                if abs(w1.start[0] - w2.start[0]) > 0.05: continue
                
                # Check if w1 is contained in w2
                w1_min_y, w1_max_y = min(w1.start[1], w1.end[1]), max(w1.start[1], w1.end[1])
                w2_min_y, w2_max_y = min(w2.start[1], w2.end[1]), max(w2.start[1], w2.end[1])
                
                if w1_min_y >= w2_min_y and w1_max_y <= w2_max_y:
                    is_redundant = True
                    break
            else:  # Horizontal
                if abs(w1.start[1] - w2.start[1]) > 0.05: continue
                
                # Check if w1 is contained in w2
                w1_min_x, w1_max_x = min(w1.start[0], w1.end[0]), max(w1.start[0], w1.end[0])
                w2_min_x, w2_max_x = min(w2.start[0], w2.end[0]), max(w2.start[0], w2.end[0])
                
                if w1_min_x >= w2_min_x and w1_max_x <= w2_max_x:
                    is_redundant = True
                    break
        
        if not is_redundant:
            cleaned_walls.append(w1)
    
    final_walls = cleaned_walls

    def is_horizontal(w: Wall) -> bool:
        dx = abs(w.start[0] - w.end[0])
        dy = abs(w.start[1] - w.end[1])
        return dx >= dy

    def split_walls_at_t_junctions(walls_in: List[Wall]) -> List[Wall]:
        """
        Split a wall when an endpoint of a perpendicular wall lands on its interior.
        This prevents long merged runs (e.g., a long top wall) from spanning across a T-junction.
        """
        split_tol = cfg.t_junction_split_tol_m
        end_margin = cfg.t_junction_end_margin_m
        min_seg_len = cfg.t_junction_min_seg_len_m

        result: List[Wall] = []
        for w in walls_in:
            horiz = is_horizontal(w)
            if horiz:
                y = w.start[1]
                x0, x1 = sorted([w.start[0], w.end[0]])
                split_xs = set()
                for other in walls_in:
                    if other is w or is_horizontal(other):
                        continue
                    for pt in (other.start, other.end):
                        if abs(pt[1] - y) > split_tol:
                            continue
                        if pt[0] < x0 - split_tol or pt[0] > x1 + split_tol:
                            continue
                        if (x0 + end_margin) < pt[0] < (x1 - end_margin):
                            split_xs.add(pt[0])
                if not split_xs:
                    result.append(w)
                    continue
                coords = [x0] + sorted(split_xs) + [x1]
                for a, b in zip(coords, coords[1:]):
                    if (b - a) < min_seg_len:
                        continue
                    s0 = float(a - x0)
                    s1 = float(b - x0)
                    child_samples = [(s - s0, v) for (s, v) in w.thickness_samples if s0 <= s <= s1]
                    result.append(Wall(
                        id=w.id,
                        start=(a, y),
                        end=(b, y),
                        thickness=w.thickness,
                        height=w.height,
                        axis_offset=w.axis_offset,
                        px_geom=None,
                        thickness_samples=child_samples,
                    ))
            else:
                x = w.start[0]
                y0, y1 = sorted([w.start[1], w.end[1]])
                split_ys = set()
                for other in walls_in:
                    if other is w or not is_horizontal(other):
                        continue
                    for pt in (other.start, other.end):
                        if abs(pt[0] - x) > split_tol:
                            continue
                        if pt[1] < y0 - split_tol or pt[1] > y1 + split_tol:
                            continue
                        if (y0 + end_margin) < pt[1] < (y1 - end_margin):
                            split_ys.add(pt[1])
                if not split_ys:
                    result.append(w)
                    continue
                coords = [y0] + sorted(split_ys) + [y1]
                for a, b in zip(coords, coords[1:]):
                    if (b - a) < min_seg_len:
                        continue
                    s0 = float(a - y0)
                    s1 = float(b - y0)
                    child_samples = [(s - s0, v) for (s, v) in w.thickness_samples if s0 <= s <= s1]
                    result.append(Wall(
                        id=w.id,
                        start=(x, a),
                        end=(x, b),
                        thickness=w.thickness,
                        height=w.height,
                        axis_offset=w.axis_offset,
                        px_geom=None,
                        thickness_samples=child_samples,
                    ))
        return result

    def extend_clip_orthogonal_junctions(walls_in: List[Wall], tol_m: float) -> None:
        """
        Extend/clip axis-aligned wall endpoints so they meet perpendicular walls.

        This only moves endpoints along the wall axis (x for horizontals, y for verticals), keeping
        walls perfectly horizontal/vertical. It does not "shift" an axis (that's axis_offset).
        """
        tol_m = float(tol_m)
        if tol_m <= 0.0:
            return

        horiz: List[Tuple[Wall, float, float, float]] = []  # (wall, y, x0, x1)
        vert: List[Tuple[Wall, float, float, float]] = []   # (wall, x, y0, y1)

        for w in walls_in:
            dx = abs(w.end[0] - w.start[0])
            dy = abs(w.end[1] - w.start[1])
            if dx < 1e-9 and dy < 1e-9:
                continue
            if dx >= dy:
                y = (w.start[1] + w.end[1]) * 0.5
                x0, x1 = sorted([w.start[0], w.end[0]])
                horiz.append((w, y, x0, x1))
            else:
                x = (w.start[0] + w.end[0]) * 0.5
                y0, y1 = sorted([w.start[1], w.end[1]])
                vert.append((w, x, y0, y1))

        def normalize(w: Wall) -> None:
            wx1, wy1 = w.start
            wx2, wy2 = w.end
            if wx1 > wx2 or (abs(wx1 - wx2) < 1e-4 and wy1 > wy2):
                w.start, w.end = w.end, w.start

        # Snap horizontal endpoints to nearby vertical axes.
        for w, y, _x0, _x1 in horiz:
            for which in ("start", "end"):
                x0, y0 = getattr(w, which)
                best_x = None
                best_d = float("inf")
                for _vw, x_v, y_v0, y_v1 in vert:
                    if y < (y_v0 - tol_m) or y > (y_v1 + tol_m):
                        continue
                    d = abs(x_v - x0)
                    if d <= tol_m and d < best_d:
                        best_d = d
                        best_x = x_v
                if best_x is not None:
                    setattr(w, which, (float(best_x), float(y)))
            normalize(w)

        # Snap vertical endpoints to nearby horizontal axes.
        for w, x, _y0, _y1 in vert:
            for which in ("start", "end"):
                x0, y0 = getattr(w, which)
                best_y = None
                best_d = float("inf")
                for _hw, y_h, x_h0, x_h1 in horiz:
                    if x < (x_h0 - tol_m) or x > (x_h1 + tol_m):
                        continue
                    d = abs(y_h - y0)
                    if d <= tol_m and d < best_d:
                        best_d = d
                        best_y = y_h
                if best_y is not None:
                    setattr(w, which, (float(x), float(best_y)))
            normalize(w)

        # Final normalization pass (in case both ends moved).
        for w in walls_in:
            normalize(w)

    extend_clip_orthogonal_junctions(final_walls, cfg.orthogonal_junction_snap_m)
    final_walls = split_walls_at_t_junctions(final_walls)

    for w in final_walls:
        if abs(w.axis_offset) < cfg.axis_offset_min_m:
            w.axis_offset = 0.0
        
    for i, w in enumerate(final_walls):
        w.id = f"w_{i:03d}"
    return final_walls

# --- PHASE 6: OPENINGS ---
def extract_openings(mask_open, mask_wall, walls: List[Wall], cfg: Config):
    print("Phase 6: Projecting Openings...")
    contours, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    openings = []
    
    min_area_px = cfg.min_opening_area_sq_m / (cfg.meters_per_pixel ** 2)
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < min_area_px: continue
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        c_pt = Point(cx, cy)
        
        best_wall, min_dist = None, float('inf')
        for w in walls:
            # We must map current World coords back to Pixel coords roughly 
            # or use the original px_geom if it still aligns.
            # Since we merged walls, px_geom might be partial. 
            # Better approach: Project World Point to World Line.
            
            # Convert centroid to world (flip y)
            wx = cx * cfg.meters_per_pixel
            wy = (mask_open.shape[0] - cy) * cfg.meters_per_pixel
            
            # Create world line
            w_line = LineString([w.start, w.end])
            p_world = Point(wx, wy)
            
            dist = w_line.distance(p_world)
            if dist < min_dist:
                min_dist = dist
                best_wall = w
        
        # Max distance (meters) between opening centroid and wall axis.
        if min_dist > float(cfg.opening_wall_max_dist_m):
            continue
        
        rect = cv2.minAreaRect(cnt)
        rect_w_px, rect_h_px = rect[1]
        long_px = float(max(rect_w_px, rect_h_px))
        short_px = float(min(rect_w_px, rect_h_px))
        width_m = long_px * cfg.meters_per_pixel

        # Door-vs-window heuristics (plan view):
        # - If pixels extend far away (perpendicular) from the wall axis => door (door swing)
        # - Long openings => door (terrace/sliding doors)

        h_px = mask_open.shape[0]
        p1 = np.array(
            [
                best_wall.start[0] / cfg.meters_per_pixel,
                h_px - best_wall.start[1] / cfg.meters_per_pixel,
            ],
            dtype=np.float32,
        )
        p2 = np.array(
            [
                best_wall.end[0] / cfg.meters_per_pixel,
                h_px - best_wall.end[1] / cfg.meters_per_pixel,
            ],
            dtype=np.float32,
        )
        v = p2 - p1
        v_norm = float(np.hypot(v[0], v[1]))
        pts = cnt[:, 0, :].astype(np.float32)
        if v_norm > 1e-6 and len(pts) > 0:
            # Perpendicular distance from point to line (p1->p2) in pixel units.
            cross = v[0] * (pts[:, 1] - p1[1]) - v[1] * (pts[:, 0] - p1[0])
            perp_dist_px = np.abs(cross) / v_norm
            max_perp_px = float(np.max(perp_dist_px))
        else:
            max_perp_px = 0.0

        wall_thickness_m = best_wall.thickness if best_wall.thickness > 0 else cfg.default_thickness
        wall_thickness_px = wall_thickness_m / cfg.meters_per_pixel
        swing_thresh_px = max(cfg.door_swing_perp_min_m / cfg.meters_per_pixel, cfg.door_swing_perp_factor * wall_thickness_px)
        door_by_swing = max_perp_px >= swing_thresh_px

        if door_by_swing:
            opening_type = "door"
            sill_m = 0.0
            height_m = cfg.default_door_height_m
        elif width_m >= cfg.long_opening_m:
            opening_type = "window"
            sill_m = 0.0
            height_m = cfg.default_door_height_m
        else:
            opening_type = "window"
            sill_m = cfg.default_window_sill_m
            height_m = cfg.default_window_height_m

        # Project centroid to wall axis, then convert to "edge closest to wall.start".
        w_line = LineString([best_wall.start, best_wall.end])
        p_world = Point(cx * cfg.meters_per_pixel, (mask_open.shape[0] - cy) * cfg.meters_per_pixel)
        at_center_m = w_line.project(p_world)
        wall_len_m = w_line.length
        edge_a = at_center_m - (width_m / 2.0)
        edge_b = at_center_m + (width_m / 2.0)
        if wall_len_m > 0:
            edge_a = min(max(edge_a, 0.0), wall_len_m)
            edge_b = min(max(edge_b, 0.0), wall_len_m)
            at_m = min(edge_a, edge_b)
        else:
            at_m = 0.0
        
        openings.append(Opening(
            id=f"o_{i:03d}",
            type=opening_type,
            wall_id=best_wall.id,
            at=round(at_m, 3),
            width=round(width_m, 3),
            height=height_m,
            sill=sill_m,
            px_center=(int(round(cx)), int(round(cy))),
        ))
    return openings

# --- PHASE 7: EMIT ---
def emit_dsl(walls: List[Wall], openings: List[Opening], cfg: Config):
    lines = ['level("L1").elev(0)']
    walls.sort(key=lambda x: x.id)
    
    for w in walls:
        line = (
            f'wall("{w.id}").from({w.start[0]:.3f},{w.start[1]:.3f})'
            f'.to({w.end[0]:.3f},{w.end[1]:.3f}).t({w.thickness:.2f}).h({w.height})'
        )
        if abs(w.axis_offset) >= cfg.axis_offset_min_m:
            line += f'.off({w.axis_offset:.3f})'
        lines.append(line)
        
    openings.sort(key=lambda x: (x.wall_id, x.at))
    for o in openings:
        if o.type == "window":
            lines.append(
                f'{o.type}("{o.id}").in("{o.wall_id}").at({o.at:.3f})'
                f'.w({o.width:.3f}).h({o.height}).sill({o.sill})'
            )
        else:
            lines.append(
                f'{o.type}("{o.id}").in("{o.wall_id}").at({o.at:.3f})'
                f'.w({o.width:.3f}).h({o.height})'
            )
    return "\n".join(lines)


# --- PHASE 7b: FURNITURE DECOMPOSITION ---

def get_rotated_bbox_area(cnt):
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    return w * h, rect

def split_blob_recursive(mask_roi, pts_roi, roi_offset, min_solidity=0.75, depth=0, max_depth=2):
    """
    Recursively splits a contour point set if its solidity (Area/BBoxArea) is too low.
    Uses a sweep-line approach to find a split that minimizes the sum of child bounding boxes.
    
    mask_roi: The binary mask of the ROI (for fast lookup, optional or used for exact cuts)
    pts_roi: specific points of the contour in ROI coordinates
    roi_offset: (x, y) of the ROI in the original image (to return global coords)
    """
    if len(pts_roi) < 3:
        return []

    # Compute Solidity
    bbox_area, rect = get_rotated_bbox_area(pts_roi)
    
    # Estimate actual contour area (approx) or use minAreaRect as baseline
    cnt_area = cv2.contourArea(pts_roi)
    
    if bbox_area < 1e-3: 
        return [pts_roi + roi_offset] # Point or line

    if bbox_area < 1e-3: 
        return [pts_roi + roi_offset] # Point or line

    ratio = cnt_area / bbox_area
    
    # Base case: Solid enough or too deep
    if ratio >= min_solidity or depth >= max_depth:
        # Return as global coordinates
        return [pts_roi + roi_offset]



    # --- Attempt Split ---
    # We sweep X and Y in the ROI to find a cut that minimizes (AreaA + AreaB)
    
    x_min, y_min = pts_roi.min(axis=0)[0]
    x_max, y_max = pts_roi.max(axis=0)[0]
    w = x_max - x_min
    h = y_max - y_min
    
    best_score = float('inf')
    best_split = None # ('x'|'y', local_coord, pts_a, pts_b)
    
    step_x = max(1, int(w / 10))
    step_y = max(1, int(h / 10))
    
    # Optimization: Only split if we have enough range
    if w > 5:
        for i in range(x_min + step_x, x_max, step_x):
            mask_left = pts_roi[:, 0, 0] < i
            pts_l = pts_roi[mask_left]
            pts_r = pts_roi[~mask_left]
            
            if len(pts_l) < 3 or len(pts_r) < 3: continue
            
            # Fast bbox approximation
            area_l = get_rotated_bbox_area(pts_l)[0]
            area_r = get_rotated_bbox_area(pts_r)[0]
            
            if (area_l + area_r) < best_score:
                best_score = area_l + area_r
                best_split = ('x', i, pts_l, pts_r)

    if h > 5:
        for i in range(y_min + step_y, y_max, step_y):
            mask_top = pts_roi[:, 0, 1] < i
            pts_t = pts_roi[mask_top]
            pts_b = pts_roi[~mask_top]
            
            if len(pts_t) < 3 or len(pts_b) < 3: continue
            
            area_t = get_rotated_bbox_area(pts_t)[0]
            area_b = get_rotated_bbox_area(pts_b)[0]
            
            if (area_t + area_b) < best_score:
                best_score = area_t + area_b
                best_split = ('y', i, pts_t, pts_b)

    # Threshold: Split must improve area efficiency
    # If best split is worse than current bbox (unlikely) or marginal improvement?
    # Actually, we rely on the solidity check of children to stop recursion.
    # But if the split doesn't strictly reduce the bbox sum significantly compared to parent, maybe don't?
    # Simple check: (AreaL + AreaR) < 0.9 * AreaParent?
    
    if best_split and best_score < bbox_area * 0.95:
        # Recurse
        # Note: We are splitting the POINT CLOUD. Bounding Box of point cloud is convex hull.
        # We lose concavity info inside the sub-regions, but for furniture BBox fitting that's fine.
        pts_a = best_split[2]
        pts_b = best_split[3]
        
        # To call recursively, we need to treat them as contours. 
        # convexHull is a good approximation for the 'contour' of the point cloud
        hull_a = cv2.convexHull(pts_a)
        hull_b = cv2.convexHull(pts_b)
        
        res = []
        res.extend(split_blob_recursive(mask_roi, hull_a, roi_offset, min_solidity, depth+1, max_depth))
        res.extend(split_blob_recursive(mask_roi, hull_b, roi_offset, min_solidity, depth+1, max_depth))
        return res

    # Failed to find good split
    return [pts_roi + roi_offset]


def detect_furniture(img_rgb: np.ndarray, cfg: Config) -> List[Furniture]:
    import numpy as np
    import cv2
    
    # Use HSV for more robust color segmentation (matches api/furniture.py)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Color definitions (HSV ranges) - mapped to mask_prompt.md colors
    # Bed: RED, Cabinet: PURPLE, Sofa: CYAN, Table: GREEN
    colors = {
        "bed": [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),   # Red (lower)
            (np.array([170, 50, 50]), np.array([180, 255, 255])) # Red (upper)
        ],
        "table": [
            (np.array([40, 50, 50]), np.array([80, 255, 255]))  # Green
        ],
        "chair": [
            (np.array([85, 50, 50]), np.array([95, 255, 255]))  # Cyan
        ],
        "cabinet": [
            (np.array([140, 50, 50]), np.array([160, 255, 255])) # Purple
        ]
    }
    
    furniture_items = []
    h_px, w_px = img_rgb.shape[:2]
    
    for f_type, ranges in colors.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        
        # Morphological operations to clean up mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 50: # Minimum area filter
                continue
            
            # --- DECOMPOSITION ---
            # Extract ROI for this contour to speed up processing
            x, y, w, h = cv2.boundingRect(cnt)
            # Make point set relative to the bounding box (but we pass offset)
            # Actually our recursive function handles ROI well if we pass points?
            # Let's just pass points.
            
            # The contour points are already global (N, 1, 2).
            # But recursive split does coordinate scanning. 
            # It's cleaner to shift to 0,0 for numerical stability / smaller loops if we used mask scanning.
            # But we use point scanning. So global coords are fine, BUT
            # split_blob_recursive implementation above used ``pts_roi`` and ``roi_offset``.
            # Let's map to local ROI 0,0
            
            pts_local = cnt - np.array([[[x, y]]], dtype=np.int32)
            roi_offset = np.array([[[x, y]]], dtype=np.int32)
            
            # Heuristic: Only decompose 'shelf-like' or 'complex' items? 
            # Cabinets (L-shaped) need it. Tables/Beds usually convex.
            # But "Bed + Nightstand" might be one blob.
            # Let's apply to all, relying on solidity check.
            
            sub_contours = split_blob_recursive(None, pts_local, roi_offset, min_solidity=0.75, depth=0, max_depth=2)
            
            for sub_i, sub_cnt in enumerate(sub_contours):
                rect = cv2.minAreaRect(sub_cnt)
                (center_px_x, center_px_y), (w, h), angle = rect
                
                # Convert to world meters
                x_m = center_px_x * cfg.meters_per_pixel
                y_m = (h_px - center_px_y) * cfg.meters_per_pixel # Flip Y for world coords
                
                # User example: width is longer side, length is shorter side
                width_m = max(w, h) * cfg.meters_per_pixel
                length_m = min(w, h) * cfg.meters_per_pixel
                
                # Adjust rotation to match "width" as longest side
                final_rotation = angle
                if w < h:
                     final_rotation = angle + 90
                
                # Normalize rotation to [0, 180) or [0, 360)? 
                final_rotation = final_rotation % 360
                
                # ID scheme: type_idx_subidx
                f_id = f"{f_type}_{i}"
                if len(sub_contours) > 1:
                    f_id += f"_{sub_i}"
                
                furniture_items.append(Furniture(
                    id=f_id,
                    type=f_type,
                    x=round(x_m, 3),
                    y=round(y_m, 3),
                    width=round(width_m, 3),
                    length=round(length_m, 3),
                    rotation=round(final_rotation, 2)
                ))
            
    return furniture_items

def emit_json(walls: List[Wall], openings: List[Opening], furniture: List[Furniture], canvas_dims: Tuple[float, float]) -> str:
    import json
    
    # Sort for deterministic output
    walls.sort(key=lambda x: x.id)
    openings.sort(key=lambda x: (x.wall_id, x.at))
    furniture.sort(key=lambda x: x.id)
    
    # Group openings by wall_id
    openings_by_wall = defaultdict(list)
    for o in openings:
        openings_by_wall[o.wall_id].append(o)
        
    walls_output = []
    for w in walls:
        w_openings = []
        for o in openings_by_wall[w.id]:
            w_openings.append({
                "id": o.id,
                "type": o.type,
                "t": round(o.at, 3),
                "width": round(o.width, 3),
                "height": round(o.height, 3),
                "z": round(o.sill, 3)
            })
            
        walls_output.append({
            "id": w.id,
            "start": {"x": round(w.start[0], 3), "y": round(w.start[1], 3)},
            "end": {"x": round(w.end[0], 3), "y": round(w.end[1], 3)},
            "thickness": round(w.thickness, 2),
            "axisOffset": round(w.axis_offset, 3),
            "openings": w_openings
        })
        
    furniture_output = []
    for f in furniture:
        furniture_output.append({
            "id": f.id,
            "type": f.type,
            "x": f.x,
            "y": f.y,
            "width": f.width,
            "length": f.length,
            "rotation": f.rotation
        })
        
    output = {
        "canvas": {
            "width": round(canvas_dims[0], 3),
            "height": round(canvas_dims[1], 3)
        },
        "walls": walls_output,
        "furniture": furniture_output
    }
    return json.dumps(output, indent=2)

def build_walls_from_masks(mask_union, dist_map, dims, cfg: Config, debug_dir: Optional[str]):
    skel_graph = build_skeleton_graph(mask_union, debug_dir, cfg)
    raw_vectors = graph_to_vectors(skel_graph, debug_dir, cfg)
    snapped_walls = gravity_snap(raw_vectors, debug_dir, cfg)
    final_topology = rebuild_topology(snapped_walls, cfg)
    walls_raw = process_attributes(final_topology, dist_map, dims[0], cfg)
    walls_merged = consolidate_walls(walls_raw, cfg)
    return walls_merged

def compute_axis_offset_from_baseline(
    snapped: Wall,
    baseline_walls: List[Wall],
    cfg: Config,
    min_overlap_m: Optional[float] = None,
    sigma_m: Optional[float] = None,
) -> float:
    if min_overlap_m is None:
        min_overlap_m = float(cfg.axis_offset_min_overlap_m)
    if sigma_m is None:
        sigma_m = float(cfg.axis_offset_sigma_m)

    dx = snapped.end[0] - snapped.start[0]
    dy = snapped.end[1] - snapped.start[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0

    # Left-hand normal of snapped wall direction.
    n_left = (-dy / length, dx / length)

    is_horiz = abs(dx) >= abs(dy)
    if is_horiz:
        axis_s = (snapped.start[1] + snapped.end[1]) * 0.5
        s0, s1 = sorted([snapped.start[0], snapped.end[0]])
    else:
        axis_s = (snapped.start[0] + snapped.end[0]) * 0.5
        s0, s1 = sorted([snapped.start[1], snapped.end[1]])

    def overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    # Allow matching up to the snap radius (in meters) plus a small buffer.
    axis_match_tol_m = max(
        float(cfg.gravity_snap_axis_snap_tol_m / cfg.meters_per_pixel) * float(cfg.meters_per_pixel) + 0.05,
        0.25,
    )

    # Weighted average over baseline candidates:
    # weight = overlap_len_m * exp(-(abs_axis_diff / sigma_m)^2)
    # This avoids tiny-overlap stubs "winning" purely because they are closest in axis.
    candidates: List[Tuple[float, float, float]] = []  # (abs_axis_diff, axis_diff, overlap_len_m)
    for b in baseline_walls:
        bdx = b.end[0] - b.start[0]
        bdy = b.end[1] - b.start[1]
        b_is_horiz = abs(bdx) >= abs(bdy)
        if b_is_horiz != is_horiz:
            continue

        if is_horiz:
            axis_b = (b.start[1] + b.end[1]) * 0.5
            b0, b1 = sorted([b.start[0], b.end[0]])
        else:
            axis_b = (b.start[0] + b.end[0]) * 0.5
            b0, b1 = sorted([b.start[1], b.end[1]])

        ov = overlap_len(s0, s1, b0, b1)
        axis_diff = axis_b - axis_s
        if ov < min_overlap_m:
            continue

        if abs(axis_diff) > axis_match_tol_m:
            continue
        candidates.append((abs(axis_diff), axis_diff, ov))

    if not candidates:
        return 0.0

    sigma_m = max(float(sigma_m), 1e-6)
    total_w = 0.0
    total_off = 0.0
    for abs_axis_diff, axis_diff, ov in candidates:
        weight = float(ov) * math.exp(-((float(abs_axis_diff) / sigma_m) ** 2))
        if weight <= 0.0:
            continue
        # Convert axis coordinate delta into a world XY shift and project onto snapped left normal.
        if is_horiz:
            shift = (0.0, axis_diff)
        else:
            shift = (axis_diff, 0.0)
        off = shift[0] * n_left[0] + shift[1] * n_left[1]
        total_off += float(off) * weight
        total_w += weight

    if total_w <= 1e-9:
        return 0.0

    axis_offset_raw = total_off / total_w
    axis_offset = axis_offset_raw
    if abs(axis_offset) < cfg.axis_offset_min_m:
        axis_offset = 0.0
    return axis_offset

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--debug-dir")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.02,
        help="Meters-per-pixel scale for converting pixels to meters.",
    )
    parser.add_argument(
        "--format",
        choices=["dsl", "json"],
        default="dsl",
        help="Output format: 'dsl' (default) or 'json'.",
    )
    parser.add_argument(
        "--real-json",
        default=None,
        help="Path to JSON file with real dimensions ('real-x', 'real-y'). Overrides --scale.",
    )
    args = parser.parse_args()
    
    # Scale resolution
    scale = args.scale
    if args.real_json:
        # Import here to avoid circular dependencies if any, though likely safe at top
        from extract_scale import (
            load_real_dims_from_json,
            derive_isotropic_meters_per_pixel,
        )

        try:
            real_width_m, real_height_m = load_real_dims_from_json(args.real_json)
            
            # We need image dimensions to compute scale. 
            # load_and_preprocess returns (mask_union, mask_wall, mask_open, dist_map, dims)
            # but we need it *before* Config init? 
            # Actually Config needs meters_per_pixel. 
            # But load_and_preprocess needs Config.
            # Catch-22?
            # 
            # Let's look at load_and_preprocess.
            # It calls create_robust_union_mask -> get_opening_jamb_context -> (uses cfg.opening_search_dilate_px etc).
            # It uses cfg.opening_bridge_px.
            # These are PIXEL values.
            # 
            # Does load_and_preprocess use meters_per_pixel?
            # It scans dist_map but ... wait.
            # 
            # _resample_thickness_from_dist_map uses cfg.meters_per_pixel.
            # build_walls_from_masks uses cfg.
            # 
            # load_and_preprocess itself only uses cfg for pixel-based kernels (opening_bridge_px).
            # It DOES NOT look like it uses meters_per_pixel.
            # So we can run load_and_preprocess with a dummy scale (or args.scale), 
            # THEN compute the real scale, update cfg, and proceed.
            
            # Initial load with default/CLI scale (might be ignored later)
            cfg_initial = Config(meters_per_pixel=scale)
            if args.debug_dir:
                os.makedirs(args.debug_dir, exist_ok=True)
                
            mask_union, mask_wall, mask_open, dist_map, dims, img_rgb = load_and_preprocess(args.input, args.debug_dir, cfg_initial)
            
            h, w = dims
            # wall_bbox_px is needed for derive_isotropic_meters_per_pixel.
            # extract_scale uses wall_mask_from_image_rgb -> wall_bbox_from_mask.
            # load_and_preprocess returns mask_wall, we can use that.
            
            ys, xs = np.nonzero(mask_wall > 0)
            if len(xs) > 0 and len(ys) > 0:
                wall_bbox_px = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                
                mpp, _, _, _, _ = derive_isotropic_meters_per_pixel(
                    wall_bbox_px=wall_bbox_px,
                    real_width_m=real_width_m,
                    real_height_m=real_height_m,
                )
                scale = mpp
                print(f"Computed scale from real dims: {scale:.6f} m/px")
            else:
                print("Warning: No wall pixels found, cannot compute scale from real dims. Using default.")

        except Exception as e:
            print(f"Error computing scale from real dims: {e}. Falling back to --scale {scale}")
            # Fallback or error? Implementation plan implies we should use real dims if present.
            # If explicit JSON is provided but fails, we should probably fail hard or warn.
            # For now, print error and proceed with fallback to avoid total crash?
            # Actually, better to raise if user explicitly asked for it. 
            # But let's stick to the prompt's preference for robustness? 
            # "If real_x... are provided, they take precedence"
            raise e

    cfg = Config(meters_per_pixel=scale)
    
    # If we didn't run load_and_preprocess yet (no real_json), run it now.
    # If we DID run it, we already have the masks.
    if not args.real_json:
        if args.debug_dir:
            os.makedirs(args.debug_dir, exist_ok=True)
        mask_union, mask_wall, mask_open, dist_map, dims, img_rgb = load_and_preprocess(args.input, args.debug_dir, cfg)

    # Rest of the pipeline matches main...
    
    # Two-pass axis offset:
    baseline_cfg = replace(
        cfg,
        gravity_snap_axis_snap_tol_m=0.0,
        consolidate_axis_cluster_tol_m=0.0,
    )
    baseline_walls = build_walls_from_masks(mask_union, dist_map, dims, baseline_cfg, debug_dir=None)

    # Build the snapped walls (with full debug output).
    walls_merged = build_walls_from_masks(mask_union, dist_map, dims, cfg, args.debug_dir)

    # Assign axis_offset for each snapped wall as the delta to the baseline axes.
    for w in walls_merged:
        w.axis_offset = compute_axis_offset_from_baseline(w, baseline_walls, cfg)

    # Re-sample thickness after axis_offset is known (helps walls with large offsets).
    _resample_thickness_from_dist_map(walls_merged, dist_map, dims[0], cfg)

    # 3. Openings (Map to merged)
    openings = extract_openings(mask_open, mask_wall, walls_merged, cfg)
    
    # 4. Furniture (If JSON requested)
    furniture = []
    if args.format == "json":
        furniture = detect_furniture(img_rgb, cfg)
    
    with open(args.out, "w") as f:
        if args.format == "json":
            h_px, w_px = dims
            canvas_width_m = w_px * scale
            canvas_height_m = h_px * scale
            f.write(emit_json(walls_merged, openings, furniture, (canvas_width_m, canvas_height_m)))
        else:
            f.write(emit_dsl(walls_merged, openings, cfg))
    print(f"Done! Written to {args.out}")

    if args.debug_dir:
        orig = cv2.imread(args.input)
        h, w_img = dims

        def to_px(pt_m: Tuple[float, float]) -> Tuple[int, int]:
            return (
                int(round(pt_m[0] / cfg.meters_per_pixel)),
                int(round(h - pt_m[1] / cfg.meters_per_pixel)),
            )

        def draw_dashed_line(
            img: np.ndarray,
            p1: Tuple[int, int],
            p2: Tuple[int, int],
            color: Tuple[int, int, int],
            thickness: int = 1,
            dash_len: int = 10,
            gap_len: int = 6,
        ) -> None:
            x1, y1 = p1
            x2, y2 = p2
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist < 1.0:
                return
            vx = (x2 - x1) / dist
            vy = (y2 - y1) / dist
            cur = 0.0
            while cur < dist:
                seg_end = min(cur + dash_len, dist)
                sx = int(round(x1 + vx * cur))
                sy = int(round(y1 + vy * cur))
                ex = int(round(x1 + vx * seg_end))
                ey = int(round(y1 + vy * seg_end))
                cv2.line(img, (sx, sy), (ex, ey), color, thickness)
                cur += dash_len + gap_len
        for w in walls_merged:
            p1 = to_px(w.start)
            p2 = to_px(w.end)
            cv2.line(orig, p1, p2, (0, 255, 0), 2)

            # Draw original (pre-snap) axis implied by axis_offset: snapped + normal_left * axis_offset
            if abs(w.axis_offset) >= cfg.axis_offset_min_m:
                dx = w.end[0] - w.start[0]
                dy = w.end[1] - w.start[1]
                length = math.hypot(dx, dy)
                if length > 1e-9:
                    nx = -dy / length
                    ny = dx / length
                    s1_m = (w.start[0] + nx * w.axis_offset, w.start[1] + ny * w.axis_offset)
                    s2_m = (w.end[0] + nx * w.axis_offset, w.end[1] + ny * w.axis_offset)
                    s1 = to_px(s1_m)
                    s2 = to_px(s2_m)
                    draw_dashed_line(orig, s1, s2, (255, 0, 255), thickness=1)

                    mid_m = ((w.start[0] + w.end[0]) * 0.5, (w.start[1] + w.end[1]) * 0.5)
                    mid_off_m = (mid_m[0] + nx * w.axis_offset, mid_m[1] + ny * w.axis_offset)
                    mid = to_px(mid_m)
                    mid_off = to_px(mid_off_m)
                    cv2.arrowedLine(orig, mid, mid_off, (0, 165, 255), 1, tipLength=0.35)
                    cv2.putText(
                        orig,
                        f"{w.axis_offset:+.3f}m",
                        (mid_off[0] + 4, mid_off[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
                        1,
                    )
            # Place label at 10% along the wall from the start point
            lx = int(round(p1[0] + 0.2 * (p2[0] - p1[0])))
            ly = int(round(p1[1] + 0.2 * (p2[1] - p1[1])))
            cv2.putText(orig, w.id, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        for o in openings:
            if o.px_center is None:
                continue
            cx, cy = o.px_center
            color = (255, 255, 0) if o.type == "window" else (0, 255, 255)
            cv2.circle(orig, (cx, cy), 3, color, -1)
        cv2.imwrite(f"{args.debug_dir}/05_final_overlay.png", orig)

if __name__ == "__main__":
    main()
