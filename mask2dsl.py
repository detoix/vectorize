import argparse
import os
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, substring
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates

# --- CONFIGURATION ---
@dataclass
class Config:
    meters_per_pixel: float = 0.02
    wall_height: float = 2.7
    default_thickness: float = 0.20 # Used if sampling fails
    snap_tol_px: float = 10.0      # Distance to snap to major axes
    collinear_tol_deg: float = 15.0 # Tolerance for merging lines (180 +/- 5)
    ortho_tol_deg: float = 15.0     # Tolerance for forcing 0/90 degrees
    min_spur_length_px: int = 15   # Prune skeletal dead ends
    opening_bridge_px: int = 3     # Dilation size for Blue Bridge
    epsilon_rdp: float = 3.0       # RDP simplification epsilon
    opening_search_dilate_px: int = 15
    corner_gap_deviation_deg: float = 30.0

    long_opening_m: float = 1.9
    door_circularity_thresh: float = 0.25
    door_swing_perp_factor: float = 1.5
    door_swing_perp_min_px: float = 8.0
    default_door_height_m: float = 2.1
    default_window_height_m: float = 1.2
    default_window_sill_m: float = 0.9
    opening_search_dilate_px: int = 15
    corner_gap_deviation_deg: float = 30.0

    long_opening_m: float = 2.5
    door_circularity_thresh: float = 0.25
    door_swing_perp_factor: float = 1.5
    door_swing_perp_min_px: float = 8.0
    default_door_height_m: float = 2.1
    default_window_height_m: float = 1.2
    default_window_sill_m: float = 0.9

# --- DATA STRUCTURES ---
@dataclass
class Wall:
    id: str
    start: Tuple[float, float] # (x, y) in World Meters
    end: Tuple[float, float]
    thickness: float = 0.2
    height: float = 2.7
    px_geom: Optional[LineString] = None # Original pixel geometry

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

# --- PHASE 0: IO & MASKS ---
def get_opening_jamb_context(cnt, mask_wall, cfg: Config):
    """
    Returns key information about the two biggest nearby wall components (jambs) around an opening contour.
    Used both for robust bridging and later for opening classification (avoid duplicated logic).
    """
    single_mask = np.zeros_like(mask_wall)
    cv2.drawContours(single_mask, [cnt], -1, 255, -1)
    search_zone = cv2.dilate(
        single_mask,
        np.ones((cfg.opening_search_dilate_px, cfg.opening_search_dilate_px), np.uint8),
    )
    nearby_walls = cv2.bitwise_and(mask_wall, search_zone)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(nearby_walls)
    if num_labels < 3:
        return None

    sorted_indices = np.argsort(stats[1:, 4])[::-1] + 1
    jamb_a_idx = sorted_indices[0]
    jamb_b_idx = sorted_indices[1]

    pt_a = (int(centroids[jamb_a_idx][0]), int(centroids[jamb_a_idx][1]))
    pt_b = (int(centroids[jamb_b_idx][0]), int(centroids[jamb_b_idx][1]))

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center_opening = (cx, cy)
    else:
        center_opening = (int(np.mean(cnt[:, 0, 0])), int(np.mean(cnt[:, 0, 1])))

    return {
        "labels": labels,
        "jamb_a_idx": jamb_a_idx,
        "jamb_b_idx": jamb_b_idx,
        "pt_a": pt_a,
        "pt_b": pt_b,
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
    half_len = int(round(max(12.0, min(250.0, 0.9 * max_dim))))

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

    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue

        ctx = get_opening_jamb_context(cnt, mask_wall, cfg)
        if ctx is None:
            continue

        pt_a = ctx["pt_a"]
        pt_b = ctx["pt_b"]
        center_opening = ctx["center_opening"]

        # Only split skew/corner gaps.
        is_skew_gap = min_axis_deviation_deg(pt_a, pt_b) > cfg.corner_gap_deviation_deg
        if not is_skew_gap:
            continue

        intersection = compute_corner_intersection(
            labels=ctx["labels"],
            jamb_a_idx=ctx["jamb_a_idx"],
            jamb_b_idx=ctx["jamb_b_idx"],
            pt_a=pt_a,
            pt_b=pt_b,
            center_opening=center_opening,
        )
        if not intersection:
            continue

        p1, p2 = cut_corner_opening_in_mask(mask_out, cnt, intersection, pt_a, pt_b)
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
    
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 10: continue

        ctx = get_opening_jamb_context(cnt, mask_wall, cfg)
        if ctx is None:
            continue

        labels = ctx["labels"]
        jamb_a_idx = ctx["jamb_a_idx"]
        jamb_b_idx = ctx["jamb_b_idx"]
        pt_a = ctx["pt_a"]
        pt_b = ctx["pt_b"]
        center_opening = ctx["center_opening"]

        is_corner = False
        intersection = None

        # If the angle is more than the threshold off-axis, it's a diagonal (corner) gap.
        is_skew_gap = min_axis_deviation_deg(pt_a, pt_b) > cfg.corner_gap_deviation_deg

        if is_skew_gap:
            intersection = compute_corner_intersection(
                labels=labels,
                jamb_a_idx=jamb_a_idx,
                jamb_b_idx=jamb_b_idx,
                pt_a=pt_a,
                pt_b=pt_b,
                center_opening=center_opening,
            )
            if intersection:
                is_corner = True

        if is_corner and intersection:
            cv2.line(union_mask, pt_a, intersection, 255, thickness=4)
            cv2.line(union_mask, intersection, pt_b, 255, thickness=4)

            cv2.line(debug_img, pt_a, intersection, (0, 255, 0), 2)
            cv2.line(debug_img, intersection, pt_b, (0, 255, 0), 2)
            cv2.circle(debug_img, intersection, 3, (255, 0, 0), -1) # Mark the corner

        else:
            cv2.line(union_mask, pt_a, pt_b, 255, thickness=4)
            cv2.line(debug_img, pt_a, pt_b, (0, 0, 255), 2)
            
    if debug_dir:
        cv2.imwrite(f"{debug_dir}/00_robust_bridges.png", debug_img)
        
    return union_mask

def load_and_preprocess(path: str, debug_dir: str, cfg: Config):
    print(f"Phase 0: Loading {path}...")
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(f"Could not load {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask_wall = cv2.inRange(img_rgb, np.array([250, 250, 250]), np.array([255, 255, 255]))
    # Robust opening threshold: use HSV so anti-aliased/compressed "blue" stays connected.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 30, 50], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    mask_open_raw = cv2.inRange(hsv, lower, upper)
    # Fill tiny gaps inside swings caused by threshold holes.
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_open_raw = cv2.morphologyEx(mask_open_raw, cv2.MORPH_CLOSE, kernel_clean)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.opening_bridge_px, cfg.opening_bridge_px))
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

    return mask_union, mask_wall, mask_open, dist_map, img_rgb.shape[:2]

# --- PHASE 1: SKELETON ---
def build_skeleton_graph(mask_union, debug_dir, cfg):
    print("Phase 1: Skeletonizing...")
    binary = mask_union > 127
    skel = skeletonize(binary)
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

    # Prune Spurs
    # Walk outward from every degree-1 node until we hit a junction (deg>=3) or the path gets long enough.
    # Using a `prev` pointer is critical here; relying on neighbor ordering makes pruning nondeterministic.
    while True:
        pruned_any = False
        deg1_nodes = [n for n in G.nodes() if G.degree(n) == 1]
        for start in deg1_nodes:
            if not G.has_node(start) or G.degree(start) != 1:
                continue

            path = [start]
            length = 0.0
            prev = None
            curr = start
            alignment = 1.0
            effective_length = 0.0

            while True:
                nbrs = list(G.neighbors(curr))
                next_candidates = [n for n in nbrs if n != prev]
                if not next_candidates:
                    break

                nxt = next_candidates[0]
                length += float(G[curr][nxt].get("weight", 1.0))
                path.append(nxt)
                prev, curr = curr, nxt

                alignment = spur_alignment_score(path)
                effective_length = length * alignment
                if G.degree(curr) != 2 or (effective_length + 1e-9) >= float(cfg.min_spur_length_px):
                    break

            # Keep longer spurs. Using `effective_length` biases pruning toward skew/diagonal spurs.
            if (effective_length + 1e-9) >= float(cfg.min_spur_length_px):
                continue

            # Only remove true spurs: a short degree-1 chain that attaches into a junction.
            if G.has_node(curr) and G.degree(curr) >= 3:
                for p in path[:-1]:
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
            simplified = line.simplify(cfg.epsilon_rdp, preserve_topology=True)
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
        for axis in axes_y:
            if abs(axis['val'] - y) < cfg.snap_tol_px:
                w['start'][1] = axis['val']; w['end'][1] = axis['val']; snapped = True; break
        if not snapped: axes_y.append({'val': y})

    # Snap Verticals
    v_walls = [w for w in walls if w['orient'] == 'V']
    v_walls.sort(key=lambda x: x['len'], reverse=True)
    axes_x = []
    for w in v_walls:
        x = w['start'][0]
        snapped = False
        for axis in axes_x:
            if abs(axis['val'] - x) < cfg.snap_tol_px:
                w['start'][0] = axis['val']; w['end'][0] = axis['val']; snapped = True; break
        if not snapped: axes_x.append({'val': x})

    return walls

# --- PHASE 4: TOPOLOGY RECONSTRUCTION ---
def rebuild_topology(walls, cfg):
    print("Phase 4: Rebuilding Topology...")
    points = []
    for w in walls: points.extend([w['start'], w['end']])
    if not points: return []

    tree = cKDTree(points)
    pairs = tree.query_pairs(cfg.snap_tol_px)
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
            'px_geom': w['geom'], 'px_len': w['len']
        })
    return final_walls

# --- PHASE 5: ATTRIBUTES ---
def process_attributes(walls, dist_map, h_img, cfg):
    print("Phase 5: Attributes & Scale...")
    wall_objs = []
    
    for i, w in enumerate(walls):
        # 1. Thickness Sampling (Ignore Bridge Zeros)
        line = w['px_geom']
        length = line.length
        num_samples = max(5, int(length)) 
        samples = [line.interpolate(n) for n in np.linspace(0, length, num_samples)]
        
        vals = []
        for p in samples:
            if 0 <= p.y < dist_map.shape[0] and 0 <= p.x < dist_map.shape[1]:
                val = dist_map[int(p.y), int(p.x)]
                if val > 1.0: vals.append(val) # Filter out the bridge pixels
                
        if vals:
            thickness_px = np.median(vals) * 2
        else:
            thickness_px = (cfg.default_thickness / cfg.meters_per_pixel)
            
        thickness_m = thickness_px * cfg.meters_per_pixel
        # Round to nearest 5cm
        thickness_m = round(thickness_m / 0.05) * 0.05
        
        # 2. Coordinate Transform
        sx, sy = w['start']; ex, ey = w['end']
        wx1 = sx * cfg.meters_per_pixel
        wy1 = (h_img - sy) * cfg.meters_per_pixel
        wx2 = ex * cfg.meters_per_pixel
        wy2 = (h_img - ey) * cfg.meters_per_pixel
        
        if wx1 > wx2 or (abs(wx1 - wx2) < 1e-4 and wy1 > wy2):
            wx1, wy1, wx2, wy2 = wx2, wy2, wx1, wy1
            
        wall_objs.append(Wall(
            id=f"tmp_{i}", # Temp ID
            start=(wx1, wy1), end=(wx2, wy2),
            thickness=thickness_m, height=cfg.wall_height,
            px_geom=w['px_geom']
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
                if axis_diff > 0.15: continue
                
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
                if axis_diff > 0.15: continue
                
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
        # Tolerance: 0.15m (15cm) to catch slightly jagged walls
        TOL = 0.15 
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
        
        # Debug: Print cleaned group size
        if len(group) > 0 and abs(group[0].start[is_horiz and 1 or 0] - 2.419) < 0.05:
            print(f"DEBUG MERGE_GROUP x=2.419. Count: {len(group)}")
            for w in group:
                print(f"  W: {w.start[idx]:.3f} -> {w.end[idx]:.3f}, t={w.thickness:.2f}")

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
            same_thick = abs(current.thickness - next_w.thickness) < 0.05
            
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
                        if abs(current.start[0] - check_w.start[0]) < 0.1: continue
                    else:
                        if abs(current.start[1] - check_w.start[1]) < 0.1: continue

                d_s = math.hypot(check_w.start[0]-join_pt[0], check_w.start[1]-join_pt[1])
                d_e = math.hypot(check_w.end[0]-join_pt[0], check_w.end[1]-join_pt[1])
                if d_s < 0.2 or d_e < 0.2: 
                    num_near += 1
            
            if num_near > 0: is_simple_joint = False
            
            # Thickness policy:
            # For collinear segments we merge whenever geometry/topology allows, and keep the thicker value.
            # This treats thickness differences as sampling noise (or conservative "max thickness wins").
            if gap < 0.2 and is_simple_joint:
                # Merge
                current.thickness = max(current.thickness, next_w.thickness)
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
        tol = 0.2        # endpoint must be within 20cm of the wall centerline
        end_margin = 0.2 # don't split within 20cm of the wall endpoints
        min_seg_len = 0.05

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
                        if abs(pt[1] - y) > tol:
                            continue
                        if pt[0] < x0 - tol or pt[0] > x1 + tol:
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
                    result.append(Wall(
                        id=w.id,
                        start=(a, y),
                        end=(b, y),
                        thickness=w.thickness,
                        height=w.height,
                        px_geom=None,
                    ))
            else:
                x = w.start[0]
                y0, y1 = sorted([w.start[1], w.end[1]])
                split_ys = set()
                for other in walls_in:
                    if other is w or not is_horizontal(other):
                        continue
                    for pt in (other.start, other.end):
                        if abs(pt[0] - x) > tol:
                            continue
                        if pt[1] < y0 - tol or pt[1] > y1 + tol:
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
                    result.append(Wall(
                        id=w.id,
                        start=(x, a),
                        end=(x, b),
                        thickness=w.thickness,
                        height=w.height,
                        px_geom=None,
                    ))
        return result

    final_walls = split_walls_at_t_junctions(final_walls)
        
    for i, w in enumerate(final_walls): w.id = f"w_{i:03d}"
    return final_walls

# --- PHASE 6: OPENINGS ---
def extract_openings(mask_open, mask_wall, walls: List[Wall], cfg: Config):
    print("Phase 6: Projecting Openings...")
    contours, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    openings = []
    
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 10: continue
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
        
        # 0.5m tolerance
        if min_dist > 0.5: continue 
        
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
        swing_thresh_px = max(cfg.door_swing_perp_min_px, cfg.door_swing_perp_factor * wall_thickness_px)
        door_by_swing = max_perp_px >= swing_thresh_px

        if door_by_swing:
            opening_type = "door"
        elif width_m >= cfg.long_opening_m:
            opening_type = "door"
        else:
            opening_type = "window"

        if opening_type == "window":
            height_m = cfg.default_window_height_m
            sill_m = cfg.default_window_sill_m
        else:
            height_m = cfg.default_door_height_m
            sill_m = 0.0

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
def emit_dsl(walls: List[Wall], openings: List[Opening]):
    lines = ['level("L1").elev(0)']
    walls.sort(key=lambda x: x.id)
    
    for w in walls:
        lines.append(f'wall("{w.id}").from({w.start[0]:.3f},{w.start[1]:.3f})'
                     f'.to({w.end[0]:.3f},{w.end[1]:.3f}).t({w.thickness:.2f}).h({w.height})')
        
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
    args = parser.parse_args()
    
    cfg = Config(meters_per_pixel=args.scale)
    if args.debug_dir: os.makedirs(args.debug_dir, exist_ok=True)
        
    mask_union, mask_wall, mask_open, dist_map, dims = load_and_preprocess(args.input, args.debug_dir, cfg)
    skel_graph = build_skeleton_graph(mask_union, args.debug_dir, cfg)
    raw_vectors = graph_to_vectors(skel_graph, args.debug_dir, cfg)
    snapped_walls = gravity_snap(raw_vectors, args.debug_dir, cfg)
    final_topology = rebuild_topology(snapped_walls, cfg)
    
    # 1. Attributes (Raw)
    walls_raw = process_attributes(final_topology, dist_map, dims[0], cfg)
    # 2. Consolidation (Merge + Junction Check)
    walls_merged = consolidate_walls(walls_raw, cfg)
    # 3. Openings (Map to merged)
    openings = extract_openings(mask_open, mask_wall, walls_merged, cfg)
    
    with open(args.out, "w") as f:
        f.write(emit_dsl(walls_merged, openings))
    print(f"Done! Written to {args.out}")

    if args.debug_dir:
        orig = cv2.imread(args.input)
        h, w_img = dims
        for w in walls_merged:
            p1 = (int(w.start[0] / cfg.meters_per_pixel), int(h - w.start[1] / cfg.meters_per_pixel))
            p2 = (int(w.end[0] / cfg.meters_per_pixel), int(h - w.end[1] / cfg.meters_per_pixel))
            cv2.line(orig, p1, p2, (0, 255, 0), 2)
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
