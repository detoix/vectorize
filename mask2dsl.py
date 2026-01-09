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
    min_spur_length_px: int = 30   # Prune skeletal dead ends
    opening_bridge_px: int = 3     # Dilation size for Blue Bridge
    epsilon_rdp: float = 3.0       # RDP simplification epsilon

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

# --- PHASE 0: IO & MASKS ---
def create_robust_union_mask(mask_wall, mask_open, debug_dir):
    """
    Bridges wall gaps ONLY where a blue opening exists.
    """
    union_mask = mask_wall.copy()
    debug_img = cv2.cvtColor(mask_wall, cv2.COLOR_GRAY2BGR)
    
    contours, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 10: continue
        
        single_door_mask = np.zeros_like(mask_wall)
        cv2.drawContours(single_door_mask, [cnt], -1, 255, -1)
        search_zone = cv2.dilate(single_door_mask, np.ones((10,10), np.uint8))
        nearby_walls = cv2.bitwise_and(mask_wall, search_zone)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(nearby_walls)
        
        if num_labels >= 3:
            sorted_indices = np.argsort(stats[1:, 4])[::-1] + 1 
            jamb_a_idx = sorted_indices[0]
            jamb_b_idx = sorted_indices[1]
            
            pt_a = (int(centroids[jamb_a_idx][0]), int(centroids[jamb_a_idx][1]))
            pt_b = (int(centroids[jamb_b_idx][0]), int(centroids[jamb_b_idx][1]))
            
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
    mask_open = cv2.inRange(img_rgb, np.array([0, 0, 200]), np.array([50, 50, 255]))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.opening_bridge_px, cfg.opening_bridge_px))
    mask_open_dilated = cv2.dilate(mask_open, kernel, iterations=1)
    
    mask_union = create_robust_union_mask(mask_wall, mask_open_dilated, debug_dir)
    mask_union = cv2.morphologyEx(mask_union, cv2.MORPH_CLOSE, kernel)

    # Compute Distance Transform on SOLID WALLS only (not the bridge)
    dist_map = cv2.distanceTransform(mask_wall, cv2.DIST_L2, 5)

    if debug_dir:
        cv2.imwrite(f"{debug_dir}/00_union_mask.png", mask_union)
        cv2.imwrite(f"{debug_dir}/00_dist_map.png", (dist_map/dist_map.max()*255).astype(np.uint8))

    return mask_union, mask_open, dist_map, img_rgb.shape[:2]

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

            while True:
                nbrs = list(G.neighbors(curr))
                next_candidates = [n for n in nbrs if n != prev]
                if not next_candidates:
                    break

                nxt = next_candidates[0]
                length += float(G[curr][nxt].get("weight", 1.0))
                path.append(nxt)
                prev, curr = curr, nxt

                if G.degree(curr) != 2 or length >= cfg.min_spur_length_px:
                    break

            if length >= cfg.min_spur_length_px:
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
            
            # Thickness Heuristics
            mergable_thick = same_thick
            if not mergable_thick and is_simple_joint:
                is_curr_def = abs(current.thickness - 0.2) < 0.01
                is_next_def = abs(next_w.thickness - 0.2) < 0.01
                
                if is_curr_def and not is_next_def and next_w.thickness > 0.2:
                    current.thickness = next_w.thickness # Inherit real thickness
                    mergable_thick = True
                elif is_next_def and not is_curr_def and current.thickness > 0.2:
                    mergable_thick = True # Keep current thickness

            if gap < 0.2 and mergable_thick and is_simple_joint:
                # Merge
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
        
    for i, w in enumerate(final_walls): w.id = f"w_{i:03d}"
    return final_walls

# --- PHASE 6: OPENINGS ---
def extract_openings(mask_open, walls: List[Wall], cfg):
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
        
        # Project on World Line
        w_line = LineString([best_wall.start, best_wall.end])
        p_world = Point(cx * cfg.meters_per_pixel, (mask_open.shape[0] - cy) * cfg.meters_per_pixel)
        at_m = w_line.project(p_world)
        
        rect = cv2.minAreaRect(cnt)
        width_m = max(rect[1]) * cfg.meters_per_pixel
        
        openings.append(Opening(
            id=f"o_{i:03d}", type="door",
            wall_id=best_wall.id, at=round(at_m, 3), width=round(width_m, 3)
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
        lines.append(f'{o.type}("{o.id}").in("{o.wall_id}").at({o.at:.3f})'
                     f'.w({o.width:.3f}).h({o.height})')
    return "\n".join(lines)

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--debug-dir")
    parser.add_argument("--scale", type=float, default=0.02)
    args = parser.parse_args()
    
    cfg = Config(meters_per_pixel=args.scale)
    if args.debug_dir: os.makedirs(args.debug_dir, exist_ok=True)
        
    mask_union, mask_open, dist_map, dims = load_and_preprocess(args.input, args.debug_dir, cfg)
    skel_graph = build_skeleton_graph(mask_union, args.debug_dir, cfg)
    raw_vectors = graph_to_vectors(skel_graph, args.debug_dir, cfg)
    snapped_walls = gravity_snap(raw_vectors, args.debug_dir, cfg)
    final_topology = rebuild_topology(snapped_walls, cfg)
    
    # 1. Attributes (Raw)
    walls_raw = process_attributes(final_topology, dist_map, dims[0], cfg)
    # 2. Consolidation (Merge + Junction Check)
    walls_merged = consolidate_walls(walls_raw, cfg)
    # 3. Openings (Map to merged)
    openings = extract_openings(mask_open, walls_merged, cfg)
    
    with open(args.out, "w") as f:
        f.write(emit_dsl(walls_merged, openings))
    print(f"Done! Written to {args.out}")

    if args.debug_dir:
        orig = cv2.imread(args.input)
        h, w_img = dims
        for w in walls_merged:
            p1 = (int(w.start[0]/cfg.meters_per_pixel), int(h - w.start[1]/cfg.meters_per_pixel))
            p2 = (int(w.end[0]/cfg.meters_per_pixel), int(h - w.end[1]/cfg.meters_per_pixel))
            cv2.line(orig, p1, p2, (0, 255, 0), 2)
            cv2.putText(orig, w.id, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.imwrite(f"{args.debug_dir}/05_final_overlay.png", orig)

if __name__ == "__main__":
    main()
