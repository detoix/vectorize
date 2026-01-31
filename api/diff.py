import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler
from typing import Dict, Optional, Tuple, Any

# Ensure local imports work
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    import diff_utils
    from extract_scale import (
        derive_isotropic_meters_per_pixel,
        load_real_dims_from_payload,
        wall_bbox_from_mask,
        wall_mask_from_image_rgb,
    )
    import cv2
    import numpy as np
except ImportError:
    pass

def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, object]) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    handler.end_headers()
    handler.wfile.write(data)

def _parse_multipart(body: bytes, boundary: bytes) -> Tuple[Dict[str, bytes], Dict[str, str]]:
    """
    Minimal multipart/form-data parser (Copied from api/vectorize.py for independence).
    """
    delimiter = b"--" + boundary
    parts = body.split(delimiter)
    files: Dict[str, bytes] = {}
    fields: Dict[str, str] = {}

    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue

        header_blob, sep, content = part.partition(b"\r\n\r\n")
        if sep == b"":
            continue

        headers = header_blob.decode("utf-8", errors="replace").split("\r\n")
        disp = ""
        ctype = ""
        for h in headers:
            if h.lower().startswith("content-disposition:"):
                disp = h.split(":", 1)[1].strip()
            if h.lower().startswith("content-type:"):
                ctype = h.split(":", 1)[1].strip()

        m = re.search(r'name="([^"]+)"', disp)
        if not m:
            continue
        field_name = m.group(1)

        # Trim trailing CRLF
        if content.endswith(b"\r\n"):
            content = content[:-2]

        has_filename = bool(re.search(r'filename="[^"]*"', disp))
        if has_filename or ctype:
            files[field_name] = content
        else:
            fields[field_name] = content.decode("utf-8", errors="replace")

    return files, fields

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("content-length") or "0")
        except ValueError:
            return _send_json(self, 400, {"ok": False, "error": "Invalid Content-Length"})

        if content_length <= 0:
            return _send_json(self, 400, {"ok": False, "error": "Empty request body"})

        body = self.rfile.read(content_length)
        content_type = (self.headers.get("content-type") or "").strip()

        if "multipart/form-data" not in content_type:
             return _send_json(self, 400, {"ok": False, "error": "Content-Type must be multipart/form-data"})

        m = re.search(r"boundary=([^;]+)", content_type)
        if not m:
            return _send_json(self, 400, {"ok": False, "error": "Missing multipart boundary"})
        boundary = m.group(1).strip().strip('"').encode("utf-8")
        
        try:
            files, fields = _parse_multipart(body, boundary)
        except Exception as e:
            return _send_json(self, 400, {"ok": False, "error": f"Failed to parse multipart: {e}"})

        # 1. Get Images
        if "original" not in files:
            return _send_json(self, 400, {"ok": False, "error": "Missing 'original' file"})
        if "target" not in files:
            return _send_json(self, 400, {"ok": False, "error": "Missing 'target' file"})

        orig_bytes = files["original"]
        targ_bytes = files["target"]

        # 2. Setup Vercel / Temporary Environment
        # In Vercel, we can only write to /tmp
        tmpdir = tempfile.mkdtemp(dir="/tmp")
        request_id = uuid.uuid4().hex
        
        orig_path = os.path.join(tmpdir, "original.png")
        target_path = os.path.join(tmpdir, "target.png")
        
        with open(orig_path, "wb") as f: f.write(orig_bytes)
        with open(target_path, "wb") as f: f.write(targ_bytes)

        # 3. Compute Diff
        try:
            orig_arr = np.frombuffer(orig_bytes, dtype=np.uint8)
            targ_arr = np.frombuffer(targ_bytes, dtype=np.uint8)
            orig_img = cv2.imdecode(orig_arr, cv2.IMREAD_COLOR)
            target_img = cv2.imdecode(targ_arr, cv2.IMREAD_COLOR)
            
            if orig_img is None or target_img is None:
                return _send_json(self, 400, {"ok": False, "error": "Could not decode images"})

            added_mask, removed_mask = diff_utils.compute_change_mask(orig_img, target_img)
            
            # Optional debug saving (won't persist in Vercel, but good for local test)
            # cv2.imwrite(os.path.join(tmpdir, "debug_added.png"), added_mask)

        except Exception as e:
            return _send_json(self, 500, {"ok": False, "error": f"Diff computation failed: {e}"})

        # 4. Determine Scale
        meters_per_pixel = 0.02 # Default
        
        dims_json = None
        if "dims" in files:
            try:
                dims_json = json.loads(files["dims"].decode("utf-8"))
            except: pass
        elif "dims" in fields:
            try:
                dims_json = json.loads(fields["dims"])
            except: pass
            
        if dims_json:
            try:
                rw, rh = load_real_dims_from_payload(dims_json)
                img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                mask_wall = wall_mask_from_image_rgb(img_rgb)
                wall_bbox_px = wall_bbox_from_mask(mask_wall)
                mpp, _, _, _, _ = derive_isotropic_meters_per_pixel(
                    wall_bbox_px, rw, rh
                )
                meters_per_pixel = mpp
            except Exception:
                pass
        elif "scale" in fields:
             try:
                 meters_per_pixel = float(fields["scale"])
             except: pass

        # 5. Vectorize Target
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        out_path = os.path.join(tmpdir, "output.dsl")
        mask2dsl_path = os.path.join(repo_root, "mask2dsl.py")
        
        cmd = [
            sys.executable,
            mask2dsl_path,
            "--input", target_path,
            "--scale", str(meters_per_pixel),
            "--out", out_path,
            "--format", "json",
            "--debug-dir", os.path.join(tmpdir, "debug")
        ]
        
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, check=False)
            if completed.returncode != 0:
                 return _send_json(self, 500, {
                     "ok": False, 
                     "error": "Vectorization failed", 
                     "stderr": (completed.stderr or "")[:1000]
                 })
                 
            if not os.path.exists(out_path):
                 return _send_json(self, 500, {"ok": False, "error": "Output file not created"})
                 
            with open(out_path, "r") as f:
                full_dsl = json.load(f)
                
            filtered_dsl = diff_utils.filter_dsl(full_dsl, added_mask, removed_mask, meters_per_pixel)
            
            return _send_json(self, 200, {
                "ok": True,
                "result": filtered_dsl
            })
            
        except Exception as e:
            return _send_json(self, 500, {"ok": False, "error": f"Processing error: {e}"})
