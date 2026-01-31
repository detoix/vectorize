import argparse
import cgi
import json
import os
import subprocess
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from pipeline_lib import PipelineError, run_vectorize_pipeline
from extract_scale import (
    derive_isotropic_meters_per_pixel,
    load_real_dims_from_payload,
    wall_bbox_from_mask,
    wall_mask_from_image_rgb,
)

import numpy as np
import cv2
import diff_utils



def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    content_length = handler.headers.get("Content-Length")
    if content_length is None:
        raise ValueError("Missing Content-Length header")
    try:
        length = int(content_length)
    except ValueError as e:
        raise ValueError("Invalid Content-Length header") from e
    body = handler.rfile.read(length)
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON body") from e


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... (truncated, {len(s) - limit} chars omitted)"

def _read_text_file(path: str, *, max_chars: Optional[int] = None) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = f.read()
    if max_chars is not None:
        return _truncate(data, max_chars)
    return data


class VectorizeAPIHandler(BaseHTTPRequestHandler):
    server_version = "VectorizeAPI/0.1"

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, status: int, text: str, *, content_type: str = "text/plain; charset=utf-8") -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        repo_root = getattr(self.server, "repo_root", os.getcwd())
        raw_path = (self.path or "").split("?", 1)[0]
        path = raw_path.rstrip("/") or "/"
        content_type = self.headers.get("Content-Type", "")

        if path == "/run":
            if content_type.startswith("multipart/form-data"):
                self._handle_run_multipart(repo_root, content_type)
                return
            self._handle_run_json(repo_root)
            return

        if path == "/":
            if content_type.startswith("multipart/form-data"):
                self._handle_mask2dsl_multipart(repo_root, content_type)
                return
            self._handle_mask2dsl_json(repo_root)
            return

        if path == "/furniture":
            if content_type.startswith("multipart/form-data"):
                self._handle_furniture_multipart(repo_root, content_type)
                return
            self._send_json(400, {"ok": False, "error": "Only multipart/form-data is supported for /furniture"})
            return

        if path == "/api/diff":
            if content_type.startswith("multipart/form-data"):
                self._handle_diff_vectorize_multipart(repo_root, content_type)
                return
            self._send_json(400, {"ok": False, "error": "Only multipart/form-data is supported for /api/diff"})
            return

        self._send_json(404, {"ok": False, "error": "Not found"})

    def _handle_run_json(self, repo_root: str) -> None:
        try:
            body = _read_json_body(self)
        except ValueError as e:
            self._send_json(400, {"ok": False, "error": str(e)})
            return

        image = body.get("image")
        if not image or not isinstance(image, str):
            self._send_json(400, {"ok": False, "error": "Field 'image' (string) is required"})
            return

        prompt = body.get("prompt")
        prompt_file = body.get("prompt_file") or body.get("promptFile")
        mask = body.get("mask", "mask.png")
        output = body.get("output", "output.dsl")
        debug_dir = body.get("debug_dir") or body.get("debugDir") or "debug_pipeline"
        skip_gen = bool(body.get("skip_gen") or body.get("skipGen") or False)
        api_key = body.get("api_key") or body.get("apiKey")
        response_format = (body.get("format") or body.get("responseFormat") or "text").lower()
        max_dsl_chars = int(body.get("max_dsl_chars") or body.get("maxDslChars") or 200000)

        try:
            run_vectorize_pipeline(
                image=image,
                prompt=prompt,
                prompt_file=prompt_file,
                mask=mask,
                output=output,
                skip_gen=skip_gen,
                debug_dir=debug_dir,
                api_key=api_key,
                cwd=repo_root,
            )
        except PipelineError as e:
            self._send_text(500, str(e) + "\n")
            return
        except Exception as e:
            self._send_text(500, f"Unexpected error: {e.__class__.__name__}: {e}\n")
            return

        dsl_path = os.path.join(repo_root, output) if not os.path.isabs(output) else output
        if not os.path.exists(dsl_path):
            self._send_text(500, "Pipeline completed but output DSL was not found.\n")
            return

        dsl = _read_text_file(dsl_path, max_chars=max_dsl_chars)
        if response_format == "json":
            self._send_json(200, {"dsl": dsl})
        else:
            self._send_text(200, dsl)

    def _handle_run_multipart(self, repo_root: str, content_type: str) -> None:
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
            )
        except Exception as e:
            self._send_json(400, {"ok": False, "error": f"Invalid multipart form: {e}"})
            return

        image_field = form["image"] if "image" in form else None
        if image_field is None:
            self._send_json(400, {"ok": False, "error": "Missing file field 'image'"})
            return
        if isinstance(image_field, list):
            image_field = image_field[0] if image_field else None
        if image_field is None or getattr(image_field, "file", None) is None:
            self._send_json(400, {"ok": False, "error": "Missing file field 'image'"})
            return

        api_key = form.getfirst("apiKey") or form.getfirst("api_key")
        prompt = form.getfirst("prompt")
        skip_gen = (form.getfirst("skipGen") or form.getfirst("skip_gen") or "").lower() in {"1", "true", "yes", "on"}
        response_format = (form.getfirst("format") or form.getfirst("responseFormat") or "text").lower()
        max_dsl_chars = int(form.getfirst("maxDslChars") or form.getfirst("max_dsl_chars") or 200000)

        request_dir_root = os.path.join(repo_root, ".api_tmp")
        os.makedirs(request_dir_root, exist_ok=True)
        request_id = uuid.uuid4().hex
        request_dir = os.path.join(request_dir_root, request_id)
        os.makedirs(request_dir, exist_ok=True)

        image_filename = os.path.basename(getattr(image_field, "filename", "") or "image.png")
        image_abspath = os.path.join(request_dir, image_filename)
        with open(image_abspath, "wb") as out:
            out.write(image_field.file.read())

        mask_name = "mask.png"
        mask_field = form["mask"] if "mask" in form else None
        if isinstance(mask_field, list):
            mask_field = mask_field[0] if mask_field else None
        if mask_field is not None and getattr(mask_field, "file", None) is not None:
            mask_filename = os.path.basename(getattr(mask_field, "filename", "") or "mask.png")
            mask_name = mask_filename
            mask_abspath = os.path.join(request_dir, mask_filename)
            with open(mask_abspath, "wb") as out:
                out.write(mask_field.file.read())

        image = os.path.join(".api_tmp", request_id, image_filename)
        mask = os.path.join(".api_tmp", request_id, mask_name)
        debug_dir = os.path.join(".api_tmp", request_id, "debug")
        output = os.path.join(".api_tmp", request_id, "output.dsl")

        try:
            run_vectorize_pipeline(
                image=image,
                prompt=prompt,
                prompt_file=None,
                mask=mask,
                output=output,
                skip_gen=skip_gen,
                debug_dir=debug_dir,
                api_key=api_key,
                cwd=repo_root,
            )
        except PipelineError as e:
            self._send_text(500, str(e) + "\n")
            return
        except Exception as e:
            self._send_text(500, f"Unexpected error: {e.__class__.__name__}: {e}\n")
            return

        dsl_path = os.path.join(repo_root, output)
        if not os.path.exists(dsl_path):
            self._send_text(500, "Pipeline completed but output DSL was not found.\n")
            return

        dsl = _read_text_file(dsl_path, max_chars=max_dsl_chars)
        if response_format == "json":
            self._send_json(200, {"dsl": dsl})
        else:
            self._send_text(200, dsl)

    def _handle_mask2dsl_json(self, repo_root: str) -> None:
        try:
            body = _read_json_body(self)
        except ValueError as e:
            self._send_json(400, {"ok": False, "error": str(e)})
            return

        dims = body.get("dims") or body.get("real_dims") or body.get("realDims")
        if not isinstance(dims, dict):
            self._send_json(400, {"ok": False, "error": "Field 'dims' (object) is required"})
            return

        mask_b64 = body.get("mask_b64") or body.get("image_b64")
        if not mask_b64 or not isinstance(mask_b64, str):
            self._send_json(400, {"ok": False, "error": "Field 'mask_b64' (base64 string) is required"})
            return

        response_format = (body.get("format") or body.get("responseFormat") or "text").lower()
        max_dsl_chars = int(body.get("max_dsl_chars") or body.get("maxDslChars") or 200000)

        import base64
        import numpy as np
        import cv2

        try:
            mask_bytes = base64.b64decode(mask_b64, validate=True)
        except Exception:
            self._send_json(400, {"ok": False, "error": "Invalid base64 in 'mask_b64'"})
            return

        img_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            self._send_json(400, {"ok": False, "error": "Could not decode mask image bytes"})
            return

        try:
            real_width_m, real_height_m = load_real_dims_from_payload(dims)
        except ValueError as e:
            self._send_json(400, {"ok": False, "error": str(e)})
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask_wall = wall_mask_from_image_rgb(img_rgb)
        try:
            wall_bbox_px = wall_bbox_from_mask(mask_wall)
        except ValueError as e:
            self._send_json(400, {"ok": False, "error": str(e)})
            return

        mpp, _mpp_x, _mpp_y, _px_w, _px_h = derive_isotropic_meters_per_pixel(
            wall_bbox_px=wall_bbox_px,
            real_width_m=real_width_m,
            real_height_m=real_height_m,
        )

        request_dir_root = os.path.join(repo_root, ".api_tmp")
        os.makedirs(request_dir_root, exist_ok=True)
        request_id = uuid.uuid4().hex
        request_dir = os.path.join(request_dir_root, request_id)
        os.makedirs(request_dir, exist_ok=True)

        mask_path = os.path.join(request_dir, "mask.png")
        with open(mask_path, "wb") as f:
            f.write(mask_bytes)

        out_path = os.path.join(request_dir, "output.dsl")
        debug_dir = os.path.join(request_dir, "debug")
        cmd = [
            sys.executable,
            os.path.join(repo_root, "mask2dsl.py"),
            "--input",
            os.path.join(".api_tmp", request_id, "mask.png"),
            "--scale",
            str(mpp),
            "--out",
            os.path.join(".api_tmp", request_id, "output.dsl"),
            "--debug-dir",
            os.path.join(".api_tmp", request_id, "debug"),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, check=False)
        if completed.returncode != 0:
            self._send_json(
                500,
                {
                    "ok": False,
                    "error": "mask2dsl failed",
                    "stdout": _truncate((completed.stdout or "").strip(), 8000),
                    "stderr": _truncate((completed.stderr or "").strip(), 8000),
                },
            )
            return

        if not os.path.exists(out_path):
            self._send_json(500, {"ok": False, "error": "mask2dsl completed but output DSL was not found"})
            return

        dsl = _read_text_file(out_path, max_chars=max_dsl_chars)
        if response_format == "json":
            self._send_json(200, {"ok": True, "meters_per_pixel": mpp, "dsl": dsl})
        else:
            self._send_text(200, dsl)

    def _handle_mask2dsl_multipart(self, repo_root: str, content_type: str) -> None:
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
            )
        except Exception as e:
            self._send_json(400, {"ok": False, "error": f"Invalid multipart form: {e}"})
            return

        mask_field = form["mask"] if "mask" in form else (form["image"] if "image" in form else None)
        if mask_field is None:
            self._send_json(400, {"ok": False, "error": "Missing file field 'mask' (or 'image')"})
            return
        if isinstance(mask_field, list):
            mask_field = mask_field[0] if mask_field else None
        if mask_field is None or getattr(mask_field, "file", None) is None:
            self._send_json(400, {"ok": False, "error": "Missing file field 'mask' (or 'image')"})
            return

        dims_payload: Optional[Dict[str, Any]] = None
        dims_field = form["dims"] if "dims" in form else None
        if isinstance(dims_field, list):
            dims_field = dims_field[0] if dims_field else None

        # cgi.FieldStorage sets `.file` even for normal text fields; use `.filename` to detect uploads.
        if dims_field is not None and getattr(dims_field, "filename", None):
            try:
                dims_payload = json.loads(dims_field.file.read().decode("utf-8"))
            except Exception:
                self._send_json(400, {"ok": False, "error": "Invalid JSON in uploaded 'dims' file"})
                return
        else:
            dims_text = None
            if dims_field is not None and hasattr(dims_field, "value"):
                dims_text = dims_field.value
            if not dims_text:
                dims_text = form.getfirst("dims") or form.getfirst("realDims") or form.getfirst("real_dims")
            if dims_text:
                try:
                    dims_payload = json.loads(dims_text)
                except Exception:
                    self._send_json(400, {"ok": False, "error": "Invalid JSON in 'dims' field"})
                    return

        if dims_payload is None or not isinstance(dims_payload, dict):
            self._send_json(400, {"ok": False, "error": "Missing 'dims' JSON (field or file)"})
            return

        response_format = (form.getfirst("format") or form.getfirst("responseFormat") or "text").lower()
        max_dsl_chars = int(form.getfirst("maxDslChars") or form.getfirst("max_dsl_chars") or 200000)

        mask_bytes = mask_field.file.read()
        if not mask_bytes:
            self._send_json(400, {"ok": False, "error": "Empty mask upload"})
            return

        import numpy as np
        import cv2

        img_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            self._send_json(400, {"ok": False, "error": "Could not decode mask image bytes"})
            return

        try:
            real_width_m, real_height_m = load_real_dims_from_payload(dims_payload)
        except ValueError as e:
            self._send_json(400, {"ok": False, "error": str(e)})
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask_wall = wall_mask_from_image_rgb(img_rgb)
        try:
            wall_bbox_px = wall_bbox_from_mask(mask_wall)
        except ValueError as e:
            self._send_json(400, {"ok": False, "error": str(e)})
            return

        mpp, _mpp_x, _mpp_y, _px_w, _px_h = derive_isotropic_meters_per_pixel(
            wall_bbox_px=wall_bbox_px,
            real_width_m=real_width_m,
            real_height_m=real_height_m,
        )

        request_dir_root = os.path.join(repo_root, ".api_tmp")
        os.makedirs(request_dir_root, exist_ok=True)
        request_id = uuid.uuid4().hex
        request_dir = os.path.join(request_dir_root, request_id)
        os.makedirs(request_dir, exist_ok=True)

        mask_path = os.path.join(request_dir, "mask.png")
        with open(mask_path, "wb") as f:
            f.write(mask_bytes)

        out_path = os.path.join(request_dir, "output.dsl")
        debug_dir = os.path.join(request_dir, "debug")
        cmd = [
            sys.executable,
            os.path.join(repo_root, "mask2dsl.py"),
            "--input",
            os.path.join(".api_tmp", request_id, "mask.png"),
            "--scale",
            str(mpp),
            "--out",
            os.path.join(".api_tmp", request_id, "output.dsl"),
            "--debug-dir",
            os.path.join(".api_tmp", request_id, "debug"),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, check=False)
        if completed.returncode != 0:
            self._send_json(
                500,
                {
                    "ok": False,
                    "error": "mask2dsl failed",
                    "stdout": _truncate((completed.stdout or "").strip(), 8000),
                    "stderr": _truncate((completed.stderr or "").strip(), 8000),
                },
            )
            return

        if not os.path.exists(out_path):
            self._send_json(500, {"ok": False, "error": "mask2dsl completed but output DSL was not found"})
            return

        dsl = _read_text_file(out_path, max_chars=max_dsl_chars)
        if response_format == "json":
            self._send_json(200, {"ok": True, "meters_per_pixel": mpp, "dsl": dsl})
        else:
            self._send_text(200, dsl)

    def _handle_furniture_multipart(self, repo_root: str, content_type: str) -> None:
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
            )
        except Exception as e:
            self._send_json(400, {"ok": False, "error": f"Invalid multipart form: {e}"})
            return

        image_field = form["image"] if "image" in form else None
        if isinstance(image_field, list):
            image_field = image_field[0] if image_field else None

        if image_field is None or getattr(image_field, "file", None) is None:
            self._send_json(400, {"ok": False, "error": "Missing file field 'image'"})
            return

        image_bytes = image_field.file.read()
        if not image_bytes:
            self._send_json(400, {"ok": False, "error": "Empty image upload"})
            return

        img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            self._send_json(400, {"ok": False, "error": "Could not decode image bytes"})
            return

        dims_payload: Optional[Dict[str, Any]] = None
        dims_field = form["dims"] if "dims" in form else None
        if isinstance(dims_field, list):
            dims_field = dims_field[0] if dims_field else None

        if dims_field is not None and getattr(dims_field, "filename", None):
            try:
                dims_payload = json.loads(dims_field.file.read().decode("utf-8"))
            except Exception:
                self._send_json(400, {"ok": False, "error": "Invalid JSON in uploaded 'dims' file"})
                return
        else:
            dims_text = None
            if dims_field is not None and hasattr(dims_field, "value"):
                dims_text = dims_field.value
            if not dims_text:
                dims_text = form.getfirst("dims") or form.getfirst("realDims") or form.getfirst("real_dims")
            if dims_text:
                try:
                    dims_payload = json.loads(dims_text)
                except Exception:
                    self._send_json(400, {"ok": False, "error": "Invalid JSON in 'dims' field"})
                    return

        # Try to get discrete params if dims payload is missing
        if not dims_payload:
            real_x = form.getfirst("real-x") or form.getfirst("realX")
            real_y = form.getfirst("real-y") or form.getfirst("realY")
            
            # If we have at least one dimension
            if real_x or real_y:
                dims_payload = {}
                if real_x:
                    unit = form.getfirst("real-x-unit") or form.getfirst("realXUnit") or "m"
                    try:
                        val = float(real_x)
                        dims_payload["real-x"] = {"value": val, "unit": unit}
                    except ValueError:
                         self._send_json(400, {"ok": False, "error": f"Invalid number for real-x: {real_x}"})
                         return

                if real_y:
                    unit = form.getfirst("real-y-unit") or form.getfirst("realYUnit") or "m"
                    try:
                        val = float(real_y)
                        dims_payload["real-y"] = {"value": val, "unit": unit}
                    except ValueError:
                         self._send_json(400, {"ok": False, "error": f"Invalid number for real-y: {real_y}"})
                         return

        meters_per_pixel = 1.0
        using_real_units = False

        # Attempt to determine scale if dims are provided
        if dims_payload:
            try:
                real_width_m, real_height_m = load_real_dims_from_payload(dims_payload)
                
                # We need to compute meters_per_pixel. 
                # This usually expects walls to be present to determine the bbox.
                # If this image is just furniture blobs, this might fail or be inaccurate 
                # if we rely on wall detection.
                # However, for now, we will try to detect walls (white pixels) 
                # OR fallback to image dimensions if no walls found?
                # Actually, derive_isotropic_meters_per_pixel uses wall_bbox_px.
                
                # Let's try to detect walls first, consistent with mask2dsl.
                # Note: This assumes the furniture image HAS wall pixels (white/gray).
                # If it's a transparency layer or black background, this will fail.
                # But typically the input to this pipeline is the masked floorplan.
                
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mask_wall = wall_mask_from_image_rgb(img_rgb)
                try:
                    wall_bbox_px = wall_bbox_from_mask(mask_wall)
                    mpp, _, _, _, _ = derive_isotropic_meters_per_pixel(
                        wall_bbox_px=wall_bbox_px,
                        real_width_m=real_width_m,
                        real_height_m=real_height_m,
                    )
                    meters_per_pixel = mpp
                    using_real_units = True
                except ValueError:
                    # Fallback: if no walls found, assume dims apply to the entire image 
                    # (less likely but safer fallback than crashing)
                    h, w = img_bgr.shape[:2]
                    # We can't really do isotropic easily without a target bbox, 
                    # so let's just use width if available, else height
                    if real_width_m:
                        meters_per_pixel = real_width_m / w
                        using_real_units = True
                    elif real_height_m:
                        meters_per_pixel = real_height_m / h
                        using_real_units = True
            except ValueError as e:
                 self._send_json(400, {"ok": False, "error": str(e)})
                 return

        # Furniture processing logic
        furniture_items = []
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Color definitions (HSV ranges)
        colors = {
            "table": [
                (np.array([35, 50, 50]), np.array([80, 255, 255]))  # Green (Broad)
            ],
            "chair": [
                (np.array([80, 50, 50]), np.array([100, 255, 255])) # Cyan (Broad)
            ],
            "bed": [
                (np.array([0, 50, 50]), np.array([15, 255, 255])),   # Red (lower)
                (np.array([165, 50, 50]), np.array([180, 255, 255])) # Red (upper)
            ],
            "cabinet": [
                (np.array([140, 50, 50]), np.array([170, 255, 255])) # Purple (Broad)
            ]
        }
        
        for f_type, ranges in colors.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
            
            # Morphological operations to clean up mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 50: # Minimum area filter (pixels)
                    continue
                
                rect = cv2.minAreaRect(cnt)
                (center_x, center_y), (w, h), angle = rect
                
                # Normalize angle/dimensions 
                length = max(w, h)
                width = min(w, h)
                
                # Adjust rotation
                final_rotation = angle
                if w < h:
                     final_rotation = angle + 90
                
                furniture_items.append({
                    "type": f_type,
                    "x": center_x * meters_per_pixel,
                    "y": center_y * meters_per_pixel,
                    "length": length * meters_per_pixel,
                    "width": width * meters_per_pixel,
                    "rotation": final_rotation
                })

        self._send_json(200, {
            "furniture": furniture_items,
            "meters_per_pixel": meters_per_pixel if using_real_units else None,
            "units": "meters" if using_real_units else "pixels"
        })

    def _handle_diff_vectorize_multipart(self, repo_root: str, content_type: str) -> None:
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
            )
        except Exception as e:
            self._send_json(400, {"ok": False, "error": f"Invalid multipart form: {e}"})
            return

        # 1. Get Images (Original, Target)
        orig_field = form["original"] if "original" in form else None
        target_field = form["target"] if "target" in form else None
        
        if not orig_field or not getattr(orig_field, "file", None):
             self._send_json(400, {"ok": False, "error": "Missing 'original' image file"})
             return
        if not target_field or not getattr(target_field, "file", None):
             self._send_json(400, {"ok": False, "error": "Missing 'target' image file"})
             return

        # 2. Setup Temp Dir
        request_dir_root = os.path.join(repo_root, ".api_tmp")
        os.makedirs(request_dir_root, exist_ok=True)
        request_id = uuid.uuid4().hex
        request_dir = os.path.join(request_dir_root, request_id)
        os.makedirs(request_dir, exist_ok=True)

        orig_path = os.path.join(request_dir, "original.png")
        target_path = os.path.join(request_dir, "target.png")
        
        with open(orig_path, "wb") as f: f.write(orig_field.file.read())
        with open(target_path, "wb") as f: f.write(target_field.file.read())

        # 3. Load Images & Compute Diff Mask
        orig_img = cv2.imread(orig_path)
        target_img = cv2.imread(target_path)
        
        if orig_img is None or target_img is None:
             self._send_json(400, {"ok": False, "error": "Failed to decode images"})
             return

        added_mask, removed_mask = diff_utils.compute_change_mask(orig_img, target_img)
        cv2.imwrite(os.path.join(request_dir, "change_mask.png"), added_mask) # Save main one for debug
        cv2.imwrite(os.path.join(request_dir, "removed_mask.png"), removed_mask)

        # 4. Determine Scale (Meters Per Pixel)
        # Try to use 'dims' JSON if provided, otherwise default or error?
        # Similar logic to _handle_mask2dsl_multipart
        dims_payload = None
        dims_field = form["dims"] if "dims" in form else None
        if dims_field and getattr(dims_field, "file", None): # File upload
             dims_payload = json.loads(dims_field.file.read().decode("utf-8"))
        elif form.getfirst("dims"): # Text field
             dims_payload = json.loads(form.getfirst("dims"))
        
        meters_per_pixel = 0.02 # Default fallback
        
        if dims_payload:
            try:
                rw, rh = load_real_dims_from_payload(dims_payload)
                # Compute mpp using Target Image walls
                img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                mask_wall = wall_mask_from_image_rgb(img_rgb)
                wall_bbox_px = wall_bbox_from_mask(mask_wall)
                meters_per_pixel, _, _, _, _ = derive_isotropic_meters_per_pixel(
                    wall_bbox_px, rw, rh
                )
            except Exception as e:
                # If cannot determine, fallback or error?
                pass
        else:
             # Legacy/Fallback scale param
             s_val = form.getfirst("scale")
             if s_val: meters_per_pixel = float(s_val)

        # 5. Run Vectorization on TARGET
        out_path = os.path.join(request_dir, "output.dsl")
        cmd = [
            sys.executable,
            os.path.join(repo_root, "mask2dsl.py"),
            "--input",
            target_path, # We vectorize the Target
            "--scale",
            str(meters_per_pixel),
            "--out",
            out_path,
            "--format", "json", # Force JSON for parsing
            "--debug-dir",
            os.path.join(request_dir, "debug"),
        ]
        
        completed = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, check=False)
        if completed.returncode != 0:
             self._send_json(500, {"ok": False, "error": "Vectorization failed", "details": completed.stderr[:500]})
             return
             
        # 6. Load DSL/JSON and Filter
        with open(out_path, "r") as f:
            full_dsl_json = json.load(f)
            
        filtered_dsl = diff_utils.filter_dsl(full_dsl_json, added_mask, removed_mask, meters_per_pixel)
        
        # 7. Return Result
        self._send_json(200, {
            "ok": True,
            "result": filtered_dsl, # The filtered entities
            "debug_id": request_id
        })



def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Vectorize pipeline over HTTP (stdlib server).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing generate_image.py, mask2dsl.py, etc.",
    )
    args = parser.parse_args(argv)

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), VectorizeAPIHandler)
    except PermissionError as e:
        print(
            f"Error: could not bind to {args.host}:{args.port} (PermissionError: {e}).\n"
            "If you're running inside a restricted sandbox/container, start this server on your host machine instead.",
        )
        return 1
    httpd.repo_root = args.repo_root
    print(f"Listening on http://{args.host}:{args.port} (repo_root={args.repo_root})")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
