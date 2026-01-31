import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler
from typing import Dict, Optional, Tuple





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


def _send_text(handler: BaseHTTPRequestHandler, status: int, text: str) -> None:
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    handler.end_headers()
    handler.wfile.write(data)


def _parse_multipart(body: bytes, boundary: bytes) -> Tuple[Dict[str, bytes], Dict[str, str]]:
    """
    Minimal multipart/form-data parser.
    Returns (files_by_field, text_fields).

    Expects each part to include Content-Disposition with name=...
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

        # Trim trailing CRLF from content (typical multipart formatting).
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

        mask_bytes: Optional[bytes] = None
        scale: Optional[float] = None
        real_x: Optional[Dict[str, object]] = None
        real_y: Optional[Dict[str, object]] = None
        response_format: str = "text"  # Default to text/dsl

        if content_type.startswith("application/json"):
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                return _send_json(self, 400, {"ok": False, "error": "Invalid JSON"})

            try:
                if "scale" in payload:
                    scale = float(payload.get("scale"))
            except (TypeError, ValueError):
                return _send_json(self, 400, {"ok": False, "error": "Invalid 'scale' (number)"})
            
            real_x = payload.get("real-x")
            real_y = payload.get("real-y")
            
            response_format = payload.get("format", "text")

            mask_b64 = payload.get("mask_b64")
            if not isinstance(mask_b64, str) or not mask_b64.strip():
                return _send_json(self, 400, {"ok": False, "error": "Missing 'mask_b64' (base64 image)"})

            try:
                mask_bytes = base64.b64decode(mask_b64, validate=True)
            except Exception:
                return _send_json(self, 400, {"ok": False, "error": "Invalid base64 in 'mask_b64'"})

        elif "multipart/form-data" in content_type:
            m = re.search(r"boundary=([^;]+)", content_type)
            if not m:
                return _send_json(self, 400, {"ok": False, "error": "Missing multipart boundary"})
            boundary = m.group(1).strip().strip('"').encode("utf-8")
            files, fields = _parse_multipart(body, boundary)

            if "mask" not in files:
                return _send_json(self, 400, {"ok": False, "error": "Missing multipart file field 'mask'"})
            mask_bytes = files["mask"]

            # Try to get scale (legacy)
            val = fields.get("scale", "").strip()
            if val:
                try:
                    scale = float(val)
                except ValueError:
                    return _send_json(self, 400, {"ok": False, "error": "Invalid multipart field 'scale'"})
            
            # Try to get real-x / real-y (expecting JSON strings in these fields)
            if "real-x" in fields:
                try:
                    real_x = json.loads(fields["real-x"])
                except json.JSONDecodeError:
                    return _send_json(self, 400, {"ok": False, "error": "Invalid JSON in field 'real-x'"})
            if "real-y" in fields:
                try:
                    real_y = json.loads(fields["real-y"])
                except json.JSONDecodeError:
                     return _send_json(self, 400, {"ok": False, "error": "Invalid JSON in field 'real-y'"})

            response_format = fields.get("format", "text")
        else:
            return _send_json(
                self,
                415,
                {"ok": False, "error": "Unsupported Content-Type (use application/json or multipart/form-data)"},
            )

        # Validation moved to command construction to allow mutually exclusive logic
        if mask_bytes is None or len(mask_bytes) < 8:
            return _send_json(self, 400, {"ok": False, "error": "Missing/invalid mask image bytes"})

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        mask2dsl_path = os.path.join(repo_root, "mask2dsl.py")
        if not os.path.exists(mask2dsl_path):
            return _send_json(self, 500, {"ok": False, "error": "Server misconfigured: mask2dsl.py not found"})

        try:
            with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
                mask_path = os.path.join(tmpdir, "mask.png")
                out_path = os.path.join(tmpdir, "out.dsl")

                with open(mask_path, "wb") as f:
                    f.write(mask_bytes)

                cmd = [
                    sys.executable,
                    mask2dsl_path,
                    "--input",
                    mask_path,
                    "--out",
                    out_path,
                    "--debug-dir",
                    os.path.join(tmpdir, "debug"),
                ]
                
                # Logic: prefer real-json if available, else scale.
                dims_payload = {}
                if real_x is not None:
                     dims_payload["real-x"] = real_x
                if real_y is not None:
                     dims_payload["real-y"] = real_y
                
                if dims_payload:
                    dims_json_path = os.path.join(tmpdir, "dims.json")
                    with open(dims_json_path, "w") as f:
                        json.dump(dims_payload, f)
                    cmd.extend(["--real-json", dims_json_path])
                    
                    # Pass a dummy scale if script insists on having one or logic requires it?
                    # mask2dsl default is 0.02, but if we pass --real-json it overrides logic.
                    # We might still want to pass a dummy scale to keep argparse happy if it was required?
                    # It's not required in argparse (default=0.02).
                else:
                    if scale is None or not (scale > 0):
                        return _send_json(self, 400, {"ok": False, "error": "Missing scale or real dimensions"})
                    cmd.extend(["--scale", str(scale)])
                
                if response_format == "json":
                    cmd.extend(["--format", "json"])

                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=repo_root,
                    check=False,
                )
                if completed.returncode != 0:
                    return _send_json(
                        self,
                        500,
                        {
                            "ok": False,
                            "error": "mask2dsl failed",
                            "stderr": (completed.stderr or "").strip()[:8000],
                            "stdout": (completed.stdout or "").strip()[:8000],
                        },
                    )

                with open(out_path, "r", encoding="utf-8") as f:
                    dsl = f.read()

        except Exception as e:
            return _send_json(self, 500, {"ok": False, "error": f"Server error: {e.__class__.__name__}: {e}"})

        if response_format == "json":
             # mask2dsl.py already wrote a JSON string to `dsl`
            try:
                # We interpret it as JSON to ensure validity before sending? 
                # Or just send it as a raw string if we want to avoid double-parsing cost?
                # Better to parse and send as object so HTTP Content-Type application/json is correct
                # and client receives an actual object, not a stringified JSON.
                json_output = json.loads(dsl)
                return _send_json(self, 200, {"ok": True, "result": json_output})
            except json.JSONDecodeError:
                return _send_json(self, 500, {"ok": False, "error": "Internal error: mask2dsl produced invalid JSON"})
        
        return _send_text(self, 200, dsl)

