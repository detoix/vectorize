import cgi
import json
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler
from typing import Dict, Any

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        content_type = self.headers.get("Content-Type", "")
        
        if not content_type.startswith("multipart/form-data"):
            self._send_json(400, {"ok": False, "error": "Only multipart/form-data is supported for this endpoint"})
            return

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
                if cv2.contourArea(cnt) < 50: # Minimum area filter
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
                    "x": center_x,
                    "y": center_y,
                    "length": length,
                    "width": width,
                    "rotation": final_rotation
                })

        self._send_json(200, {"furniture": furniture_items})
