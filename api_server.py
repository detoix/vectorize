import argparse
import cgi
import json
import os
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from pipeline_lib import PipelineError, run_vectorize_pipeline


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
        if self.path.rstrip("/") != "/run":
            self._send_json(404, {"ok": False, "error": "Not found"})
            return

        repo_root = getattr(self.server, "repo_root", os.getcwd())
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("multipart/form-data"):
            self._handle_run_multipart(repo_root, content_type)
            return

        self._handle_run_json(repo_root)

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
