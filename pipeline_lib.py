import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class CommandResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str


class PipelineError(RuntimeError):
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


def _redact_command(command: Sequence[str]) -> List[str]:
    redacted = [str(c) for c in command]
    for i, token in enumerate(redacted[:-1]):
        if token in {"--api-key", "--apikey", "--key"}:
            redacted[i + 1] = "***"
    return redacted


def run_command(
    command: Sequence[Optional[str]],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout_s: Optional[int] = None,
) -> CommandResult:
    cmd = [str(c) for c in command if c is not None]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=cwd,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        raise PipelineError(
            f"Command timed out after {timeout_s}s",
            details={"command": _redact_command(cmd), "stdout": e.stdout or "", "stderr": e.stderr or ""},
        ) from e

    result = CommandResult(
        command=_redact_command(cmd),
        returncode=completed.returncode,
        stdout=(completed.stdout or "").strip(),
        stderr=(completed.stderr or "").strip(),
    )
    if completed.returncode != 0:
        raise PipelineError(
            f"Command failed with exit code {completed.returncode}",
            details={"command": result.command, "stdout": result.stdout, "stderr": result.stderr},
        )
    return result


def run_vectorize_pipeline(
    *,
    image: str,
    prompt: Optional[str] = None,
    prompt_file: Optional[str] = None,
    mask: str = "mask.png",
    output: str = "output.dsl",
    skip_gen: bool = False,
    debug_dir: str = "debug_pipeline",
    api_key: Optional[str] = None,
    python_executable: Optional[str] = None,
    cwd: Optional[str] = None,
    parallelize_ai_steps: bool = True,
) -> Dict[str, Any]:
    python_executable = python_executable or sys.executable
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise PipelineError("API Key not found. Set GOOGLE_API_KEY or pass api_key.")

    env = os.environ.copy()
    env["GOOGLE_API_KEY"] = api_key

    if cwd:
        image_path = os.path.join(cwd, image) if not os.path.isabs(image) else image
        mask_path = os.path.join(cwd, mask) if not os.path.isabs(mask) else mask
        output_path = os.path.join(cwd, output) if not os.path.isabs(output) else output
        debug_dir_path = os.path.join(cwd, debug_dir) if not os.path.isabs(debug_dir) else debug_dir
    else:
        image_path, mask_path, output_path, debug_dir_path = image, mask, output, debug_dir

    if not os.path.exists(image_path):
        raise PipelineError(f"Input image not found: {image}")

    os.makedirs(debug_dir_path, exist_ok=True)

    if not skip_gen and not prompt and not prompt_file:
        default_prompt_path = os.path.join(cwd or ".", "mask_prompt.md")
        if os.path.exists(default_prompt_path):
            prompt_file = "mask_prompt.md"
        else:
            raise PipelineError("--prompt or --prompt-file is required (and mask_prompt.md not found).")

    logs: List[CommandResult] = []

    def _generate_mask() -> Optional[CommandResult]:
        if skip_gen:
            if not os.path.exists(mask_path):
                raise PipelineError(f"Mask file not found and generation skipped: {mask}")
            return None
        gen_cmd: List[Optional[str]] = [
            python_executable,
            "generate_image.py",
            "--output",
            mask,
            "--image",
            image,
            "--api-key",
            api_key,
        ]
        if prompt:
            gen_cmd.extend(["--prompt", prompt])
        if prompt_file:
            gen_cmd.extend(["--prompt-file", prompt_file])
        return run_command(gen_cmd, env=env, cwd=cwd)

    def _extract_dims() -> CommandResult:
        return run_command(
            [python_executable, "ai_extract_real_dims.py", "--image", image, "--api-key", api_key],
            env=env,
            cwd=cwd,
        )

    dims_json_path = os.path.join(debug_dir_path, "dims.json")
    if parallelize_ai_steps:
        with ThreadPoolExecutor(max_workers=2) as pool:
            gen_future = pool.submit(_generate_mask)
            dims_future = pool.submit(_extract_dims)
            gen_res = gen_future.result()
            dims = dims_future.result()
        if gen_res is not None:
            logs.append(gen_res)
        logs.append(dims)
    else:
        gen_res = _generate_mask()
        if gen_res is not None:
            logs.append(gen_res)
        dims = _extract_dims()
        logs.append(dims)

    try:
        json.loads(dims.stdout)
    except json.JSONDecodeError as e:
        raise PipelineError(
            "ai_extract_real_dims.py did not return valid JSON",
            details={"stdout": dims.stdout, "stderr": dims.stderr},
        ) from e
    with open(dims_json_path, "w", encoding="utf-8") as f:
        f.write(dims.stdout)

    scale_res = run_command(
        [
            python_executable,
            "extract_scale.py",
            "--input",
            image,
            "--real-json",
            dims_json_path,
            "--format",
            "plain",
        ],
        env=env,
        cwd=cwd,
    )
    logs.append(scale_res)
    try:
        scale = float(scale_res.stdout)
    except ValueError as e:
        raise PipelineError(f"Could not parse scale as float: {scale_res.stdout}") from e

    vec_res = run_command(
        [
            python_executable,
            "mask2dsl.py",
            "--input",
            mask,
            "--scale",
            str(scale),
            "--out",
            output,
            "--debug-dir",
            debug_dir,
        ],
        env=env,
        cwd=cwd,
    )
    logs.append(vec_res)

    return {
        "ok": True,
        "image": image,
        "mask": mask,
        "output": output,
        "debug_dir": debug_dir,
        "scale_m_per_px": scale,
        "dims_json": os.path.relpath(dims_json_path, cwd) if cwd else dims_json_path,
        "logs": [
            {
                "command": r.command,
                "returncode": r.returncode,
                "stdout": r.stdout,
                "stderr": r.stderr,
            }
            for r in logs
        ],
    }
