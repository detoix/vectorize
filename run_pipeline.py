import argparse
import subprocess
import sys
import os
import json

def run_command(command, capture_output=True, env=None):
    """Runs a shell command and returns the output."""
    # Filter out None values from command list
    command = [str(c) for c in command if c is not None]
    # Avoid leaking secrets in logs (e.g., --api-key ...).
    printable = command[:]
    for i, token in enumerate(printable[:-1]):
        if token in {"--api-key", "--apikey", "--key"}:
            printable[i + 1] = "***"
    print(f"Running: {' '.join(printable)}")
    try:
        result = subprocess.run(
            command, 
            capture_output=capture_output, 
            text=True, 
            check=True,
            env=env  # Pass environment variables
        )
        # When capture_output=False, subprocess sets stdout/stderr to None.
        return result.stdout.strip() if result.stdout is not None else ""
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}", file=sys.stderr)
        if e.stderr:
            print(f"Stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the full Vectorize pipeline.")
    parser.add_argument("--image", required=True, help="Input floorplan image file.")
    parser.add_argument("--prompt", help="Prompt for mask generation (if generating mask).")
    parser.add_argument("--prompt-file", help="File containing prompt for mask generation. Defaults to 'mask_prompt.md' if it exists.")
    parser.add_argument("--mask", help="Path to the mask file (input or output). Default: mask.png", default="mask.png")
    parser.add_argument("--output", help="Path to the output DSL file. Default: output.dsl", default="output.dsl")
    parser.add_argument("--skip-gen", action="store_true", help="Skip mask generation (use existing mask).")
    parser.add_argument("--debug-dir", help="Directory for debug outputs.", default="debug_pipeline")
    parser.add_argument("--api-key", help="Google API Key. Defaults to GOOGLE_API_KEY env var.")
    args = parser.parse_args()

    # 1. Handle API Key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: API Key not found. Set GOOGLE_API_KEY env var or pass --api-key.", file=sys.stderr)
        sys.exit(1)
    
    # Update env for subprocesses
    env = os.environ.copy()
    env["GOOGLE_API_KEY"] = api_key

    if not os.path.exists(args.image):
        print(f"Error: Input image '{args.image}' not found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.debug_dir, exist_ok=True)
    
    # 2. Determine Prompt (if needed)
    prompt_file = args.prompt_file
    if not args.skip_gen and not args.prompt and not prompt_file:
        # Check for default prompt file
        default_prompt_path = "mask_prompt.md"
        if os.path.exists(default_prompt_path):
            print(f"Info: No prompt specified, using default file: {default_prompt_path}")
            prompt_file = default_prompt_path
        else:
             print("Error: --prompt or --prompt-file is required (and 'mask_prompt.md' not found).", file=sys.stderr)
             sys.exit(1)

    # 3. Generate Mask
    if not args.skip_gen:
        gen_cmd = [sys.executable, "generate_image.py", "--output", args.mask, "--image", args.image, "--api-key", api_key]
        if args.prompt:
            gen_cmd.extend(["--prompt", args.prompt])
        if prompt_file:
            gen_cmd.extend(["--prompt-file", prompt_file])
            
        run_command(gen_cmd, env=env)
    else:
        if not os.path.exists(args.mask):
             print(f"Error: Mask file '{args.mask}' not found and generation skipped.", file=sys.stderr)
             sys.exit(1)

    # 4. Extract Real Dimensions
    dims_json_path = os.path.join(args.debug_dir, "dims.json")
    print("Extracting real dimensions...")
    # Pass api_key explicitly
    dims_output = run_command(
        [sys.executable, "ai_extract_real_dims.py", "--image", args.image, "--api-key", api_key],
        env=env
    )
    
    try:
        json.loads(dims_output)
        with open(dims_json_path, "w") as f:
            f.write(dims_output)
    except json.JSONDecodeError:
        print(f"Error: ai_extract_real_dims.py did not return valid JSON:\n{dims_output}", file=sys.stderr)
        sys.exit(1)

    # 5. Calculate Scale
    print("Calculating scale...")
    scale_str = run_command([
        sys.executable, "extract_scale.py",
        "--input", args.image,
        "--real-json", dims_json_path,
        "--format", "plain"
    ], env=env)
    
    try:
        scale = float(scale_str)
        print(f"Calculated scale: {scale} meters/pixel")
    except ValueError:
         print(f"Error: Could not parse scale '{scale_str}' as float.", file=sys.stderr)
         sys.exit(1)

    # 6. Vectorize
    print("Vectorizing...")
    run_command([
        sys.executable, "mask2dsl.py",
        "--input", args.mask,
        "--scale", str(scale),
        "--out", args.output,
        "--debug-dir", args.debug_dir
    ], capture_output=False, env=env)

    print(f"\nPipeline complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()
