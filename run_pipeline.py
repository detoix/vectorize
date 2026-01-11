import argparse
import sys
import os
from pipeline_lib import PipelineError, run_vectorize_pipeline

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

    try:
        run_vectorize_pipeline(
            image=args.image,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            mask=args.mask,
            output=args.output,
            skip_gen=args.skip_gen,
            debug_dir=args.debug_dir,
            api_key=args.api_key or os.environ.get("GOOGLE_API_KEY"),
        )
    except PipelineError as e:
        print(f"Error: {e}", file=sys.stderr)
        if getattr(e, "details", None):
            details = e.details
            if "stderr" in details and details["stderr"]:
                print(f"Stderr: {details['stderr']}", file=sys.stderr)
        return 1

    print(f"\nPipeline complete. Output saved to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
