import argparse
import os
import sys
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Define the output schema using Pydantic
class Dimension(BaseModel):
    value: float
    unit: str = Field(description="The unit of measurement: m, cm, mm, ft, or in.")

class RealDimensions(BaseModel):
    real_x: Optional[Dimension] = Field(default=None, alias="real-x", description="The real-world width.")
    real_y: Optional[Dimension] = Field(default=None, alias="real-y", description="The real-world height.")

def main():
    parser = argparse.ArgumentParser(description="Extract real-world dimensions from a floorplan image.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use.")
    parser.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY"), help="Google API Key.")
    parser.add_argument(
        "--prompt-file",
        default="ai_extract_real_dims_prompt.md",
        help="Path to a prompt .md/.txt file (or '-' to read from stdin). Defaults to ai_extract_real_dims_prompt.md.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Error: GOOGLE_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=args.api_key)

    with open(args.image, "rb") as f:
        image_bytes = f.read()

    # Simple MIME type detection
    mime_type = "image/png"
    if args.image.lower().endswith(".jpg") or args.image.lower().endswith(".jpeg"):
        mime_type = "image/jpeg"

    if args.prompt_file and args.prompt_file != "ai_extract_real_dims_prompt.md":
        if args.prompt_file == "-":
            prompt = sys.stdin.read()
        else:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read()
        if not prompt.strip():
            print("Error: prompt file is empty.", file=sys.stderr)
            sys.exit(1)
    else:
        # Default prompt file, with a built-in fallback for robustness.
        prompt_path = "ai_extract_real_dims_prompt.md"
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read()
        else:
            prompt = (
                "Extract the real-world outer dimensions of the wall-union bounding box. "
                "Return a JSON object with optional keys 'real-x' and 'real-y'."
            )
        if not prompt.strip():
            print(f"Error: prompt file '{prompt_path}' is empty.", file=sys.stderr)
            sys.exit(1)

    try:
        response = client.models.generate_content(
            model=args.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RealDimensions,
                temperature=0.0,
            ),
        )
        
        # Output the parsed JSON
        if response.parsed:
            print(response.parsed.model_dump_json(by_alias=True, indent=2))
        else:
            # Fallback if parsing isn't populated
            print(response.text)

    except Exception as e:
        print(f"Error calling Gemini API: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
