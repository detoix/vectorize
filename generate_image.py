import argparse
import mimetypes
import os
import sys
from google import genai
from google.genai import types

def main():
    parser = argparse.ArgumentParser(description="Generate an image using Gemini.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="The prompt for image generation.")
    group.add_argument("--prompt-file", help="Path to a markdown file containing the prompt.")
    
    parser.add_argument("--image", help="Optional input image (e.g., for editing or reference).")
    parser.add_argument("--output", required=True, help="Path to save the generated image.")
    parser.add_argument("--model", default="gemini-3-pro-image-preview", help="Gemini model to use.")
    parser.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY"), help="Google API Key.")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: GOOGLE_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    prompt_text = args.prompt
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt_text = f.read()
        except Exception as e:
            print(f"Error reading prompt file: {e}", file=sys.stderr)
            sys.exit(1)

    client = genai.Client(api_key=args.api_key)

    # Prepare contents
    contents_parts = [types.Part.from_text(text=prompt_text)]
    
    # Handle input image if provided (Multimodal prompt)
    if args.image:
        try:
            with open(args.image, "rb") as f:
                img_bytes = f.read()
            # Simple MIME detection
            mime_type = "image/png"
            if args.image.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif args.image.lower().endswith(".webp"):
                mime_type = "image/webp"
            
            contents_parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
            print(f"Info: Included input image '{args.image}' in the prompt.", file=sys.stderr)
        except Exception as e:
            print(f"Error reading input image: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        # Request both text and image output in one call.
        response = client.models.generate_content(
            model=args.model,
            contents=[types.Content(role="user", parts=contents_parts)],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"], 
                temperature=1.0, 
            ),
        )
        
        image_saved = False

        def write_inline_data(inline_data) -> None:
            nonlocal image_saved
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

            data = inline_data.data
            # AI Studio examples write inline_data.data directly (it is already bytes).
            # Guard against SDK/version differences where this might be a base64 string.
            if isinstance(data, str):
                import base64
                data = base64.b64decode(data)

            out_path = args.output
            if not os.path.splitext(out_path)[1] and inline_data.mime_type:
                out_path += mimetypes.guess_extension(inline_data.mime_type) or ""

            with open(out_path, "wb") as f:
                f.write(data)
            print(f"Image saved to {out_path}")
            image_saved = True
        
        # Parse response for inline image data
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        # Check for inline_data image bytes
                        if part.inline_data:
                            print(f"Found image in response (MIME: {part.inline_data.mime_type})")
                            write_inline_data(part.inline_data)
                            break # Save first image found
                        
                        # Also check if text is present to print it (debug info or explanation)
                        if part.text:
                            print(f"Model Output Text: {part.text}")
                if image_saved: break

        if not image_saved:
            print("Error: No image data found in the response.", file=sys.stderr)
            # Inspecting the response might help debugging
            if response.candidates:
                print(f"Candidate content: {response.candidates[0].content}", file=sys.stderr)
            sys.exit(1) # Exit with error so pipeline stops

    except Exception as e:
        print(f"Error calling Gemini API: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
