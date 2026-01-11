import os
import sys
from google import genai

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY not set", file=sys.stderr)
    sys.exit(1)

client = genai.Client(api_key=api_key)

print("Listing models...")
try:
    for model in client.models.list_models():
        print(f"Model: {model.name}")
        print(f"  DisplayName: {model.display_name}")
        print(f"  SupportedActions: {model.supported_actions}")
        print("-" * 20)
except Exception as e:
    print(f"Error listing models: {e}")
