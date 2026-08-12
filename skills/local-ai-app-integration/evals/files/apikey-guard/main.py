import os

from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    raise SystemExit("No API key set. Please enter your OpenAI key.")

client = OpenAI(api_key=api_key)
