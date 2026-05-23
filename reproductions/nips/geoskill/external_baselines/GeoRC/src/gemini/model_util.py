from google import genai
from google.genai import types

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

def image_and_text_to_text(image_path, prompt, model_id="gemini-2.5-flash"):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    response = client.models.generate_content(
    model=model_id,
    contents=[
    types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/png',
    ),
    prompt]
    )
    return response.text
