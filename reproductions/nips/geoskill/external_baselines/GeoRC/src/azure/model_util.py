from openai import AzureOpenAI
import os
import base64
from PIL import Image
from io import BytesIO

def convert_to_jpeg_and_encode(image_path):
    im = Image.open(image_path)
    rgb_im = im.convert('RGB')

    image_buffer = BytesIO()
    rgb_im.save(image_buffer, format='JPEG')
    image_bytes = image_buffer.getvalue()
    # print("size of image ", (image_bytes.__sizeof__()/1024))
    return base64.b64encode(image_bytes).decode('utf-8')

def image_and_text_to_text(image_path, prompt, endpoint, api_key, api_version, model="gpt-5"):
    
    max_size = 20 * 1024 * 1024  # 20 MB

    if os.path.getsize(image_path) > max_size:
        image_data = convert_to_jpeg_and_encode(image_path)
        format = "jpeg"
    else:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        format = "png"
    
    client = AzureOpenAI(
        azure_endpoint = endpoint,
        api_key=api_key,
        api_version=api_version
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{format};base64,{image_data}"
                    },
                },
            ],
        }
    ],
    )
    return response.choices[0].message.content