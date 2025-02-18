from google_vision_client import detect_text, ImageData, ImageDataType
from openai_client import get_completions
import asyncio
import io

async def receipt_understanding(image_data):
    extracted_text = await detect_text(image_data)
    with open("./prompts/receipt_understanding_system_prompt.txt", "r") as file:
        system_prompt = file.read()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": extracted_text}
    ]
    completion = await get_completions(messages=messages)
    return completion

async def main():
    image_path = "./example_images/receipt1.png"
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
        image_data = ImageData(ImageDataType.BASE64, content)
    await receipt_understanding(image_data)

if __name__ == "__main__":
    asyncio.run(main())