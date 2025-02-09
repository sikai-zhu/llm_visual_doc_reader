from google_vision_client import detect_text
from openai_client import get_completions
import asyncio

async def receipt_understanding(image_path):
    extracted_text = await detect_text(image_path)
    with open("./prompts/receipt_understanding_system_prompt.txt", "r") as file:
        system_prompt = file.read()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": extracted_text}
    ]
    completion = await get_completions(messages=messages)
    print(completion)

async def main():
    image_path = "./example_images/receipt1.png"
    await receipt_understanding(image_path)

if __name__ == "__main__":
    asyncio.run(main())