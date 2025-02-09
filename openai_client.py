from openai import OpenAI
import asyncio

# Load API key from a local file
with open("/Users/sikaizhu/openai_key.txt", "r") as file:
    api_key =  file.read().strip()
client = OpenAI(api_key=api_key)

async def get_completions(model = "gpt-4o-mini", messages = []):
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

async def main():
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    completion = await get_completions(messages=messages)
    print(completion)

if __name__ == "__main__":
    asyncio.run(main())