from google.cloud import vision
from google.auth import load_credentials_from_file
import logging
import io
import asyncio

credentials, project_id = load_credentials_from_file("/Users/sikaizhu/google_credentials.json")

async def detect_text(image_path):
    async with vision.ImageAnnotatorAsyncClient(credentials=credentials) as client:
        # Load image
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Correct way to create request
        request = vision.AnnotateImageRequest(
            image=image,
            features=[vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)]
        )

        batch_request = vision.BatchAnnotateImagesRequest(requests=[request])

        # Properly call async batch annotation
        response = await client.batch_annotate_images(batch_request)

        # Extract response data
        texts = response.responses[0].text_annotations
        if not texts:
            print("No text found")
            return ""

        extracted_text = texts[0].description
        logging.debug("Extracted Text:\n", extracted_text)

        # Check for errors
        if response.responses[0].error.message:
            raise Exception(f"Error: {response.responses[0].error.message}")

        return extracted_text

async def main():
    image_path = "./example_images/receipt1.png"
    result = await detect_text(image_path)
    logging.debug(result)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
