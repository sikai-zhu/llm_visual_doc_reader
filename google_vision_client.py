from google.cloud import vision
from google.auth import load_credentials_from_file
import logging
import io
import asyncio
from enum import Enum
import aiohttp


# Load Google credentials
credentials, project_id = load_credentials_from_file("/Users/sikaizhu/google_credentials.json")

class ImageDataType(Enum):
    URL = 'url'
    BASE64 = 'base64'


class ImageData:
    def __init__(self, data_type: ImageDataType, data):
        self.data_type = data_type
        self.data = data


async def get_image_content(image_data: ImageData):
    if image_data.data_type == ImageDataType.URL:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_data.data) as response:
                if response.status == 200:
                    return await response.read()  # Use .read() to get the content as bytes
                else:
                    raise ValueError(f"Failed to download image from URL. Status code: {response.status}")
    elif image_data.data_type == ImageDataType.BASE64:
        return image_data.data  # Return the actual base64 content


async def detect_text(image_data: ImageData):
    async with vision.ImageAnnotatorAsyncClient(credentials=credentials) as client:
        content = await get_image_content(image_data)
        image = vision.Image(content=content)

        # Correct way to create request
        request = vision.AnnotateImageRequest(
            image=image,
            features=[vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)]
        )

        batch_request = vision.BatchAnnotateImagesRequest(requests=[request])

        try:
            # Properly call async batch annotation
            response = await client.batch_annotate_images(batch_request)
        except Exception as e:
            logging.error(f"Error during image annotation: {e}")
            return ""

        # Extract response data
        texts = response.responses[0].text_annotations
        if not texts:
            print("No text found")
            return ""

        extracted_text = texts[0].description
        logging.debug("Extracted Text:\n%s", extracted_text)

        # Check for errors
        if response.responses[0].error.message:
            raise Exception(f"Error: {response.responses[0].error.message}")

        return extracted_text


async def main():
    image_path = "./example_images/receipt1.png"
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
        image_data = ImageData(ImageDataType.BASE64, content)
        result = await detect_text(image_data)
    logging.debug(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
