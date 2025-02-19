from flask import Flask, jsonify, request
from google_vision_client import ImageData, ImageDataType
from doc_understanding import receipt_understanding
import asyncio


# Create the Flask app
app = Flask(__name__)

# Handle POST request (example)
@app.route('/receipt_understanding', methods=['POST'])
def doc_understanding():
    data = request.get_json()
    image_data = ImageData(ImageDataType(data["data_type"]), data["data"])
    extracted_data = asyncio.run(receipt_understanding(image_data))
    return jsonify({"data": extracted_data}), 200

# Start the web server
if __name__ == '__main__':
    app.run(debug=True)
