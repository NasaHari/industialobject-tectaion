import os
from flask import Flask, send_file

app = Flask(__name__)

# The path to the test image we want to serve
TEST_IMAGE_PATH = 'test_image.jpg'

@app.route('/capture', methods=['GET'])
def capture():
    """
    Simulates a camera capturing an image by returning a static test image.
    """
    if os.path.exists(TEST_IMAGE_PATH):
        return send_file(TEST_IMAGE_PATH, mimetype='image/jpeg')
    else:
        return {"error": "Test image not found"}, 404

if __name__ == '__main__':
    # Run on port 5000
    app.run(host='0.0.0.0', port=5000)
