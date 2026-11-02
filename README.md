# Industrial Object Detection Demo

This project demonstrates a lightweight and fast object detection solution that interfaces with a camera exposing an HTTP API. It uses [YOLOv8](https://github.com/ultralytics/ultralytics) (`yolov8n` nano model) to ensure high speed and efficiency.

Since real cameras might not be available during development, this project includes a mock camera API to simulate the image capture process.

## Prerequisites

Python 3.8+ is recommended.

1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install ultralytics flask requests opencv-python-headless
   ```

## Project Structure

- `mock_camera_api.py`: A lightweight Flask server simulating a camera API. It exposes a `/capture` endpoint that returns an image.
- `detector.py`: The object detection client. It fetches an image from the API and runs YOLOv8 inference.
- `test_image.jpg`: A sample image used by the mock API (you can replace this with your own image for testing).

## How to Run the Demo

### 1. Start the Mock Camera API

In one terminal (with your virtual environment activated), start the Flask server:

```bash
python mock_camera_api.py
```
This server runs on `http://0.0.0.0:5000`. It will serve the `test_image.jpg` when requested.

### 2. Run the Object Detector

In a second terminal (also with your virtual environment activated), run the detector:

```bash
python detector.py
```

The script will:
- Connect to the mock camera API (`http://127.0.0.1:5000/capture`).
- Fetch the image.
- Download the `yolov8n.pt` model weights (if not already downloaded).
- Perform object detection.
- Print out the detected objects and their confidence scores.
- Save an output image with bounding boxes to `detection_result.jpg`.

## Integrating with a Real Camera

Once you have a real camera that exposes an HTTP API for capturing images:

1. Open `detector.py`.
2. Locate the `CAMERA_API_URL` variable.
3. Change the URL to point to your real camera's API endpoint (e.g., `http://<camera_ip>/api/capture`).

If you need the detector to run continuously monitoring an industrial line, you can uncomment `run_loop(interval=5)` at the bottom of `detector.py`.
