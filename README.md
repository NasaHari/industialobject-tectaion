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
   pip install ultralytics flask requests opencv-python
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

## Advanced Usage for Industrial Scenarios

Based on the requirements to handle large datasets of industrial images and support moving cameras for hazard detection, two advanced scripts are included.

### 1. Fine-Tuning on a Custom Industrial Dataset

If you have a large dataset of industrial images (e.g., hard hats, forklifts, specific machinery), you can fine-tune the YOLO model for higher accuracy using `train_industrial.py`.

1. Prepare your dataset in YOLO format and create a `data.yaml` file describing the classes and paths to your images.
2. Open `train_industrial.py` and modify the `model.train()` call to point to your `data.yaml`.
3. Run the script:
   ```bash
   python train_industrial.py
   ```
   This will output a new model weight file (e.g., `best.pt`) which you can then use in the detector.

### 2. Video Stream Detection (Moving Cameras)

For a moving camera on an industrial site, a continuous video stream is more appropriate than fetching single images. Use `video_detector.py` to connect to an RTSP, HTTP stream, or a local webcam. It uses YOLO's built-in tracking capabilities to persist object IDs across frames.

1. Open `video_detector.py`.
2. Scroll to the bottom and update the `stream_url` to match your camera's address (e.g., `rtsp://192.168.1.100:554/stream`).
3. Run the script:
   ```bash
   python video_detector.py
   ```
   A window will pop up showing the live video feed with bounding boxes and tracking IDs. Press `q` to quit.

### 3. Using Ready-Made Industrial Models

If you don't want to train your own model or don't have a dataset, you can use pre-trained open-source models fine-tuned specifically for industrial safety (e.g., detecting hard hats, goggles, gloves, etc.).

We have included a script to automatically download and evaluate some of the best publicly available models:

```bash
python download_evaluate_models.py
```

This script will download the models, inspect their capabilities, and provide a recommendation. Once downloaded, you can plug the recommended `.pt` file directly into your `detector.py` or `video_detector.py` by changing the model initialization line:

```python
# In detector.py or video_detector.py
# Change this:
model = YOLO('yolov8n.pt')

# To this (for example):
model = YOLO('yolov8s_ppe.pt')
```
