import time
import requests
import cv2
import numpy as np
from ultralytics import YOLO

# URL of the camera API (mock or real)
CAMERA_API_URL = "http://127.0.0.1:5000/capture"

# Load a pre-trained, lightweight YOLOv8 nano model
# This will download 'yolov8n.pt' the first time it is run
model = YOLO('yolov8n.pt')

def fetch_image(url):
    """
    Fetches an image from the given URL and decodes it for OpenCV/Ultralytics.
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        # Convert the response content (bytes) to a numpy array, then to an OpenCV image
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        return img
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from API: {e}")
        return None
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def detect_objects():
    """
    Fetches an image and runs object detection on it.
    """
    print(f"Fetching image from {CAMERA_API_URL}...")
    img = fetch_image(CAMERA_API_URL)

    if img is not None:
        print("Image fetched successfully. Running inference...")

        # Run inference using YOLOv8
        # We can adjust confidence threshold if needed (e.g., conf=0.5)
        results = model(img)

        # Process results
        for r in results:
            # Print detected classes and bounding boxes
            boxes = r.boxes
            if len(boxes) == 0:
                print("No objects detected.")
            else:
                print(f"Detected {len(boxes)} object(s):")
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    print(f"  - {class_name} (Confidence: {confidence:.2f})")

            # Save the result image to disk (optional)
            output_image_path = 'detection_result.jpg'
            r.save(filename=output_image_path)
            print(f"Detection result saved to {output_image_path}")

def run_loop(interval=5):
    """
    Runs the detection loop continuously at a given interval (in seconds).
    """
    try:
        while True:
            detect_objects()
            print(f"Waiting for {interval} seconds...\n")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopping detector.")

if __name__ == "__main__":
    # To run just once:
    detect_objects()

    # To run continuously, uncomment the following line:
    # run_loop(interval=5)
