import cv2
from ultralytics import YOLO

def process_video_stream(stream_url="0", model_path="yolov8n.pt"):
    """
    Processes a video stream for continuous object detection and tracking.
    This is ideal for a moving camera in an industrial site.

    Args:
        stream_url: Can be a local webcam index (e.g., '0' or 0),
                    an RTSP stream URL, or an HTTP stream URL.
        model_path: Path to the YOLOv8 model weights.
    """
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)

    print(f"Opening video stream: {stream_url}")
    # Convert '0' (string) to 0 (int) for local webcam
    if str(stream_url).isdigit():
        stream_url = int(stream_url)

    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"Error: Could not open video stream {stream_url}.")
        return

    print("Stream opened. Processing frames... (Press 'q' in the window to quit)")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or failed to grab frame.")
            break

        frame_count += 1

        # We can skip frames to speed up processing on slower devices
        # if frame_count % 2 != 0:
        #     continue

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        # This helps in tracking hazards as the camera moves
        results = model.track(frame, persist=True, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Industrial Hazard Detection Stream", annotated_frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit signal received. Exiting...")
            break

    # Release the video capture object and close display windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    # process_video_stream(stream_url="rtsp://your_camera_ip:port/stream")
    # For testing, we just print instructions since a real GUI might not be available here.
    print("To run the video detector, uncomment the call below with your stream URL.")
    # process_video_stream("0") # Uses the default local webcam
