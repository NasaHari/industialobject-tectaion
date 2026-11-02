from ultralytics import YOLO

def train_custom_model():
    """
    Demonstrates how to fine-tune a YOLOv8 model on a custom dataset of industrial images.
    """
    # 1. Load a pre-trained model (starting with nano for fast training/inference)
    model = YOLO('yolov8n.pt')

    # 2. Train the model
    # To run this, you need a 'data.yaml' file that defines your dataset.
    # An example data.yaml looks like:
    #
    # train: /path/to/train/images
    # val: /path/to/val/images
    # nc: 2 # number of classes
    # names: ['helmet', 'forklift'] # class names
    print("Starting training process...")
    try:
        # We limit epochs and imgsz for demonstration.
        # For a real large dataset, increase epochs (e.g., 100) and imgsz (e.g., 640)
        results = model.train(
            data='construction-ppe.yaml', # Uses built-in dataset that auto-downloads
            epochs=2,          # Very small number for demonstration
            imgsz=320,         # Smaller image size for faster training
            device='cpu',      # Use '0' if you have a GPU, or 'mps' for Apple Silicon
            project='industrial_model', # Output directory
            name='run_1'       # Run name
        )
        print("Training completed successfully.")

        # 3. Evaluate the model on the validation set
        metrics = model.val()
        print(f"Validation metrics: {metrics}")

    except Exception as e:
        print(f"Error during training: {e}")
        print("Make sure you have a valid data.yaml and the dataset is accessible.")

if __name__ == "__main__":
    print("Starting automatic training process on Construction PPE dataset...")
    train_custom_model()
