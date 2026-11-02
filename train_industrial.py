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
            data='coco8.yaml', # Replace 'coco8.yaml' with your 'data.yaml'
            epochs=1,          # Set to 1 for quick testing. Use 50-100+ for actual training.
            imgsz=640,         # Image size
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
    # WARNING: Training can take a long time and requires a dataset.
    # This script is provided as a template for fine-tuning.
    print("Run `train_custom_model()` when you have your dataset ready.")
    # train_custom_model()
