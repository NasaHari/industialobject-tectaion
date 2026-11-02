import os
import urllib.request
from ultralytics import YOLO

# URLs for ready-made YOLOv8 models fine-tuned on industrial safety datasets
MODELS_TO_TEST = {
    "Hard Hat Model (Medium)": {
        "url": "https://huggingface.co/keremberke/yolov8m-hard-hat-detection/resolve/main/best.pt",
        "file": "yolov8m_hardhat.pt",
        "description": "Specialized in detecting Hardhats and No-Hardhats."
    },
    "PPE Detection Model (Small)": {
        "url": "https://huggingface.co/keremberke/yolov8s-protective-equipment-detection/resolve/main/best.pt",
        "file": "yolov8s_ppe.pt",
        "description": "Detects gloves, goggles, helmets, masks, shoes, and their absence."
    }
}

def download_model(name, url, filename):
    print(f"Downloading {name} from {url}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename} ({os.path.getsize(filename) / (1024*1024):.2f} MB)")
        return True
    except Exception as e:
        print(f"Failed to download {name}: {e}")
        return False

def evaluate_models():
    print("--- Acquiring Ready-Made Industrial Models ---")

    # 1. Download the models
    for name, info in MODELS_TO_TEST.items():
        if not os.path.exists(info["file"]):
            download_model(name, info["url"], info["file"])
        else:
            print(f"Model {info['file']} already exists locally.")

    print("\n--- Model Capabilities ---")
    best_recommendation = None

    # 2. Inspect the models
    for name, info in MODELS_TO_TEST.items():
        try:
            model = YOLO(info["file"])
            print(f"\n{name} ({info['file']}):")
            print(f"Description: {info['description']}")
            print(f"Detects {len(model.names)} classes:")
            for i, cls_name in model.names.items():
                print(f"  - {cls_name}")

            # Suggest the PPE model as the best overall choice for general industrial hazard avoidance
            if "ppe" in info["file"]:
                best_recommendation = info["file"]

        except Exception as e:
            print(f"Error loading {name}: {e}")

    print("\n" + "="*50)
    print("CONCLUSION & RECOMMENDATION")
    print("="*50)
    if best_recommendation:
        print(f"For a comprehensive industrial hazard detection system, we recommend using: {best_recommendation}")
        print("It covers a wide range of personal protective equipment (PPE) beyond just hard hats.")
        print("\nTo use it in your detector, update detector.py or video_detector.py:")
        print(f"Change: model = YOLO('yolov8n.pt')")
        print(f"To:     model = YOLO('{best_recommendation}')")
    else:
        print("Could not evaluate models. Check your internet connection.")

if __name__ == "__main__":
    evaluate_models()
