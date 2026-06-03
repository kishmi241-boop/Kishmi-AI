# train_colab.py
# This script is designed to be run in Google Colab to train your custom model.

import os
import subprocess

def install_dependencies():
    print("Installing Ultralytics YOLOv8...")
    subprocess.run(["pip", "install", "-q", "ultralytics"])

def main():
    # 1. Install YOLOv8 if not already installed
    try:
        from ultralytics import YOLO
    except ImportError:
        install_dependencies()
        from ultralytics import YOLO

    dataset_yaml = "/content/merged_yolo_dataset/dataset.yaml"
    
    if not os.path.exists(dataset_yaml):
        print(f"❌ Error: {dataset_yaml} not found.")
        print("Please upload merged_yolo_dataset.zip to Colab and unzip it to /content/")
        print("Command to unzip in Colab: !unzip -q merged_yolo_dataset.zip -d /content/")
        return

    print("🚀 Starting YOLOv8 Training on Custom Dataset...")
    
    # Load a pretrained model (YOLOv8 Medium for good accuracy)
    model = YOLO("yolov8m.pt")
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=50,             # Number of epochs
        imgsz=640,             # Image size
        batch=16,              # Batch size (reduce if out of memory)
        device=0,              # Use GPU (0)
        project="/content/runs", # Where to save results
        name="custom_skin_model"
    )
    
    print("✅ Training complete! Best model is saved at /content/runs/custom_skin_model/weights/best.pt")
    print("Download this best.pt file to your local computer to use in your app.")

if __name__ == "__main__":
    main()
