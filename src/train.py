from ultralytics import YOLO

def train_model():
    # Load a pre-trained YOLOv8 segmentation model (nano size for mobile/fast inference)
    model = YOLO("yolov8n-seg.pt")

    # Train the model on the custom dataset
    # You will need to replace 'dataset.yaml' with the path to your actual data configuration file
    # Ensure your dataset is split into train/val/test sets
    print("Starting training...")
    results = model.train(
        data="dataset.yaml",  # Path to your dataset YAML file
        epochs=50,           # Number of training epochs
        imgsz=640,           # Image size
        batch=16,            # Batch size
        device="cpu",        # Use '0' if you have an Nvidia GPU, otherwise 'cpu'
        project="models",    # Directory to save training runs
        name="skin_analysis_v1" # Name of this specific training run
    )
    
    print("Training complete. Models saved to models/skin_analysis_v1/weights")

if __name__ == "__main__":
    train_model()
