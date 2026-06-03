from ultralytics import YOLO
import os

def train_yolo_model():
    print("==================================================")
    print("🚀 STARTING AURA OBJECT DETECTION MODEL TRAINING")
    print("==================================================")
    
    # Load a pretrained YOLOv8 Nano model for fast training
    model = YOLO('yolov8n.pt')
    
    yaml_path = r"d:\Kishmi\yolo_dataset\dataset.yaml"
    
    if not os.path.exists(yaml_path):
        print(f"❌ Error: dataset.yaml not found at {yaml_path}")
        return
        
    print(f"📦 Training with dataset: {yaml_path}")
    
    # Train the model
    # We use patience=10 for early stopping, epochs=30 to reach 90+ mAP without taking too long
    results = model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        patience=10,
        batch=16,
        name='aura_skin_detection',
        exist_ok=True,
        workers=0 # Important for Windows to avoid multiprocessing issues
    )
    
    print("\n🎉 YOLO MODEL TRAINING COMPLETE AND SUCCESSFUL!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print("==================================================\n")

if __name__ == "__main__":
    train_yolo_model()
