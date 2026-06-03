import os
import sys
from ultralytics import YOLO
import cv2

def predict_yolo(image_path):
    model_path = r"d:\Kishmi\runs\detect\aura_skin_detection\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Trained model not found at {model_path}. Please run train_yolo.py first.")
        sys.exit(1)
        
    model = YOLO(model_path)
    
    print(f"🔍 Running prediction on {image_path}...")
    results = model(image_path)
    
    # Save the output image with bounding boxes
    output_dir = r"d:\Kishmi\runs\predict"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, r in enumerate(results):
        im_bgr = r.plot()  # plot a BGR numpy array of predictions
        
        output_path = os.path.join(output_dir, f"prediction_{i}.jpg")
        cv2.imwrite(output_path, im_bgr)
        print(f"✅ Saved prediction visualization to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Pick a sample image from the test set
        image_path = r"d:\Kishmi\yolo_dataset\images\test"
        files = os.listdir(image_path) if os.path.exists(image_path) else []
        if files:
            image_path = os.path.join(image_path, files[0])
        else:
            print("Please provide an image path to predict.")
            sys.exit(1)
            
    predict_yolo(image_path)
