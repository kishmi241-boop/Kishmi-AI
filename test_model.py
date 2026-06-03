# test_model.py
"""Simple script to verify that your trained YOLO model works.
It loads the best.pt model (expected in ./models or ./trained_model.zip), runs inference on a sample image,
and prints the detected classes with confidence.
If you have a custom image, replace SAMPLE_IMAGE_PATH.
"""
import os
import base64
import cv2
import numpy as np
from ultralytics import YOLO

# ----- Locate the model ------------------------------------------------------
MODEL_PATHS = [
    "models/best.pt",
    "trained_model/best.pt",
    "trained_model.zip",  # fallback: unzip and look for best.pt inside
]
model_path = None
for p in MODEL_PATHS:
    if os.path.isdir(p):
        # if it's a zip directory, try to find best.pt inside
        for root, _, files in os.walk(p):
            if "best.pt" in files:
                model_path = os.path.join(root, "best.pt")
                break
    elif os.path.isfile(p):
        # If it's a zip file, unzip to a temp folder
        if p.lower().endswith('.zip'):
            import zipfile, tempfile
            tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(p, 'r') as z:
                z.extractall(tmpdir)
            # search extracted folder for best.pt
            for root, _, files in os.walk(tmpdir):
                if "best.pt" in files:
                    model_path = os.path.join(root, "best.pt")
                    break
        else:
            model_path = p
    if model_path:
        break

if not model_path:
    raise FileNotFoundError("Could not locate best.pt model. Ensure it exists in ./models or ./trained_model.zip")

print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# ----- Sample image ---------------------------------------------------------
# You can replace this with any local image path to test your own data.
SAMPLE_IMAGE_PATH = "sample_face.jpg"
# If the sample does not exist, download a public example image.
if not os.path.isfile(SAMPLE_IMAGE_PATH):
    import urllib.request
    url = "https://raw.githubusercontent.com/ultralytics/assets/main/images/bus.jpg"  # generic image for demo
    urllib.request.urlretrieve(url, SAMPLE_IMAGE_PATH)
    print(f"Downloaded example image to {SAMPLE_IMAGE_PATH}")

img = cv2.imread(SAMPLE_IMAGE_PATH)
if img is None:
    raise ValueError(f"Failed to read image {SAMPLE_IMAGE_PATH}")

# ----- Run inference --------------------------------------------------------
results = model.predict(img, imgsz=640, conf=0.25, verbose=False)
boxes = results[0].boxes

if boxes is None or len(boxes) == 0:
    print("No detections found.")
else:
    print("Detections:")
    for box in boxes:
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])
        # YOLO model stores class names in model.names
        class_name = model.names.get(cls_idx, f"class_{cls_idx}")
        print(f"  - {class_name}: {conf:.2%}")

# ----- Optional: save image with boxes --------------------------------------
output_path = "detections_output.jpg"
annotated = results[0].plot()  # returns image with boxes drawn
cv2.imwrite(output_path, annotated)
print(f"Annotated image saved to {output_path}")
