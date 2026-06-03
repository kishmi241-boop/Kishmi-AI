import os
import glob
import json
import zipfile
import shutil
from PIL import Image

def get_image_size(zip_file, image_path):
    with zip_file.open(image_path) as f:
        img = Image.open(f)
        return img.size

def prep_yolo_data():
    base_dir = r"d:\Kishmi"
    output_dir = os.path.join(base_dir, "yolo_dataset")
    
    # Class mapping
    class_map = {
        'Acne': 0,
        'Dark Circle': 1,
        'Dark_circle - v1 2023-02-07 10-19am': 1,
        'oily-skin': 2,
        'oilyskin': 2,
        'dryskin': 3,
        'redness': 4,
        'visible-pores': 5,
        'object': 6 # Spots / Eyebags from the 4th dataset
    }
    
    # Create directories
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
    zips = [
        "Acne - Dark Circle.v1i.createml.zip",
        "oily-skin-detection.v1i.createml.zip",
        "Skin Analysis Oily.v2i.createml.zip",
        "Spots- Cirlcles- Eyebag.v414i.createml.zip"
    ]
    
    processed_count = 0
    for zip_name in zips:
        zip_path = os.path.join(base_dir, zip_name)
        if not os.path.exists(zip_path):
            print(f"Skipping missing {zip_name}")
            continue
            
        print(f"Processing {zip_name}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            for split in ['train', 'valid', 'test']:
                json_path = f"{split}/_annotations.createml.json"
                if json_path not in z.namelist():
                    continue
                    
                data = json.loads(z.read(json_path))
                
                for item in data:
                    img_filename = item['image']
                    img_zip_path = f"{split}/{img_filename}"
                    
                    if img_zip_path not in z.namelist():
                        continue
                        
                    try:
                        w, h = get_image_size(z, img_zip_path)
                    except Exception as e:
                        print(f"Error reading image {img_zip_path}: {e}")
                        continue
                        
                    # Write label file
                    label_filename = os.path.splitext(img_filename)[0] + ".txt"
                    label_path = os.path.join(output_dir, 'labels', split, label_filename)
                    
                    valid_annotations = False
                    with open(label_path, 'w') as lf:
                        for ann in item.get('annotations', []):
                            raw_label = ann.get('label')
                            if raw_label not in class_map:
                                continue
                                
                            class_id = class_map[raw_label]
                            coords = ann['coordinates']
                            
                            # CreateML format uses center x, y
                            center_x = coords['x'] / w
                            center_y = coords['y'] / h
                            box_w = coords['width'] / w
                            box_h = coords['height'] / h
                            
                            # Clamp values to [0, 1]
                            center_x = max(0, min(1, center_x))
                            center_y = max(0, min(1, center_y))
                            box_w = max(0, min(1, box_w))
                            box_h = max(0, min(1, box_h))
                            
                            lf.write(f"{class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}\n")
                            valid_annotations = True
                            
                    # Only extract image if it has valid annotations
                    if valid_annotations:
                        target_img_path = os.path.join(output_dir, 'images', split, img_filename)
                        with z.open(img_zip_path) as source, open(target_img_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        processed_count += 1
                    else:
                        os.remove(label_path) # Remove empty label file
                        
    print(f"Dataset preparation complete. Extracted {processed_count} images.")
    
    # Write dataset.yaml
    yaml_content = f"""path: {output_dir}
train: images/train
val: images/valid
test: images/test

names:
  0: acne
  1: dark_circle
  2: oily_skin
  3: dry_skin
  4: redness
  5: pores
  6: spots_eyebags
"""
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
    print(f"Created {os.path.join(output_dir, 'dataset.yaml')}")

if __name__ == "__main__":
    prep_yolo_data()
