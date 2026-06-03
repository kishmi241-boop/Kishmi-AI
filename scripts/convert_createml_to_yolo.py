import os
import json
import shutil
from PIL import Image
from glob import glob

def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size # (width, height)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None, None

def convert_to_yolo_format(annotation, img_w, img_h, class_map):
    label = annotation['label']
    if label not in class_map:
        class_map[label] = len(class_map)
    class_id = class_map[label]
    
    # CreateML coordinates: x (center), y (center), width, height
    coords = annotation['coordinates']
    x_center = coords['x']
    y_center = coords['y']
    w = coords['width']
    h = coords['height']
    
    # YOLO format requires normalized values (0 to 1)
    x_center_norm = x_center / img_w
    y_center_norm = y_center / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    
    return f"{class_id} {x_center_norm} {y_center_norm} {w_norm} {h_norm}"

def main():
    base_dir = r"d:\facial-skin-analysis\DataSets"
    output_dir = r"d:\facial-skin-analysis\merged_yolo_dataset"
    
    # Define splits mapping (folders in datasets to output splits)
    # We will just map whatever parent folder the json is in to train/val/test
    # E.g. Train -> train, Validation -> val, Test -> test
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
    class_map = {}
    
    # Find all json files
    json_files = glob(os.path.join(base_dir, "**", "_annotations.createml.json"), recursive=True)
    
    print(f"Found {len(json_files)} annotation files.")
    
    total_images_processed = 0
    total_labels_processed = 0
    
    for json_file in json_files:
        print(f"Processing: {json_file}")
        # Determine split from folder name
        parent_folder = os.path.basename(os.path.dirname(json_file)).lower()
        if 'train' in parent_folder:
            split = 'train'
        elif 'val' in parent_folder:
            split = 'val'
        elif 'test' in parent_folder:
            split = 'test'
        else:
            split = 'train' # default to train if unknown
            
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error parsing JSON: {json_file}")
                continue
                
        dataset_folder = os.path.dirname(json_file)
        
        for item in data:
            image_name = item.get('image')
            annotations = item.get('annotations', [])
            
            src_image_path = os.path.join(dataset_folder, image_name)
            if not os.path.exists(src_image_path):
                print(f"Warning: Image not found {src_image_path}")
                continue
                
            img_w, img_h = get_image_size(src_image_path)
            if not img_w:
                continue
                
            # Copy image
            dst_image_path = os.path.join(output_dir, 'images', split, image_name)
            # Add prefix if image name already exists to prevent overwrite
            if os.path.exists(dst_image_path):
                name, ext = os.path.splitext(image_name)
                # use dataset folder name to make unique
                dataset_name = os.path.basename(os.path.dirname(dataset_folder)).replace(" ", "_")
                new_image_name = f"{dataset_name}_{name}{ext}"
                dst_image_path = os.path.join(output_dir, 'images', split, new_image_name)
                label_name = f"{dataset_name}_{name}.txt"
            else:
                label_name = os.path.splitext(image_name)[0] + ".txt"
                
            shutil.copy2(src_image_path, dst_image_path)
            
            # Write labels
            dst_label_path = os.path.join(output_dir, 'labels', split, label_name)
            with open(dst_label_path, 'w', encoding='utf-8') as lf:
                for ann in annotations:
                    # Skip empty/invalid annotations
                    if 'coordinates' not in ann or not ann['coordinates']:
                        continue
                        
                    yolo_line = convert_to_yolo_format(ann, img_w, img_h, class_map)
                    lf.write(yolo_line + "\n")
                    total_labels_processed += 1
                    
            total_images_processed += 1
            
    print(f"\nProcessing complete!")
    print(f"Total Images: {total_images_processed}")
    print(f"Total Labels: {total_labels_processed}")
    
    # Write dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write("path: ./  # dataset root dir (relative to the colab notebook or training script)\n")
        f.write("train: images/train  # train images\n")
        f.write("val: images/val  # val images\n")
        f.write("test: images/test  # test images\n\n")
        f.write(f"nc: {len(class_map)}  # number of classes\n\n")
        f.write("names:\n")
        # Sort by class ID to ensure correct order
        sorted_classes = sorted(class_map.items(), key=lambda x: x[1])
        for class_name, class_id in sorted_classes:
            f.write(f"  {class_id}: {class_name}\n")
            
    print(f"\nCreated dataset.yaml at {yaml_path}")
    print("Class mapping:")
    for class_name, class_id in sorted_classes:
        print(f"  {class_id}: {class_name}")

if __name__ == "__main__":
    main()
