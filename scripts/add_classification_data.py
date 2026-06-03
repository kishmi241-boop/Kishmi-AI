import os
import shutil

# Paths
DATASET_ROOT = r"d:\facial-skin-analysis\merged_yolo_dataset"
CLASS_MAPPING = {
    "dry": 1,
    "dry_data": 1,
    "oily": 3,
    "oily_data": 3,
    "normal": 13,
    "normal_data": 13,
    "combination": 14,
    "sensitive": 15
}

# The folders containing classification data
SOURCE_DIRS = [
    {
        "path": r"d:\facial-skin-analysis\DataSets\Skin Type Identification Research\Skin Type Identification Research",
        "split_mapping": {
            "Train": "train",
            "Validation": "val",
            "Test": "test"
        }
    },
    {
        "path": r"d:\facial-skin-analysis\DataSets\Machine-Learning-master\Machine-Learning-master\skin_type",
        "split_mapping": {
            "train": "train",
            "validation": "val"
        }
    }
]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_classification_data():
    total_images_added = 0
    
    for source in SOURCE_DIRS:
        base_path = source["path"]
        if not os.path.exists(base_path):
            print(f"Skipping {base_path} - not found.")
            continue
            
        print(f"Processing source: {base_path}")
        
        for src_split, dst_split in source["split_mapping"].items():
            split_path = os.path.join(base_path, src_split)
            if not os.path.exists(split_path):
                continue
                
            for class_dir in os.listdir(split_path):
                class_path = os.path.join(split_path, class_dir)
                if not os.path.isdir(class_path):
                    continue
                
                # Get the YOLO class ID
                norm_name = class_dir.lower()
                if norm_name not in CLASS_MAPPING:
                    print(f"  Skipping class dir '{class_dir}' - no mapping found.")
                    continue
                    
                class_id = CLASS_MAPPING[norm_name]
                
                # Process images
                dest_img_dir = os.path.join(DATASET_ROOT, "images", dst_split)
                dest_lbl_dir = os.path.join(DATASET_ROOT, "labels", dst_split)
                
                ensure_dir(dest_img_dir)
                ensure_dir(dest_lbl_dir)
                
                for img_name in os.listdir(class_path):
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                        continue
                        
                    src_img_path = os.path.join(class_path, img_name)
                    
                    # Prefix with class and source split to prevent collisions
                    safe_name = f"class_{class_id}_{src_split}_{img_name}"
                    dst_img_path = os.path.join(dest_img_dir, safe_name)
                    
                    # YOLO label text file
                    txt_name = os.path.splitext(safe_name)[0] + ".txt"
                    dst_txt_path = os.path.join(dest_lbl_dir, txt_name)
                    
                    # Copy image
                    shutil.copy2(src_img_path, dst_img_path)
                    
                    # Write YOLO label (full image bounding box)
                    with open(dst_txt_path, 'w') as f:
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                        
                    total_images_added += 1

    print(f"\n✅ Processing complete! Added {total_images_added} images and full-image labels.")

if __name__ == "__main__":
    process_classification_data()
