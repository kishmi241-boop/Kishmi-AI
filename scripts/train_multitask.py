import os
import zipfile
import io
import csv
import sys
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Ensure scripts folder and root folder are in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from scripts.face_cropper import FaceCropper

# Define the PyTorch Dataset for Multi-Task attributes
class MultitaskDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        """
        Args:
            images: List of PIL Images (pre-cropped).
            targets: List or array of shape (N, 4) with binary labels.
            transform: PyTorch transform pipeline.
        """
        self.images = images
        self.targets = torch.FloatTensor(targets)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

# Multi-head classifier outputting logits for each of the 4 binary attributes
class MultitaskHead(nn.Module):
    def __init__(self, in_features):
        super(MultitaskHead, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc_clear = nn.Linear(in_features, 1)
        self.fc_oily = nn.Linear(in_features, 1)
        self.fc_circles = nn.Linear(in_features, 1)
        self.fc_wrinkle = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.dropout(x)
        return torch.cat([
            self.fc_clear(x),
            self.fc_oily(x),
            self.fc_circles(x),
            self.fc_wrinkle(x)
        ], dim=1)

def train_multitask_models():
    zip_path = r"d:\Kishmi\archive (4).zip"
    model_output_path = r"d:\Kishmi\scripts\aura_multitask_models.pth"
    
    print("==================================================")
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
        
    print("🚀 STARTING AURA MULTITASK SKIN ATTRIBUTE CNN TRAINING")
    print("==================================================")
    
    if not os.path.exists(zip_path):
        print(f"❌ Error: Dataset archive not found at {zip_path}")
        sys.exit(1)
        
    target_attributes = ["clear_skin", "oily_skin", "dark_circles", "wrinkle"]
    
    print("📋 Parsing Attributes.csv from ZIP...")
    csv_rows = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        try:
            info = zip_ref.getinfo("Attributes.csv")
            csv_text = zip_ref.read(info).decode('utf-8')
            reader = csv.DictReader(io.StringIO(csv_text))
            for row in reader:
                csv_rows.append(row)
        except Exception as e:
            print(f"❌ Error reading Attributes.csv: {e}")
            sys.exit(1)
            
    print(f"✅ Successfully parsed {len(csv_rows)} rows of image attributes!")
    
    # Filter out blurry images
    valid_rows = [row for row in csv_rows if row.get("blurry_image") == "0"]
    print(f"🧹 Filtered blurry images: {len(valid_rows)} sharp images available.")
    
    # Sample a representative subset of 2000 images to crop and train on CPU efficiently
    random.seed(42)
    sample_size = min(2000, len(valid_rows))
    sampled_rows = random.sample(valid_rows, sample_size)
    print(f"Sampling {sample_size} images for balanced multitask training...")
    
    raw_images = []
    targets = []
    
    print("\n📦 Loading images and cropping faces with MediaPipe...")
    cropper = FaceCropper(min_detection_confidence=0.4)
    
    detected_count = 0
    fallback_count = 0
    processed_count = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for row in sampled_rows:
            img_id = f"Images/{row['image_id']}"
            try:
                info = zip_ref.getinfo(img_id)
                img_bytes = zip_ref.read(info)
                
                # Crop face
                cropped_img, was_detected = cropper.crop_face(img_bytes, target_size=(224, 224), margin_ratio=0.25)
                
                raw_images.append(cropped_img)
                
                # Extract binary attributes labels
                y_vector = [
                    int(row.get("clear_skin", 0)),
                    int(row.get("oily_skin", 0)),
                    int(row.get("dark_circles", 0)),
                    int(row.get("wrinkle", 0))
                ]
                targets.append(y_vector)
                
                if was_detected:
                    detected_count += 1
                else:
                    fallback_count += 1
                    
                processed_count += 1
                if processed_count % 200 == 0:
                    print(f"  Processed {processed_count} / {sample_size} images...")
            except Exception as e:
                pass
                
    cropper.close()
    print(f"\n✅ Finished processing {processed_count} images successfully!")
    print(f"  -> Face detection success: {detected_count} ({detected_count/max(1, processed_count)*100:.1f}%)")
    print(f"  -> Fallback center crop: {fallback_count} ({fallback_count/max(1, processed_count)*100:.1f}%)")
    
    X_indices = np.arange(len(raw_images))
    y = np.array(targets)
    
    # Print label distribution
    print("\n📊 Attribute Distributions:")
    for i, attr in enumerate(target_attributes):
        pos_count = np.sum(y[:, i] == 1)
        neg_count = np.sum(y[:, i] == 0)
        print(f"  - {attr}: {pos_count} positive, {neg_count} negative (Pos Ratio: {pos_count/len(y)*100:.1f}%)")
        
    # Split 80/20 train/val
    train_idx, val_idx = train_test_split(X_indices, test_size=0.20, random_state=42)
    
    train_images = [raw_images[i] for i in train_idx]
    val_images = [raw_images[i] for i in val_idx]
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    # Define Data Transforms (Heavy Augmentation)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MultitaskDataset(train_images, y_train, transform=train_transforms)
    val_dataset = MultitaskDataset(val_images, y_val, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize Pretrained ResNet18 Model
    print("\n🎯 Loading pre-trained ResNet18 model...")
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=True)
        
    # Unfreeze all layers for better fine-tuning accuracy
    for param in model.parameters():
        param.requires_grad = True
            
    num_features = model.fc.in_features
    model.fc = MultitaskHead(num_features)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🏋️ Training on device: {device}")
    
    # Compute positive weights for BCE loss to handle imbalance
    pos_weights = []
    for i in range(4):
        pos_count = np.sum(y_train[:, i] == 1)
        neg_count = np.sum(y_train[:, i] == 0)
        pos_weight = neg_count / max(1.0, pos_count)
        pos_weights.append(pos_weight)
        
    pos_weights_tensor = torch.FloatTensor(pos_weights).to(device)
    print(f"⚖️ Applied BCE Positive Weights: {pos_weights}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
    
    # Optimizer and Learning Rate Scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Training Loop
    epochs = 25
    best_avg_acc = 0.0
    
    print("\n🚀 Commencing multitask training (Target: 90%+ Accuracy)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for imgs, targets_batch in train_loader:
            imgs, targets_batch = imgs.to(device), targets_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation Evaluation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, targets_batch in val_loader:
                imgs, targets_batch = imgs.to(device), targets_batch.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets_batch)
                val_loss += loss.item() * imgs.size(0)
                
                # Apply sigmoid to logits to get predictions
                preds = (torch.sigmoid(outputs) > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets_batch.cpu().numpy())
                
        val_epoch_loss = val_loss / len(val_dataset)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate validation accuracies for each attribute
        val_accs = {}
        total_acc_sum = 0.0
        for i, attr in enumerate(target_attributes):
            acc = accuracy_score(all_labels[:, i], all_preds[:, i]) * 100.0
            val_accs[attr] = acc
            total_acc_sum += acc
            
        avg_val_acc = total_acc_sum / 4.0
        scheduler.step(avg_val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Avg Val Acc: {avg_val_acc:.1f}%")
        acc_str = "  " + " | ".join([f"{attr[:8]}: {val_accs[attr]:.1f}%" for attr in target_attributes])
        print(acc_str)
        
        # Save best model
        if avg_val_acc > best_avg_acc:
            best_avg_acc = avg_val_acc
            print(f"  🌟 New best average validation accuracy: {best_avg_acc:.2f}%! Saving model state dict...")
            torch.save({
                "state_dict": model.state_dict(),
                "attributes": target_attributes,
                "accuracy": best_avg_acc / 100.0,
                "accuracies": {attr: val_accs[attr]/100.0 for attr in target_attributes}
            }, model_output_path)
            
    print(f"\n🎯 Multitask Model Training Complete! Best Average Accuracy: {best_avg_acc:.2f}%")
    
    # Load best model for final report
    print("\n📋 Generating final Classification Report on best model...")
    checkpoint = torch.load(model_output_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, targets_batch in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets_batch.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    for i, attr in enumerate(target_attributes):
        print(f"\n--- Attribute: {attr} ---")
        print(classification_report(all_labels[:, i], all_preds[:, i], target_names=["Negative", "Positive"]))
        
    print("🎉 MULTITASK MODEL TRAINING COMPLETE AND SUCCESSFUL!")
    print("==================================================\n")

if __name__ == "__main__":
    train_multitask_models()
