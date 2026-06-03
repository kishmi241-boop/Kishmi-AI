import os
import zipfile
import io
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ensure scripts folder and root folder are in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from scripts.face_cropper import FaceCropper

# Define Focal Loss for severe class imbalance
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Define the PyTorch Dataset
class AcneDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: List of PIL Images (pre-cropped).
            labels: List of integer labels.
            transform: PyTorch transform pipeline.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def train_acne_model():
    zip_path = r"d:\Kishmi\archive.zip"
    model_output_path = r"d:\Kishmi\scripts\aura_acne_model.pth"
    
    print("==================================================")
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
        
    print("🚀 STARTING AURA ACNE DEEP LEARNING MODEL TRAINING")
    print("==================================================")
    
    if not os.path.exists(zip_path):
        print(f"❌ Error: Dataset archive not found at {zip_path}")
        sys.exit(1)
        
    # Mapping of zip directory prefixes to integer target labels
    label_mapping = {
        "acne_1024/acne0_1024": 0,  # Clear / Healthy
        "acne_1024/acne1_1024": 1,  # Mild
        "acne_1024/acne2_1024": 2,  # Moderate
        "acne_1024/acne3_1024": 3   # Severe / Active Hormonal
    }
    
    raw_images = []
    labels = []
    
    print("📦 Reading ZIP archive and cropping faces with MediaPipe...")
    cropper = FaceCropper(min_detection_confidence=0.4)
    
    detected_count = 0
    fallback_count = 0
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        infos = zip_ref.infolist()
        
        # Filter files belonging to our target labels
        target_infos = []
        for info in infos:
            if info.is_dir():
                continue
            filename = info.filename
            for prefix in label_mapping.keys():
                if filename.startswith(prefix) and filename.endswith(('.jpg', '.jpeg', '.png')):
                    target_infos.append((info, label_mapping[prefix]))
                    break
                    
        total_files = len(target_infos)
        print(f"Total target files found: {total_files}")
        
        processed_count = 0
        for info, label in target_infos:
            try:
                img_bytes = zip_ref.read(info)
                # Crop face using MediaPipe cropper (resize to 224x224)
                cropped_img, was_detected = cropper.crop_face(img_bytes, target_size=(224, 224), margin_ratio=0.25)
                
                raw_images.append(cropped_img)
                labels.append(label)
                
                if was_detected:
                    detected_count += 1
                else:
                    fallback_count += 1
                    
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count} / {total_files} images (Detected: {detected_count}, Fallback: {fallback_count})...")
            except Exception as e:
                print(f"  Failed reading/cropping {info.filename}: {e}")
                
    cropper.close()
    print(f"\n✅ Finished processing {processed_count} images successfully!")
    print(f"  -> Face detection success: {detected_count} ({detected_count/max(1, processed_count)*100:.1f}%)")
    print(f"  -> Fallback center crop: {fallback_count} ({fallback_count/max(1, processed_count)*100:.1f}%)")
    
    y = np.array(labels)
    
    # Class distribution check
    unique, counts = np.unique(y, return_counts=True)
    print("\n📊 Dataset Class Distribution:")
    class_names = ["Level 0 (Clear)", "Level 1 (Mild)", "Level 2 (Moderate)", "Level 3 (Severe)"]
    for val, count in zip(unique, counts):
        print(f"  - {class_names[val]}: {count} samples")
        
    # Split into train & validation sets (80/20 split)
    indices = np.arange(len(raw_images))
    train_idx, val_idx, y_train, y_val = train_test_split(
        indices, y, test_size=0.20, random_state=42, stratify=y
    )
    
    train_images = [raw_images[i] for i in train_idx]
    val_images = [raw_images[i] for i in val_idx]
    
    # Define Data Transforms (Heavy Data Augmentation for Training)
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
    
    # Datasets and Loaders
    train_dataset = AcneDataset(train_images, y_train, transform=train_transforms)
    val_dataset = AcneDataset(val_images, y_val, transform=val_transforms)
    
    # CPU optimized training - set num_workers=0 to avoid multiprocessing overhead on Windows
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
            
    # Replace Classification Head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 4)  # 4 classes
    )
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🏋️ Training on device: {device}")
    
    # Setup Focal Loss with balanced class weights to handle imbalance
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    weights = total_samples / (len(class_counts) * class_counts)
    weights_tensor = torch.FloatTensor(weights).to(device)
    print(f"⚖️ Applied Class Weights: {weights}")
    
    criterion = FocalLoss(weight=weights_tensor, gamma=2.0)
    
    # Optimizer and Learning Rate Scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training Loop
    epochs = 25
    best_acc = 0.0
    
    print("\n🚀 Commencing model training (Target: 90%+ Accuracy)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += lbls.size(0)
            correct_train += (predicted == lbls).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_train / total_train * 100.0
        
        # Validation evaluation
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total_val += lbls.size(0)
                correct_val += (predicted == lbls).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_acc = correct_val / total_val * 100.0
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1f}% | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"  🌟 New best validation accuracy: {best_acc:.2f}%! Saving model state dict...")
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "accuracy": best_acc / 100.0
            }, model_output_path)
            
    print(f"\n🎯 Model Training Complete! Best Validation Accuracy: {best_acc:.2f}%")
    
    # Load best model for final evaluation report
    print("\n📋 Generating final Classification Report on best model...")
    checkpoint = torch.load(model_output_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(lbls.numpy())
            
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("🎉 ACNE MODEL TRAINING COMPLETE AND SUCCESSFUL!")
    print("==================================================\n")

if __name__ == "__main__":
    train_acne_model()
