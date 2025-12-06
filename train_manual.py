#!/usr/bin/env python3
"""
Train YOLO model on manually annotated dataset.
This script uses the manually annotated bounding boxes for better accuracy.
"""

from ultralytics import YOLO
import os
import yaml
import shutil
from pathlib import Path
from glob import glob
import random

def verify_dataset():
    """Verify that dataset exists and has proper structure"""
    data_dir = Path("data")
    images_train = data_dir / "images" / "train"
    labels_train = data_dir / "labels" / "train"
    
    if not images_train.exists():
        print("❌ Training images directory not found!")
        print(f"   Expected: {images_train}")
        return False
        
    if not labels_train.exists():
        print("❌ Training labels directory not found!")
        print(f"   Expected: {labels_train}")
        return False
    
    # Count images and labels
    images = list(images_train.glob("*.bmp")) + \
             list(images_train.glob("*.jpg")) + \
             list(images_train.glob("*.png"))
    labels = list(labels_train.glob("*.txt"))
    
    print(f"📊 Found {len(images)} images and {len(labels)} labels")
    
    if len(images) == 0:
        print("❌ No images found!")
        print("   Please run: python prepare_manual_dataset.py")
        return False
        
    if len(labels) == 0:
        print("❌ No labels found!")
        print("   Please run: python annotation_tool.py to create annotations")
        return False
    
    # Count images with annotations
    annotated_count = 0
    empty_count = 0
    
    for label_path in labels:
        with open(label_path, 'r') as f:
            content = f.read().strip()
            if content:
                annotated_count += 1
            else:
                empty_count += 1
    
    print(f"   Images with anomalies: {annotated_count}")
    print(f"   Clean images (no anomalies): {empty_count}")
    
    if annotated_count == 0:
        print("⚠️ Warning: No images with anomaly annotations!")
        print("   The model will only learn clean images.")
    
    return True

def split_dataset(train_ratio=0.8):
    """Split dataset into train and validation sets"""
    data_dir = Path("data")
    images_train = data_dir / "images" / "train"
    labels_train = data_dir / "labels" / "train"
    images_val = data_dir / "images" / "val"
    labels_val = data_dir / "labels" / "val"
    
    # Create validation directories
    images_val.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    all_images = list(images_train.glob("*.bmp")) + \
                 list(images_train.glob("*.jpg")) + \
                 list(images_train.glob("*.png"))
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * train_ratio)
    val_images = all_images[split_idx:]
    
    print(f"📂 Splitting dataset: {len(all_images) - len(val_images)} train, {len(val_images)} val")
    
    # Move validation images and labels
    for img_path in val_images:
        # Move image
        dest_img = images_val / img_path.name
        if not dest_img.exists():
            shutil.move(str(img_path), str(dest_img))
        
        # Move label
        label_name = img_path.stem + ".txt"
        label_path = labels_train / label_name
        dest_label = labels_val / label_name
        
        if label_path.exists() and not dest_label.exists():
            shutil.move(str(label_path), str(dest_label))

def train_on_manual_annotations(
    epochs=50,
    imgsz=640,
    batch=4,
    model='yolov8n.pt',
    device='cpu',
    patience=20
):
    """
    Train YOLO model on manually annotated dataset.
    
    Args:
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        model: Base model to use (yolov8n.pt, yolov8s.pt, etc.)
        device: Device to use ('cpu' or 'cuda')
        patience: Early stopping patience
    """
    
    print("🚗 RTG Anomaly Detection - Manual Annotation Training")
    print("=" * 70)
    
    # Step 1: Verify dataset
    if not verify_dataset():
        print("\n❌ Dataset verification failed!")
        print("   Run: python prepare_manual_dataset.py")
        return None
    
    # Step 2: Split dataset
    try:
        split_dataset(train_ratio=0.85)
    except Exception as e:
        print(f"⚠️ Dataset split warning: {e}")
    
    # Step 3: Load model
    print(f"\n🤖 Loading base model: {model}")
    yolo_model = YOLO(model)
    
    # Step 4: Configure training parameters
    data_yaml = 'data.yaml'
    
    train_args = {
        # Dataset
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        
        # Project settings
        'name': 'manual_annotations',
        'project': 'runs/detect',
        'exist_ok': True,
        'resume': False,
        
        # Optimization
        'lr0': 0.01,              # Initial learning rate
        'lrf': 0.1,               # Final learning rate (lr0 * lrf)
        'momentum': 0.937,        # SGD momentum
        'weight_decay': 0.0005,   # Optimizer weight decay
        'warmup_epochs': 3.0,     # Warmup epochs
        'warmup_momentum': 0.8,   # Warmup initial momentum
        'warmup_bias_lr': 0.1,    # Warmup initial bias lr
        
        # Data augmentation (moderate for X-ray images)
        'hsv_h': 0.01,            # HSV-Hue augmentation (fraction)
        'hsv_s': 0.3,             # HSV-Saturation augmentation (fraction)
        'hsv_v': 0.2,             # HSV-Value augmentation (fraction)
        'degrees': 5.0,           # Image rotation (+/- deg)
        'translate': 0.1,         # Image translation (+/- fraction)
        'scale': 0.5,             # Image scale (+/- gain)
        'shear': 2.0,             # Image shear (+/- deg)
        'perspective': 0.0,       # Image perspective (+/- fraction)
        'flipud': 0.0,            # Vertical flip (disabled for X-rays)
        'fliplr': 0.5,            # Horizontal flip probability
        'mosaic': 0.5,            # Mosaic augmentation probability
        'mixup': 0.05,            # Mixup augmentation probability
        
        # Loss weights
        'box': 7.5,               # Box loss gain (higher for small objects)
        'cls': 0.5,               # Class loss gain (single class)
        'dfl': 1.5,               # DFL loss gain
        
        # Validation
        'val': True,              # Validate during training
        'save': True,             # Save checkpoints
        'save_period': 10,        # Save checkpoint every N epochs
        'cache': False,           # Cache images (disable for large datasets)
        'patience': patience,     # Early stopping patience
        
        # Inference settings for validation
        'conf': 0.10,             # Confidence threshold for validation
        'iou': 0.45,              # IoU threshold for NMS
        
        # Other
        'verbose': True,          # Verbose output
        'plots': True,            # Save plots
    }
    
    print("\n🔧 Training Configuration:")
    print(f"   Model: {model}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch}")
    print(f"   Device: {device}")
    print(f"   Learning rate: {train_args['lr0']} -> {train_args['lr0'] * train_args['lrf']}")
    print(f"   Early stopping patience: {patience}")
    
    # Step 5: Train
    print("\n🚀 Starting training...")
    print("-" * 70)
    
    try:
        results = yolo_model.train(**train_args)
        
        print("\n" + "=" * 70)
        print("✅ Training completed successfully!")
        print(f"📊 Best weights: runs/detect/manual_annotations/weights/best.pt")
        print(f"📊 Last weights: runs/detect/manual_annotations/weights/last.pt")
        
        # Step 6: Validate
        print("\n🔍 Running final validation...")
        metrics = yolo_model.val(
            data=data_yaml,
            imgsz=imgsz,
            conf=0.10,
            iou=0.45,
            device=device
        )
        
        print(f"\n📈 Validation Metrics:")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main training function"""
    
    # Check if dataset is prepared
    if not Path("data/images/train").exists():
        print("❌ Dataset not prepared!")
        print("   Run: python prepare_manual_dataset.py")
        return
    
    # Train with optimized parameters for CPU
    print("Starting training with manual annotations...")
    print("This may take a while depending on your hardware.\n")
    
    results = train_on_manual_annotations(
        epochs=50,              # 50 epochs should be enough for small dataset
        imgsz=640,              # Standard size, good for CPU
        batch=2,                # Small batch for CPU
        model='yolov8n.pt',     # Nano model - fastest for CPU
        device='cpu',           # Use CPU
        patience=15             # Early stopping after 15 epochs without improvement
    )
    
    if results:
        print("\n" + "=" * 70)
        print("🎉 Training completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Review training plots: runs/detect/manual_annotations/")
        print("   2. Test the model: python test_model.py")
        print("   3. Run inference: python test_api.py")
        print("   4. Start web interface: ./start.sh or python app.py")
        print("=" * 70)
    else:
        print("\n❌ Training failed. Please check the errors above.")

if __name__ == "__main__":
    main()
