# train_yolo.py - Enhanced training for vehicle anomaly detection
from ultralytics import YOLO
import os
import yaml
import shutil
from pathlib import Path

def create_enhanced_data_yaml():
    """Create enhanced data.yaml with better configuration"""
    data_config = {
        'path': './data',
        'train': 'images/train',
        'val': 'images/train',  # Will split during training
        'test': 'images/train',
        
        # Classes for vehicle anomaly detection
        'nc': 1,
        'names': ['anomaly']
    }
    
    with open('data_enhanced.yaml', 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    return 'data_enhanced.yaml'

def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.2):
    """Split dataset into train and validation"""
    import random
    from glob import glob
    
    images_dir = os.path.join(data_dir, 'images', 'train')
    labels_dir = os.path.join(data_dir, 'labels', 'train')
    
    # Create val directories
    val_images_dir = os.path.join(data_dir, 'images', 'val')
    val_labels_dir = os.path.join(data_dir, 'labels', 'val')
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Get all image files
    image_files = glob(os.path.join(images_dir, '*.bmp')) + \
                  glob(os.path.join(images_dir, '*.jpg')) + \
                  glob(os.path.join(images_dir, '*.png'))
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    
    val_images = image_files[split_idx:]
    
    print(f"Moving {len(val_images)} images to validation set...")
    
    for img_path in val_images:
        # Move image
        img_name = os.path.basename(img_path)
        shutil.move(img_path, os.path.join(val_images_dir, img_name))
        
        # Move corresponding label
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(val_labels_dir, label_name))

def train_enhanced(data_yaml='data_enhanced.yaml', epochs=100, imgsz=1280, batch=8, model='yolov8s.pt', device='cpu'):
    """Enhanced training function with better parameters"""
    
    # Check if dataset exists
    if not os.path.exists('data/images/train'):
        print("❌ Dataset not found! Please run data_prep.py first.")
        return
    
    print("🚗 Starting Vehicle Anomaly Detection Training")
    print("=" * 50)
    
    # Create enhanced data config
    data_yaml = create_enhanced_data_yaml()
    print(f"📋 Created enhanced data config: {data_yaml}")
    
    # Split dataset
    try:
        split_dataset('data')
    except Exception as e:
        print(f"⚠️ Dataset split failed (using train set for validation): {e}")
    
    # Use better base model
    print(f"🤖 Loading base model: {model}")
    yolo_model = YOLO(model)
    
    # Enhanced training parameters (cleaned for newer YOLO versions)
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'name': 'vehicle_anomaly',
        'project': 'runs/detect',
        
        # Enhanced parameters for anomaly detection
        'lr0': 0.01,           # Learning rate
        'lrf': 0.1,            # Final learning rate factor
        'momentum': 0.937,     # SGD momentum
        'weight_decay': 0.0005,# Weight decay
        'warmup_epochs': 3.0,  # Warmup epochs
        'warmup_momentum': 0.8,# Warmup momentum
        'warmup_bias_lr': 0.1, # Warmup bias lr
        
        # Data augmentation
        'hsv_h': 0.015,        # HSV-Hue augmentation
        'hsv_s': 0.7,          # HSV-Saturation augmentation  
        'hsv_v': 0.4,          # HSV-Value augmentation
        'degrees': 10.0,       # Rotation degrees
        'translate': 0.1,      # Translation
        'scale': 0.9,          # Scale
        'shear': 2.0,          # Shear degrees
        'perspective': 0.0,    # Perspective
        'flipud': 0.0,         # Vertical flip probability
        'fliplr': 0.5,         # Horizontal flip probability
        'mosaic': 1.0,         # Mosaic probability
        'mixup': 0.1,          # Mixup probability
        'copy_paste': 0.1,     # Copy paste probability
        
        # Validation and saving
        'save': True,          # Save checkpoints
        'save_period': 10,     # Save every N epochs
        'cache': True,         # Cache images for faster training
        'rect': False,         # Rectangular training
        'resume': False,       # Resume training
        'val': True,           # Enable validation
        'exist_ok': False,     # Existing project/name ok
        'patience': 50,        # Early stopping patience
        'freeze': [0],         # Freeze layers (backbone)
        
        # Loss function weights
        'box': 0.05,           # Box loss gain
        'cls': 0.3,            # Class loss gain  
        'dfl': 1.5,            # DFL loss gain
        
        # Confidence and IoU thresholds for validation
        'conf': 0.15,          # Lower confidence threshold
        'iou': 0.45,           # IoU threshold for NMS
    }
    
    print("🔧 Training parameters:")
    for key, value in train_args.items():
        if key in ['lr0', 'lrf', 'conf', 'iou', 'epochs', 'imgsz', 'batch']:
            print(f"   {key}: {value}")
    
    print("\n🚀 Starting training...")
    
    # Train the model
    results = yolo_model.train(**train_args)
    
    print("\n✅ Training completed!")
    print(f"📊 Best weights saved to: runs/detect/vehicle_anomaly/weights/best.pt")
    print(f"📈 Training results: {results}")
    
    # Model validation
    print("\n🔍 Validating model...")
    metrics = yolo_model.val(data=data_yaml, imgsz=imgsz, conf=0.15, iou=0.45)
    print(f"📈 Validation metrics: {metrics}")
    
    return results

if __name__ == "__main__":
    # Enhanced training with better parameters - using CPU to avoid CUDA issues
    results = train_enhanced(
        epochs=25,      # Reduced epochs for CPU training
        imgsz=640,      # Lower resolution for CPU
        batch=2,        # Smaller batch for CPU
        model='yolov8n.pt',  # Lighter model for CPU training
        device='cpu'    # Force CPU usage
    )
    
    print("\n🎉 Training pipeline completed!")
    print("📝 Next steps:")
    print("   1. Check training metrics in runs/detect/vehicle_anomaly/")
    print("   2. Test the model with: python test_api.py")
    print("   3. Run the web interface with: ./start.sh")
