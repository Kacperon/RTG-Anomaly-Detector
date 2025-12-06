#!/usr/bin/env python3
"""
Prepare dataset from manually annotated images.
This script uses the manually annotated bounding boxes from annotation_tool.py
and adds clean images with empty labels for better training.
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

def verify_manual_annotations(data_dir, output_images_dir, output_labels_dir):
    """
    Verify manually annotated images and labels from annotation_tool output.
    These are the dirty images with manually marked anomalies.
    """
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)
    
    print("📦 Verifying manually annotated images...")
    
    # Check if directories exist
    if not output_images_dir.exists() or not output_labels_dir.exists():
        print(f"⚠️ Manual annotations not found in {output_images_dir}")
        print("   Please run annotation_tool.py first to create annotations")
        return 0
    
    # Get all manually annotated images
    annotated_images = list(output_images_dir.glob("brudne_*.bmp"))
    
    print(f"Found {len(annotated_images)} manually annotated dirty images")
    
    # Verify each has a corresponding label
    missing_labels = []
    for img_path in tqdm(annotated_images):
        label_name = img_path.stem + ".txt"
        label_path = output_labels_dir / label_name
        
        if not label_path.exists():
            # Create empty label if missing
            label_path.touch()
            missing_labels.append(img_path.name)
    
    if missing_labels:
        print(f"⚠️ Created {len(missing_labels)} missing label files")
            
    return len(annotated_images)

def add_clean_images(data_dir, output_images_dir, output_labels_dir):
    """
    Add clean images (without anomalies) with empty label files.
    This helps the model learn what normal images look like.
    """
    data_dir = Path(data_dir)
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)
    
    clean_dir = data_dir / "czyste"
    
    if not clean_dir.exists():
        print(f"⚠️ Clean directory not found: {clean_dir}")
        return 0
    
    print("📦 Adding clean images (no anomalies)...")
    
    clean_count = 0
    for folder in tqdm(sorted(clean_dir.iterdir())):
        if not folder.is_dir():
            continue
            
        # Find the main .bmp file (without "czarno" in name)
        files = [f for f in folder.glob("*.bmp") if "czarno" not in f.name]
        if not files:
            continue
            
        img_path = files[0]
        
        try:
            # Load and save clean image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Save with clean_ prefix
            output_name = f"clean_{folder.name}.bmp"
            output_img_path = output_images_dir / output_name
            output_label_path = output_labels_dir / f"clean_{folder.name}.txt"
            
            cv2.imwrite(str(output_img_path), img)
            
            # Create empty label file (no anomalies)
            output_label_path.touch()
            
            clean_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Added {clean_count} clean images")
    return clean_count

def verify_dataset(output_images_dir, output_labels_dir):
    """Verify that all images have corresponding labels"""
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)
    
    images = list(output_images_dir.glob("*.bmp")) + \
             list(output_images_dir.glob("*.jpg")) + \
             list(output_images_dir.glob("*.png"))
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images: {len(images)}")
    
    dirty_count = len([img for img in images if "brudne" in img.name])
    clean_count = len([img for img in images if "clean" in img.name])
    
    print(f"   Dirty images (with anomalies): {dirty_count}")
    print(f"   Clean images (no anomalies): {clean_count}")
    
    # Check labels
    missing_labels = []
    empty_labels = 0
    non_empty_labels = 0
    
    for img_path in images:
        label_path = output_labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            missing_labels.append(img_path.name)
        else:
            # Check if label is empty or has boxes
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if content:
                    non_empty_labels += 1
                else:
                    empty_labels += 1
    
    print(f"   Labels with anomalies: {non_empty_labels}")
    print(f"   Labels without anomalies: {empty_labels}")
    
    if missing_labels:
        print(f"\n⚠️ Warning: {len(missing_labels)} images missing labels:")
        for name in missing_labels[:10]:
            print(f"      - {name}")
        if len(missing_labels) > 10:
            print(f"      ... and {len(missing_labels) - 10} more")
    else:
        print("✅ All images have corresponding labels")
    
    return len(images), dirty_count, clean_count

def main():
    script_dir = Path(__file__).parent
    
    # Paths
    data_dir = script_dir / "data"
    output_images_dir = script_dir / "data" / "images" / "train"
    output_labels_dir = script_dir / "data" / "labels" / "train"
    
    # Ensure output directories exist
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Preparing dataset from manual annotations...")
    print("=" * 60)
    
    # Step 1: Verify manually annotated dirty images (already in place)
    dirty_count = verify_manual_annotations(data_dir, output_images_dir, output_labels_dir)
    
    if dirty_count == 0:
        print("\n❌ No manual annotations found!")
        print("   Please run: python annotation_tool.py")
        print("   And annotate at least some dirty images.")
        return
    
    # Step 2: Add clean images
    clean_count = add_clean_images(data_dir, output_images_dir, output_labels_dir)
    
    # Step 3: Verify dataset
    total, dirty, clean = verify_dataset(output_images_dir, output_labels_dir)
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation completed!")
    print(f"📂 Images saved to: {output_images_dir}")
    print(f"📂 Labels saved to: {output_labels_dir}")
    print(f"\n📊 Final dataset: {total} images ({dirty} dirty + {clean} clean)")
    print("\n🚀 Next steps:")
    print("   1. Run: python train_yolo.py")
    print("   2. Wait for training to complete")
    print("   3. Test the model with: python test_api.py")

if __name__ == "__main__":
    main()
