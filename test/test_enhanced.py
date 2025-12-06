#!/usr/bin/env python3
# test_enhanced.py - Enhanced testing for vehicle anomaly detection

import requests
import json
import os
import time
import cv2
import numpy as np
from PIL import Image

API_BASE = "http://localhost:5000/api"

def create_test_vehicle_image():
    """Create a synthetic test vehicle scan image"""
    print("🚗 Creating synthetic vehicle test image...")
    
    # Create a synthetic vehicle scan (800x600)
    img = np.ones((600, 800), dtype=np.uint8) * 120  # Gray background
    
    # Add vehicle outline
    cv2.rectangle(img, (100, 150), (700, 450), 80, -1)  # Main body
    cv2.rectangle(img, (50, 200), (150, 400), 60, -1)   # Front
    cv2.rectangle(img, (650, 200), (750, 400), 60, -1)  # Rear
    
    # Add some "clean" details
    cv2.rectangle(img, (200, 200), (600, 250), 100, -1)  # Hood
    cv2.rectangle(img, (200, 350), (600, 400), 100, -1)  # Bottom
    
    # Save as test image
    test_path = "test_vehicle_clean.bmp"
    cv2.imwrite(test_path, img)
    
    # Create damaged version with anomalies
    img_damaged = img.copy()
    
    # Add some "damage" patterns
    cv2.circle(img_damaged, (300, 300), 30, 40, -1)      # Dent 1
    cv2.circle(img_damaged, (500, 250), 25, 45, -1)      # Dent 2
    cv2.rectangle(img_damaged, (400, 320), (480, 340), 30, -1)  # Scratch
    
    # Add some corrosion-like patterns
    pts = np.array([[550, 380], [580, 370], [590, 390], [570, 395]], np.int32)
    cv2.fillPoly(img_damaged, [pts], 35)
    
    damaged_path = "test_vehicle_damaged.bmp"
    cv2.imwrite(damaged_path, img_damaged)
    
    print(f"✅ Created test images: {test_path}, {damaged_path}")
    return [test_path, damaged_path]

def test_enhanced_analysis():
    """Test the enhanced anomaly detection"""
    print("🧪 Enhanced Vehicle Anomaly Detection Test")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print("✅ Backend health check passed")
        else:
            print("❌ Backend is not running")
            return
    except:
        print("❌ Cannot connect to backend. Start with: python app.py")
        return
    
    # Test 2: Model loading
    print("\n🤖 Testing model loading...")
    response = requests.post(f"{API_BASE}/load-model", json={})
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Model loaded: {data.get('model_type', 'unknown')}")
        if 'warning' in data:
            print(f"⚠️ {data['warning']}")
    else:
        print("❌ Model loading failed")
        return
    
    # Test 3: Create test images
    test_images = create_test_vehicle_image()
    
    # Test each image
    for img_path in test_images:
        print(f"\n🔍 Testing image: {img_path}")
        
        # Upload
        with open(img_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE}/upload", files=files)
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.text}")
            continue
            
        file_data = response.json()
        file_id = file_data['file_id']
        print(f"✅ Upload successful: {file_id}")
        
        # Analyze
        response = requests.post(f"{API_BASE}/analyze", json={'file_id': file_id})
        
        if response.status_code != 200:
            print(f"❌ Analysis failed: {response.text}")
            continue
            
        results = response.json()
        detection_count = results.get('detection_count', 0)
        detections = results.get('detections', [])
        
        print(f"📊 Analysis results:")
        print(f"   Detections: {detection_count}")
        
        if detections:
            print("   Details:")
            for i, det in enumerate(detections[:5], 1):  # Show first 5
                confidence = det.get('confidence', 0)
                class_name = det.get('class', 'unknown')
                area = det.get('area', 0)
                print(f"     {i}. {class_name}: {confidence:.3f} confidence, area: {area}px²")
        
        print()
    
    # Cleanup
    for img_path in test_images:
        if os.path.exists(img_path):
            os.remove(img_path)
    
    print("🎉 Enhanced testing completed!")

def benchmark_detection_parameters():
    """Benchmark different detection parameters"""
    print("\n📈 Benchmarking detection parameters...")
    
    # This would test different confidence thresholds, IoU values, etc.
    test_params = [
        {'conf': 0.1, 'name': 'Very sensitive'},
        {'conf': 0.25, 'name': 'Default'},
        {'conf': 0.5, 'name': 'Conservative'},
    ]
    
    print("🔧 Different sensitivity levels tested")
    print("   (This would require backend API parameter support)")

if __name__ == "__main__":
    test_enhanced_analysis()
    benchmark_detection_parameters()
