#!/usr/bin/env python3
# test_api.py - Simple API testing script

import requests
import json
import os
import time

API_BASE = "http://localhost:5000/api"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print("🤖 Testing model loading...")
    try:
        response = requests.post(f"{API_BASE}/load-model", json={})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model loaded: {data['message']}")
            return True
        else:
            print(f"❌ Model loading failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_upload(image_path):
    """Test image upload"""
    print(f"📤 Testing image upload: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE}/upload", files=files)
            
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Upload successful: {data['filename']}")
            return data['file_id']
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

def test_analysis(file_id):
    """Test image analysis"""
    print(f"🔍 Testing analysis for file_id: {file_id}")
    
    try:
        response = requests.post(f"{API_BASE}/analyze", json={'file_id': file_id})
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analysis completed: {data['detection_count']} detections")
            
            # Print detections summary
            if data['detections']:
                print("📊 Detections:")
                for i, det in enumerate(data['detections']):
                    print(f"  {i+1}. Confidence: {det['confidence']:.3f}, Box: {det['bbox']}")
            
            return data
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return None

def main():
    print("🧪 RTG Anomaly Detector API Test")
    print("=" * 40)
    
    # Test 1: Health check
    if not test_health():
        print("❌ Backend is not running. Please start it with: python app.py")
        return
    
    print()
    
    # Test 2: Model loading
    if not test_model_loading():
        print("⚠️ Model loading failed, but continuing tests...")
    
    print()
    
    # Test 3: Find a test image
    test_images = [
        "data/czyste/202511180021/48001F003202511180021.bmp",
        "data/brudne/202511190032/48001F003202511190032.bmp",
        "test_image.bmp",
        "sample.bmp"
    ]
    
    image_path = None
    for path in test_images:
        if os.path.exists(path):
            image_path = path
            break
    
    if not image_path:
        print("⚠️ No test images found. Creating a dummy image...")
        # Create a simple test image
        try:
            from PIL import Image
            import numpy as np
            
            # Create a simple gray image
            img_array = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
            img = Image.fromarray(img_array, 'L')
            image_path = "test_dummy.bmp"
            img.save(image_path)
            print(f"✅ Created dummy image: {image_path}")
        except Exception as e:
            print(f"❌ Could not create dummy image: {e}")
            return
    
    # Test 4: Upload
    file_id = test_upload(image_path)
    if not file_id:
        return
    
    print()
    
    # Test 5: Analysis
    results = test_analysis(file_id)
    if results:
        print(f"🎉 All tests completed successfully!")
        print(f"📊 Final results: {results['detection_count']} anomalies detected")
    
    # Cleanup dummy file
    if image_path == "test_dummy.bmp" and os.path.exists(image_path):
        os.remove(image_path)

if __name__ == "__main__":
    main()
