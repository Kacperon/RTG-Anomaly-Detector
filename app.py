# app.py - Flask backend for Vehicle Scan Anomaly Detector
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from ultralytics import YOLO
import json
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_PATH = "runs/detect/vehicle_anomaly3/weights/best.pt"  # Updated to latest training
FALLBACK_MODEL = "yolov8n.pt"

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global model variable
model = None

def draw_enhanced_bbox(image, box, confidence, label="anomaly", color=(0, 0, 255)):
    """Draw enhanced bounding box with better visibility"""
    x1, y1, x2, y2 = map(int, box)
    
    # Draw main rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    
    # Draw corner markers for better visibility
    corner_size = 15
    corner_thickness = 4
    
    # Top-left corner
    cv2.line(image, (x1, y1), (x1 + corner_size, y1), color, corner_thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_size), color, corner_thickness)
    
    # Top-right corner
    cv2.line(image, (x2, y1), (x2 - corner_size, y1), color, corner_thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_size), color, corner_thickness)
    
    # Bottom-left corner
    cv2.line(image, (x1, y2), (x1 + corner_size, y2), color, corner_thickness)
    cv2.line(image, (x1, y2), (x1, y2 - corner_size), color, corner_thickness)
    
    # Bottom-right corner
    cv2.line(image, (x2, y2), (x2 - corner_size, y2), color, corner_thickness)
    cv2.line(image, (x2, y2), (x2, y2 - corner_size), color, corner_thickness)
    
    # Label background
    label_text = f"{label}: {confidence:.2f}"
    font_scale = 0.8
    font_thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    # Draw label background
    cv2.rectangle(image, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), color, -1)
    
    # Draw label text
    cv2.putText(image, label_text, (x1 + 5, y1 - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    return image

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load YOLO model with fallback"""
    global model
    try:
        data = request.get_json() if request.get_json() else {}
        model_path = data.get('model_path', MODEL_PATH)
        
        # Try to load custom trained model first
        if os.path.exists(model_path):
            model = YOLO(model_path)
            model.to('cpu')  # Force CPU usage
            return jsonify({
                "message": f"Custom model loaded successfully on CPU: {model_path}",
                "model_path": model_path,
                "model_type": "custom_trained",
                "device": "cpu",
                "timestamp": datetime.now().isoformat()
            })
        else:
            # Fallback to pre-trained model
            model = YOLO(FALLBACK_MODEL)
            model.to('cpu')  # Force CPU usage
            return jsonify({
                "message": f"Fallback model loaded on CPU: {FALLBACK_MODEL}",
                "model_path": FALLBACK_MODEL,
                "model_type": "pretrained",
                "device": "cpu",
                "warning": f"Custom model not found at {model_path}",
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload and analyze RTG image"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save uploaded file
        file.save(filepath)
        
        # Return file info
        return jsonify({
            "message": "File uploaded successfully",
            "file_id": file_id,
            "filename": filename,
            "filepath": filepath
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for anomalies"""
    global model
    
    try:
        if model is None:
            return jsonify({"error": "Model not loaded. Please load a model first."}), 400
        
        data = request.get_json()
        file_id = data.get('file_id')
        
        if not file_id:
            return jsonify({"error": "No file_id provided"}), 400
        
        # Find the uploaded file
        uploaded_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(file_id)]
        if not uploaded_files:
            return jsonify({"error": "File not found"}), 404
        
        filepath = os.path.join(UPLOAD_FOLDER, uploaded_files[0])
        
        # Load image
        image = Image.open(filepath).convert('L')
        image_np = np.array(image)
        
        # Run YOLO prediction with very low threshold to detect anomalies
        results = model.predict(
            filepath, 
            imgsz=640,   # Lower resolution for better performance
            conf=0.05,   # Very low confidence threshold - adjust based on results
            iou=0.3,     # Lower IoU to get more overlapping detections
            max_det=300, # Allow more detections
            save=False,
            device='cpu',  # Force CPU usage
            verbose=False
        )
        
        # Process results
        detections = []
        annotated_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for i, box in enumerate(results[0].boxes):
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                
                # Get class name
                class_name = model.names[cls] if hasattr(model, 'names') and cls in model.names else "anomaly"
                
                # Color coding based on confidence
                if conf > 0.7:
                    color = (0, 0, 255)  # Red - High confidence
                elif conf > 0.4:
                    color = (0, 165, 255)  # Orange - Medium confidence  
                else:
                    color = (0, 255, 255)  # Yellow - Low confidence
                
                # Add detection to list
                detections.append({
                    "id": i + 1,
                    "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "confidence": round(conf, 3),
                    "class": class_name,
                    "class_id": cls,
                    "area": int((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])),
                    "center": [int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)]
                })
                
                # Draw enhanced bounding box
                annotated_image = draw_enhanced_bbox(
                    annotated_image, xyxy, conf, class_name, color
                )
        
        # Save annotated image
        result_filename = f"result_{file_id}.png"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        cv2.imwrite(result_path, annotated_image)
        
        # Convert images to base64 for frontend
        original_b64 = image_to_base64(image_np)
        annotated_b64 = image_to_base64(annotated_image)
        
        return jsonify({
            "analysis_complete": True,
            "detections": detections,
            "detection_count": len(detections),
            "original_image": original_b64,
            "annotated_image": annotated_b64,
            "result_path": result_path,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/download-report/<file_id>', methods=['GET'])
def download_report(file_id):
    """Download analysis report as JSON"""
    try:
        # Implementation for generating and downloading reports
        report_data = {
            "file_id": file_id,
            "analysis_date": datetime.now().isoformat(),
            "model_used": MODEL_PATH,
            "status": "Analysis completed"
        }
        
        report_filename = f"report_{file_id}.json"
        report_path = os.path.join(RESULTS_FOLDER, report_filename)
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return send_file(report_path, as_attachment=True)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def image_to_base64(image_np):
    """Convert numpy image to base64 string"""
    if len(image_np.shape) == 3:
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    else:
        image_pil = Image.fromarray(image_np)
    
    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode()

if __name__ == '__main__':
    print("Starting RTG Anomaly Detector Backend...")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Results folder: {os.path.abspath(RESULTS_FOLDER)}")
    app.run(debug=True, port=5000)
