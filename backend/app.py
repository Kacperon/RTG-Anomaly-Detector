# app.py - Flask backend for Vehicle Scan Anomaly Detector
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import base64
from PIL import Image
import numpy as np
import cv2
import json
from datetime import datetime
import uuid

# Import modelu v1
from modelv2 import detector as model_detector

# Import nowego systemu detekcji anomalii
try:
    from anomaly_detector import RTGAnomalySystem
    ANOMALY_DETECTOR_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTOR_AVAILABLE = False
    print("⚠️ Moduł anomaly_detector niedostępny - używaj starego systemu YOLO")

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = '../data/uploads'
RESULTS_FOLDER = '../data/results'
ANOMALY_REPORTS_FOLDER = '../data/anomaly_reports'
REFERENCE_DIR = "../data/czyste"  # Katalog z obrazami wzorcowymi

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(ANOMALY_REPORTS_FOLDER, exist_ok=True)

# Global variables
anomaly_system = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load YOLO model"""
    try:
        result = model_detector.load_model()
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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
    """Analyze uploaded image for anomalies using model v1"""
    try:
        # Sprawdź czy model jest załadowany
        if not model_detector.is_model_loaded():
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
        
        # Wywołaj detekcję przez model v1
        result = model_detector.detect_anomalies(filepath)
        
        if not result["success"]:
            return jsonify(result), 500
        
        # Przygotuj obrazy dla frontendu
        image = Image.open(filepath).convert('L')
        image_np = np.array(image)
        annotated_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        # Narysuj wykrycia na obrazie
        for detection in result["detections"]:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox
            detection_id = detection["id"]
            color = (0, 0, 255)  # Czerwony
            
            # Narysuj prostokąt
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            # Narysuj markery w rogach
            corner_size = 15
            corner_thickness = 4
            
            # Lewy górny róg
            cv2.line(annotated_image, (x1, y1), (x1 + corner_size, y1), color, corner_thickness)
            cv2.line(annotated_image, (x1, y1), (x1, y1 + corner_size), color, corner_thickness)
            
            # Prawy górny róg
            cv2.line(annotated_image, (x2, y1), (x2 - corner_size, y1), color, corner_thickness)
            cv2.line(annotated_image, (x2, y1), (x2, y1 + corner_size), color, corner_thickness)
            
            # Lewy dolny róg
            cv2.line(annotated_image, (x1, y2), (x1 + corner_size, y2), color, corner_thickness)
            cv2.line(annotated_image, (x1, y2), (x1, y2 - corner_size), color, corner_thickness)
            
            # Prawy dolny róg
            cv2.line(annotated_image, (x2, y2), (x2 - corner_size, y2), color, corner_thickness)
            cv2.line(annotated_image, (x2, y2), (x2, y2 - corner_size), color, corner_thickness)
            
            # Etykieta
            label_text = f"#{detection_id}"
            font_scale = 0.8
            font_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # Tło etykiety
            cv2.rectangle(annotated_image, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), color, -1)
            
            # Tekst etykiety
            cv2.putText(annotated_image, label_text, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Zapisz wynik
        result_filename = f"result_{file_id}.png"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        cv2.imwrite(result_path, annotated_image)
        
        # Konwertuj obrazy do base64
        original_b64 = image_to_base64(image_np)
        annotated_b64 = image_to_base64(annotated_image)
        
        return jsonify({
            "analysis_complete": True,
            "detections": result["detections"],
            "detection_count": result["detection_count"],
            "has_anomaly": result["has_anomaly"],
            "original_image": original_b64,
            "annotated_image": annotated_b64,
            "result_path": result_path,
            "model_info": result["model_info"],
            "timestamp": result["timestamp"]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze-comparison', methods=['POST'])
def analyze_image_comparison():
    """
    Nowy endpoint - Analiza obrazu poprzez porównanie z wzorcami
    Znajduje najbardziej podobny obraz czysty i wykrywa anomalie na podstawie różnic
    """
    global anomaly_system
    
    try:
        # Sprawdź czy system jest dostępny
        if not ANOMALY_DETECTOR_AVAILABLE:
            return jsonify({
                "error": "System detekcji anomalii nie jest dostępny. Zainstaluj wymagane biblioteki (scipy, scikit-image)"
            }), 503
        
        # Inicjalizuj system jeśli nie istnieje
        if anomaly_system is None:
            if not os.path.exists(REFERENCE_DIR):
                return jsonify({
                    "error": f"Katalog z obrazami wzorcowymi nie istnieje: {REFERENCE_DIR}"
                }), 404
            
            try:
                anomaly_system = RTGAnomalySystem(REFERENCE_DIR, ANOMALY_REPORTS_FOLDER)
            except Exception as e:
                return jsonify({
                    "error": f"Nie można zainicjalizować systemu detekcji: {str(e)}"
                }), 500
        
        # Pobierz dane z requestu
        data = request.get_json()
        file_id = data.get('file_id')
        use_alignment = data.get('use_alignment', True)
        use_ssim = data.get('use_ssim', True)
        
        if not file_id:
            return jsonify({"error": "No file_id provided"}), 400
        
        # Znajdź plik
        uploaded_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(file_id)]
        if not uploaded_files:
            return jsonify({"error": "File not found"}), 404
        
        filepath = os.path.join(UPLOAD_FOLDER, uploaded_files[0])
        
        # Przetwórz obraz
        result = anomaly_system.process_image(
            filepath,
            use_alignment=use_alignment,
            use_ssim=use_ssim,
            save_report=True
        )
        
        # Wczytaj raport jako base64
        report_b64 = None
        if result.get('report_path') and os.path.exists(result['report_path']):
            with open(result['report_path'], 'rb') as f:
                report_b64 = base64.b64encode(f.read()).decode()
        
        # Przygotuj szczegółowe informacje o anomaliach
        anomalies_detail = []
        for i, anomaly in enumerate(result.get('anomalies', []), 1):
            x, y, w, h = anomaly['bbox']
            anomalies_detail.append({
                "id": i,
                "bbox": [x, y, x+w, y+h],  # Convert to [x1, y1, x2, y2]
                "area": int(anomaly['area']),
                "solidity": round(anomaly['solidity'], 3),
                "aspect_ratio": round(anomaly['aspect_ratio'], 3),
                "center": [x + w//2, y + h//2]
            })
        
        return jsonify({
            "method": "comparison_based",
            "analysis_complete": True,
            "has_anomaly": result['has_anomaly'],
            "anomaly_count": result['anomaly_count'],
            "anomalies": anomalies_detail,
            "reference_match": result['reference_match'],
            "similarity": round(result['similarity'], 4),
            "ssim_score": round(result['ssim_score'], 4) if result.get('ssim_score') else None,
            "report_image": report_b64,
            "report_path": result.get('report_path'),
            "settings": {
                "alignment_used": use_alignment,
                "ssim_used": use_ssim
            },
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Endpoint do przetwarzania wielu obrazów naraz
    """
    global anomaly_system
    
    try:
        if not ANOMALY_DETECTOR_AVAILABLE:
            return jsonify({"error": "System detekcji anomalii niedostępny"}), 503
        
        # Inicjalizuj system
        if anomaly_system is None:
            if not os.path.exists(REFERENCE_DIR):
                return jsonify({"error": f"Katalog wzorcowy nie istnieje: {REFERENCE_DIR}"}), 404
            anomaly_system = RTGAnomalySystem(REFERENCE_DIR, ANOMALY_REPORTS_FOLDER)
        
        # Pobierz parametry
        data = request.get_json()
        directory = data.get('directory', UPLOAD_FOLDER)
        pattern = data.get('pattern', '*.bmp')
        
        if not os.path.exists(directory):
            return jsonify({"error": f"Katalog nie istnieje: {directory}"}), 404
        
        # Przetwarzaj partię
        results = anomaly_system.batch_process(directory, pattern)
        
        # Podsumowanie
        anomaly_count = sum(1 for r in results if r.get('has_anomaly', False))
        
        return jsonify({
            "batch_complete": True,
            "total_processed": len(results),
            "anomalies_found": anomaly_count,
            "clean_images": len(results) - anomaly_count,
            "results": results[:50],  # Ogranicz do 50 dla wydajności
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/detector-status', methods=['GET'])
def detector_status():
    """
    Sprawdź status systemu detekcji
    """
    return jsonify({
        "yolo_model_loaded": model_detector.is_model_loaded(),
        "comparison_detector_available": ANOMALY_DETECTOR_AVAILABLE,
        "comparison_detector_initialized": anomaly_system is not None,
        "reference_dir": REFERENCE_DIR,
        "reference_dir_exists": os.path.exists(REFERENCE_DIR),
        "model_info": model_detector.get_model_info(),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/download-report/<file_id>', methods=['GET'])
def download_report(file_id):
    """Download analysis report as JSON"""
    try:
        # Implementation for generating and downloading reports
        model_info = model_detector.get_model_info()
        
        report_data = {
            "file_id": file_id,
            "analysis_date": datetime.now().isoformat(),
            "model_used": model_info.get("model_path", "unknown"),
            "model_type": model_info.get("model_type", "unknown"),
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
