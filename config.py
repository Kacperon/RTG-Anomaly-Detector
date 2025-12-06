# config.py - Configuration for RTG Anomaly Detector

import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # Model settings
    MODEL_PATH = os.path.join(BASE_DIR, "runs/detect/rtg_anomaly/weights/best.pt")
    BACKUP_MODEL_PATH = "yolov8n.pt"  # Fallback to pretrained model
    
    # Analysis settings
    IMAGE_SIZE = 1280
    CONFIDENCE_THRESHOLD = 0.25
    MAX_DETECTIONS = 50
    
    # Upload settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'bmp', 'png', 'jpg', 'jpeg'}
    
    # Flask settings
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000
    
    # Frontend settings
    FRONTEND_URL = 'http://localhost:3000'
    
    @staticmethod
    def init_app(app):
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
        
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
