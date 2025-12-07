import cv2
import os
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

model = YOLO('yolov8n.pt') 
clean_folder = r'czyste'
output_folder = r'obrobione_bmp'
os.makedirs(output_folder, exist_ok=True)

def get_all_images(folder_path):
    """Pobierz wszystkie zdjęcia z folderu i podfolderów"""
    image_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))
    return image_files

def expand_bbox(bbox_coords, img_shape, expansion_factor=0.1):
    """
    Powiększa bounding box o określony procent, zachowując jego środek.
    
    bbox_coords: (x1, y1, x2, y2)
    img_shape: (height, width, channels) - wymiary obrazu
    expansion_factor: współczynnik powiększenia (0.1 = 10%)
    """
    H, W = img_shape[:2]
    x1, y1, x2, y2 = bbox_coords
    
    # Oblicz szerokość i wysokość boxa
    width = x2 - x1
    height = y2 - y1
    
    # Oblicz powiększenie
    width_expansion = width * expansion_factor / 2
    height_expansion = height * expansion_factor / 2
    
    # Nowe współrzędne
    new_x1 = max(0, int(x1 - width_expansion))
    new_y1 = max(0, int(y1 - height_expansion))
    new_x2 = min(W, int(x2 + width_expansion))
    new_y2 = min(H, int(y2 + height_expansion))
    
    return (new_x1, new_y1, new_x2, new_y2)

def crop_to_bbox(img, bbox_coords):
    """
    Przycina obraz do obszaru bounding boxa.
    
    img: oryginalny obraz
    bbox_coords: (x1, y1, x2, y2) - współrzędne obszaru do wycięcia
    """
    x1, y1, x2, y2 = bbox_coords
    
    H, W = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    
    # Wytnij obszar
    cropped_img = img[y1:y2, x1:x2]
    
    return cropped_img

def detect_and_display(folder_path, folder_name, output_folder):
    """
    Detectuj obiekty, znajdź największy box, powiększ go o 10%,
    i przytnij obraz do tego obszaru.
    """
    
    image_files = get_all_images(folder_path)
    
    for img_path in image_files:
        print(f"Przetwarzam: {os.path.basename(img_path)}")
        
        results = model.predict(img_path,
                                imgsz=640,
                                conf=0.03,
                                iou=0.3,
                                max_det=300,
                                save=False,
                                device='cpu',
                                verbose=False)
        
        img = cv2.imread(img_path)
        
        largest_box = None
        max_area = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    largest_box = (x1, y1, x2, y2)
                    
        if largest_box is not None:
            # Powiększ bounding box o 10%
            expanded_box = expand_bbox(largest_box, img.shape, expansion_factor=0.1)
            
            # Przytnij obraz do powiększonego bounding boxa
            cropped_img = crop_to_bbox(img, expanded_box)
            
            x1, y1, x2, y2 = expanded_box
            
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_file_path = os.path.join(output_folder, f'{base_name}_cropped.bmp')
            
            cv2.imwrite(output_file_path, cropped_img)
            print(f"  -> Zapisano obrobiony plik: {output_file_path}")
        else:
            print("  -> Nie wykryto żadnych obiektów. Obraz nie został przetworzony.")

# Processuj foldery
print("--- Rozpoczynam przetwarzanie ---")
detect_and_display(clean_folder, 'CZYSTE', output_folder)
print("--- Zakończono przetwarzanie ---")