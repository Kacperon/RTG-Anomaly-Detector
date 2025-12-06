# data_prep.py
import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

def load_gray(path):
    """Load image as grayscale with error handling"""
    try:
        return np.array(Image.open(path).convert('L'), dtype=np.uint8)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def compute_diff_mask(template, sample, blur=7, thresh=25, min_area=300, max_area=50000):
    # template, sample: numpy uint8 HxW
    diff = cv2.absdiff(template, sample)
    
    # Enhanced preprocessing for vehicle scans
    # Apply histogram equalization for better contrast
    template_eq = cv2.equalizeHist(template)
    sample_eq = cv2.equalizeHist(sample)
    diff_eq = cv2.absdiff(template_eq, sample_eq)
    
    # Combine both difference methods
    diff_combined = cv2.addWeighted(diff, 0.7, diff_eq, 0.3, 0)
    
    # Gaussian blur to reduce noise
    diff_blurred = cv2.GaussianBlur(diff_combined, (blur, blur), 0)
    
    # Adaptive threshold for better anomaly detection
    _, bw = cv2.threshold(diff_blurred, thresh, 255, cv2.THRESH_BINARY)
    
    # Morphological operations for better shapes
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Opening to remove noise
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_open, iterations=2)
    # Closing to fill gaps
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Find contours and filter by area
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bw)
    boxes = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
            
        # Additional filtering based on shape
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.3:  # Filter out very irregular shapes
                continue
        
        x, y, w, h = cv2.boundingRect(c)
        
        # Filter based on aspect ratio (avoid very thin lines)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            continue
            
        boxes.append((x, y, w, h))
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        
    return mask, boxes, diff_combined

def bbox_to_yolo(box, img_w, img_h, cls=0):
    x, y, w, h = box
    xc = (x + w/2) / img_w
    yc = (y + h/2) / img_h
    return f"{cls} {xc:.6f} {yc:.6f} {w/img_w:.6f} {h/img_h:.6f}"

def prepare_dataset_from_clean_dirty(data_dir, out_images_dir, out_labels_dir):
    """
    data_dir structure:
      data_dir/
        czyste/
          202511180021/
            48001F003202511180021.bmp (main image)
            48001F003202511180021 czarno.bmp (black version)
        brudne/
          202511190032/
            48001F003202511190032.bmp (main image)
            48001F003202511190032 czarno.bmp (black version)
    """
    # Ensure absolute paths
    data_dir = os.path.abspath(data_dir)
    out_images_dir = os.path.abspath(out_images_dir)
    out_labels_dir = os.path.abspath(out_labels_dir)
    
    print(f"Data directory: {data_dir}")
    print(f"Output images: {out_images_dir}")
    print(f"Output labels: {out_labels_dir}")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    
    clean_dir = os.path.join(data_dir, 'czyste')
    dirty_dir = os.path.join(data_dir, 'brudne')
    
    if not os.path.exists(clean_dir):
        print(f"Error: Missing 'czyste' directory: {clean_dir}")
        return
    if not os.path.exists(dirty_dir):
        print(f"Error: Missing 'brudne' directory: {dirty_dir}")
        return
    
    # Process clean images (no anomalies - empty labels)
    print("Processing clean images...")
    for folder_name in tqdm(sorted(os.listdir(clean_dir))):
        folder_path = os.path.join(clean_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        # Find the main .bmp file (without "czarno" in name)
        files = [f for f in os.listdir(folder_path) if f.endswith('.bmp') and 'czarno' not in f]
        if not files:
            continue
            
        main_file = files[0]
        img_path = os.path.join(folder_path, main_file)
        
        if os.path.exists(img_path):
            image = load_gray(img_path)
            if image is not None:
                base_name = f"clean_{folder_name}.bmp"
                img_out = os.path.join(out_images_dir, base_name)
                lbl_out = os.path.join(out_labels_dir, f"clean_{folder_name}.txt")
                
                # Save image
                Image.fromarray(image).save(img_out)
                # Save empty label file (no anomalies)
                with open(lbl_out, 'w') as f:
                    pass  # Empty file for clean images
            else:
                print(f"Failed to load clean image: {img_path}")
        else:
            print(f"Clean image not found: {img_path}")
    
    # Process dirty images (with anomalies - need template for comparison)
    print("Processing dirty images...")
    clean_folders = sorted(os.listdir(clean_dir))
    dirty_folders = sorted(os.listdir(dirty_dir))
    
    # Try to match clean templates with dirty samples
    for dirty_folder in tqdm(dirty_folders):
        dirty_path = os.path.join(dirty_dir, dirty_folder)
        if not os.path.isdir(dirty_path):
            continue
            
        # Find the main .bmp file
        files = [f for f in os.listdir(dirty_path) if f.endswith('.bmp') and 'czarno' not in f]
        if not files:
            continue
            
        dirty_file = files[0]
        dirty_img_path = os.path.join(dirty_path, dirty_file)
        dirty_id = extract_id_from_filename(dirty_file)
        
        # Find the best matching clean template
        template_path = find_best_template(dirty_id, clean_dir)
        
        if template_path and os.path.exists(template_path) and os.path.exists(dirty_img_path):
            template = load_gray(template_path)
            sample = load_gray(dirty_img_path)
            
            if template is not None and sample is not None:
                # Resize template to match sample if needed
                if template.shape != sample.shape:
                    template = cv2.resize(template, (sample.shape[1], sample.shape[0]))
                
                mask, boxes, diff = compute_diff_mask(template, sample, 
                                                      blur=7, thresh=20, 
                                                      min_area=500, max_area=100000)
                base_name = f"dirty_{dirty_folder}.bmp"
                img_out = os.path.join(out_images_dir, base_name)
                lbl_out = os.path.join(out_labels_dir, f"dirty_{dirty_folder}.txt")
                
                # Save sample image
                Image.fromarray(sample).save(img_out)
                # Save YOLO labels
                with open(lbl_out, 'w') as f:
                    for b in boxes:
                        f.write(bbox_to_yolo(b, sample.shape[1], sample.shape[0]) + "\n")
            else:
                print(f"Failed to load template or sample for {dirty_file}")
        else:
            print(f"Warning: Could not find suitable template for {dirty_file}")
            print(f"  Template path: {template_path}")
            print(f"  Dirty image path: {dirty_img_path}")

def prepare_dataset(pairs_dir, out_images_dir, out_labels_dir):
    """
    Legacy function - maintains compatibility with old structure
    pairs_dir structure:
      pairs_dir/
        pair_001/
          template.bmp
          sample.bmp
        pair_002/ ...
    """
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    for pair in tqdm(sorted(os.listdir(pairs_dir))):
        pdir = os.path.join(pairs_dir, pair)
        t_path = os.path.join(pdir, 'template.bmp')
        s_path = os.path.join(pdir, 'sample.bmp')
        if not (os.path.exists(t_path) and os.path.exists(s_path)):
            continue
        template = load_gray(t_path)
        sample = load_gray(s_path)
        mask, boxes, diff = compute_diff_mask(template, sample)
        base_name = pair + ".bmp"
        img_out = os.path.join(out_images_dir, base_name)
        lbl_out = os.path.join(out_labels_dir, pair + ".txt")
        # Zapisz sample jako .bmp do images
        Image.fromarray(sample).save(img_out)
        # Zapisz label txt (YOLO)
        with open(lbl_out, 'w') as f:
            for b in boxes:
                f.write(bbox_to_yolo(b, sample.shape[1], sample.shape[0]) + "\n")

def extract_id_from_filename(filename):
    """Extract ID/timestamp from filename like '48001F003202511180021.bmp'"""
    # Remove file extension
    name = filename.replace('.bmp', '').replace(' czarno', '')
    # Extract the timestamp part (last 12 digits)
    if len(name) >= 12:
        return name[-12:]
    return name

def find_best_template(target_id, clean_dir):
    """Find the best matching clean template for a given dirty image ID"""
    best_match = None
    best_score = float('inf')
    
    for folder_name in os.listdir(clean_dir):
        folder_path = os.path.join(clean_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        files = [f for f in os.listdir(folder_path) if f.endswith('.bmp') and 'czarno' not in f]
        if not files:
            continue
            
        template_id = extract_id_from_filename(files[0])
        
        # Calculate similarity score (lower is better)
        # Simple approach: compare timestamp difference
        try:
            target_num = int(target_id[-8:]) if len(target_id) >= 8 else 0
            template_num = int(template_id[-8:]) if len(template_id) >= 8 else 0
            score = abs(target_num - template_num)
            
            if score < best_score:
                best_score = score
                best_match = os.path.join(folder_path, files[0])
        except:
            continue
    
    return best_match

if __name__ == "__main__":
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use absolute paths for data structure
    data_dir = os.path.join(script_dir, "data")  # directory containing 'czyste' and 'brudne' folders
    images_dir = os.path.join(script_dir, "data", "images", "train")
    labels_dir = os.path.join(script_dir, "data", "labels", "train")
    
    # Create output directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    prepare_dataset_from_clean_dirty(data_dir, images_dir, labels_dir)
    
    # Legacy usage (if you have old pairs structure):
    # pairs_dir = os.path.join(script_dir, "pairs")  # katalog z podfolderami pair_xxx/template.bmp i sample.bmp
    # prepare_dataset(pairs_dir, images_dir, labels_dir)
    
    print("Dataset prepared.")
