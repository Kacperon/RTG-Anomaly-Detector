#!/usr/bin/env python3
"""
Simple annotation tool for marking anomalies in X-ray images.
Click and drag to draw bounding boxes around anomalies.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

class AnnotationTool:
    def __init__(self, root, data_dir, output_images_dir, output_labels_dir):
        self.root = root
        self.root.title("RTG Anomaly Annotation Tool")
        
        self.data_dir = Path(data_dir)
        self.output_images_dir = Path(output_images_dir)
        self.output_labels_dir = Path(output_labels_dir)
        
        # Create output directories
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # State variables
        self.image_files = []
        self.current_index = 0
        self.current_image = None
        self.display_image = None
        self.original_size = None
        self.scale = 1.0
        
        # Drawing state
        self.boxes = []  # List of (x1, y1, x2, y2) in original image coordinates
        self.drawing = False
        self.start_point = None
        self.temp_box = None
        
        # Setup UI
        self.setup_ui()
        
        # Load images
        self.load_image_list()
        if self.image_files:
            self.load_current_image()
    
    def setup_ui(self):
        # Top frame - controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Navigation buttons
        tk.Button(control_frame, text="◀ Previous", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Next ▶", command=self.next_image).pack(side=tk.LEFT, padx=2)
        
        # Image counter
        self.counter_label = tk.Label(control_frame, text="0/0")
        self.counter_label.pack(side=tk.LEFT, padx=10)
        
        # Current file name
        self.filename_label = tk.Label(control_frame, text="", fg="blue")
        self.filename_label.pack(side=tk.LEFT, padx=10)
        
        # Action buttons
        tk.Button(control_frame, text="Clear All Boxes", command=self.clear_boxes, bg="#ffcccc").pack(side=tk.RIGHT, padx=2)
        tk.Button(control_frame, text="Undo Last Box", command=self.undo_last_box, bg="#ffffcc").pack(side=tk.RIGHT, padx=2)
        tk.Button(control_frame, text="Save & Next", command=self.save_and_next, bg="#ccffcc").pack(side=tk.RIGHT, padx=2)
        tk.Button(control_frame, text="Skip (Empty)", command=self.skip_image, bg="#ccccff").pack(side=tk.RIGHT, padx=2)
        
        # Instructions frame
        info_frame = tk.Frame(self.root, bg="#f0f0f0")
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        instructions = "Instructions: Click and drag to draw boxes around anomalies | Right-click on box to delete it | ESC to clear current drawing"
        tk.Label(info_frame, text=instructions, bg="#f0f0f0", font=("Arial", 9)).pack(pady=2)
        
        # Canvas frame
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg="gray", cursor="cross")
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)
        
        # Bind keyboard events
        self.root.bind("<Escape>", lambda e: self.cancel_drawing())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("s", lambda e: self.save_and_next())
        self.root.bind("d", lambda e: self.skip_image())
        self.root.bind("u", lambda e: self.undo_last_box())
        
    def load_image_list(self):
        """Load all images from brudne folder only"""
        self.image_files = []
        
        # Load from brudne (dirty) folder - only these need manual annotation
        brudne_dir = self.data_dir / "brudne"
        if brudne_dir.exists():
            for folder in sorted(brudne_dir.iterdir()):
                if folder.is_dir():
                    for img_file in folder.glob("*.bmp"):
                        if "czarno" not in img_file.name:
                            self.image_files.append(("brudne", folder.name, img_file))
        else:
            print(f"Warning: Brudne directory not found: {brudne_dir}")
        
        print(f"Found {len(self.image_files)} dirty images to annotate")
        
    def load_current_image(self):
        """Load and display the current image"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
            
        category, folder_name, img_path = self.image_files[self.current_index]
        
        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", f"Failed to load image: {img_path}")
            return
            
        self.current_image = img
        self.original_size = img.shape[:2]  # (height, width)
        
        # Check if annotations already exist
        label_file = self.output_labels_dir / f"{category}_{folder_name}.txt"
        self.boxes = []
        if label_file.exists():
            self.load_existing_annotations(label_file)
        
        # Update display
        self.update_display()
        self.update_info()
        
    def load_existing_annotations(self, label_file):
        """Load existing YOLO format annotations"""
        h, w = self.original_size
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class x_center y_center width height (normalized)
                    _, xc, yc, bw, bh = map(float, parts[:5])
                    
                    # Convert to pixel coordinates
                    x_center = xc * w
                    y_center = yc * h
                    box_w = bw * w
                    box_h = bh * h
                    
                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)
                    
                    self.boxes.append((x1, y1, x2, y2))
    
    def update_display(self):
        """Update the canvas with current image and boxes"""
        if self.current_image is None:
            return
            
        # Convert to RGB for display
        img_display = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB)
        
        # Draw existing boxes
        for box in self.boxes:
            cv2.rectangle(img_display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
        # Draw temporary box (while drawing)
        if self.temp_box:
            cv2.rectangle(img_display, (self.temp_box[0], self.temp_box[1]), 
                         (self.temp_box[2], self.temp_box[3]), (255, 255, 0), 2)
        
        # Calculate scale to fit window (max 1200x800)
        max_w, max_h = 1200, 800
        h, w = img_display.shape[:2]
        scale_w = max_w / w
        scale_h = max_h / h
        self.scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        
        img_resized = cv2.resize(img_display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to PhotoImage
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.display_image = ImageTk.PhotoImage(img_pil)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def update_info(self):
        """Update info labels"""
        if self.image_files:
            self.counter_label.config(text=f"{self.current_index + 1}/{len(self.image_files)}")
            category, folder_name, img_path = self.image_files[self.current_index]
            self.filename_label.config(text=f"{category}/{folder_name} | Boxes: {len(self.boxes)}")
        
    def canvas_to_image_coords(self, x, y):
        """Convert canvas coordinates to original image coordinates"""
        img_x = int(x / self.scale)
        img_y = int(y / self.scale)
        return img_x, img_y
        
    def on_mouse_down(self, event):
        """Start drawing a box"""
        self.drawing = True
        x, y = self.canvas_to_image_coords(event.x, event.y)
        self.start_point = (x, y)
        
    def on_mouse_drag(self, event):
        """Update temporary box while dragging"""
        if not self.drawing or not self.start_point:
            return
            
        x, y = self.canvas_to_image_coords(event.x, event.y)
        self.temp_box = (self.start_point[0], self.start_point[1], x, y)
        self.update_display()
        
    def on_mouse_up(self, event):
        """Finish drawing a box"""
        if not self.drawing or not self.start_point:
            return
            
        x, y = self.canvas_to_image_coords(event.x, event.y)
        
        # Normalize coordinates (ensure x1 < x2, y1 < y2)
        x1 = min(self.start_point[0], x)
        y1 = min(self.start_point[1], y)
        x2 = max(self.start_point[0], x)
        y2 = max(self.start_point[1], y)
        
        # Only add if box has reasonable size
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            self.boxes.append((x1, y1, x2, y2))
            
        self.drawing = False
        self.start_point = None
        self.temp_box = None
        self.update_display()
        self.update_info()
        
    def on_right_click(self, event):
        """Delete box under cursor"""
        x, y = self.canvas_to_image_coords(event.x, event.y)
        
        # Find box under cursor
        for i, box in enumerate(self.boxes):
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                self.boxes.pop(i)
                self.update_display()
                self.update_info()
                break
                
    def cancel_drawing(self):
        """Cancel current drawing"""
        self.drawing = False
        self.start_point = None
        self.temp_box = None
        self.update_display()
        
    def clear_boxes(self):
        """Clear all boxes"""
        if messagebox.askyesno("Clear All", "Remove all boxes from this image?"):
            self.boxes = []
            self.update_display()
            self.update_info()
            
    def undo_last_box(self):
        """Remove the last drawn box"""
        if self.boxes:
            self.boxes.pop()
            self.update_display()
            self.update_info()
            
    def save_current_annotations(self):
        """Save current annotations to YOLO format"""
        if not self.image_files:
            return
            
        category, folder_name, img_path = self.image_files[self.current_index]
        
        # Save image
        output_img_name = f"{category}_{folder_name}.bmp"
        output_img_path = self.output_images_dir / output_img_name
        cv2.imwrite(str(output_img_path), self.current_image)
        
        # Save labels
        output_label_name = f"{category}_{folder_name}.txt"
        output_label_path = self.output_labels_dir / output_label_name
        
        h, w = self.original_size
        with open(output_label_path, 'w') as f:
            for box in self.boxes:
                x1, y1, x2, y2 = box
                
                # Convert to YOLO format (normalized center coordinates)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_w = (x2 - x1) / w
                box_h = (y2 - y1) / h
                
                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                
        print(f"Saved: {output_img_name} with {len(self.boxes)} boxes")
        
    def save_and_next(self):
        """Save current annotations and move to next image"""
        self.save_current_annotations()
        self.next_image()
        
    def skip_image(self):
        """Save empty label file and move to next"""
        self.boxes = []
        self.save_current_annotations()
        self.next_image()
        
    def next_image(self):
        """Move to next image"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
        else:
            messagebox.showinfo("Done", "No more images to annotate!")
            
    def prev_image(self):
        """Move to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

def main():
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Setup paths
    data_dir = script_dir / "data"
    output_images_dir = script_dir / "data" / "images" / "train"
    output_labels_dir = script_dir / "data" / "labels" / "train"
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please ensure you have a 'data' folder with 'czyste' and 'brudne' subdirectories")
        return
    
    # Create and run GUI
    root = tk.Tk()
    root.geometry("1400x900")
    app = AnnotationTool(root, data_dir, output_images_dir, output_labels_dir)
    root.mainloop()

if __name__ == "__main__":
    main()
