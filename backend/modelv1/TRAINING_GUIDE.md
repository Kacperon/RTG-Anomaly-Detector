# Training with Manual Annotations - Quick Guide

## 📋 Workflow

### 1. Annotate Images (Already Done! ✅)
```bash
python annotation_tool.py
```
You've already done this step - great!

### 2. Prepare Dataset
This copies your manual annotations and adds clean images:
```bash
python prepare_manual_dataset.py
```

### 3. Train the Model
Train YOLO on your manual annotations:
```bash
python train_manual.py
```

### 4. Test the Model
```bash
python test_model.py
# or
python test_api.py
```

### 5. Run the Application
```bash
./start.sh
# or
python app.py
```

## 📊 What happens during training?

1. **Dataset Preparation**: 
   - Copies manually annotated dirty images (with bounding boxes)
   - Adds clean images (without anomalies, empty labels)
   - This helps the model learn both normal and anomalous patterns

2. **Dataset Split**:
   - 85% training
   - 15% validation
   - Automatically handled

3. **Training**:
   - Uses YOLOv8 nano model (fast on CPU)
   - 50 epochs (or early stopping if no improvement)
   - Saves best model based on validation performance

4. **Output**:
   - Best weights: `runs/detect/manual_annotations/weights/best.pt`
   - Training plots and metrics in `runs/detect/manual_annotations/`

## 🎯 Expected Results

- **With good annotations**: Model should detect anomalies accurately
- **Training time**: ~30-60 minutes on CPU (depends on dataset size)
- **Minimum dataset**: At least 50-100 annotated images recommended

## 🐛 Troubleshooting

**No annotations found?**
- Make sure you ran `annotation_tool.py` and saved images

**Training too slow?**
- Reduce epochs: edit `train_manual.py` line 270
- Reduce image size: change `imgsz=640` to `imgsz=416`

**Poor results?**
- Need more annotated images
- Check annotation quality in saved images
- Try training longer (more epochs)

## 📁 File Structure

```
data/
  images/
    train/
      brudne_202511190032.bmp    # Manually annotated
      brudne_202511190035.bmp
      clean_202511180021.bmp     # Clean (no anomalies)
      ...
    val/                         # Auto-created during training
      ...
  labels/
    train/
      brudne_202511190032.txt    # Your manual annotations
      brudne_202511190035.txt
      clean_202511180021.txt     # Empty (no anomalies)
      ...
    val/                         # Auto-created during training
      ...
```

## 🚀 Quick Start (All Steps)

```bash
# If you already annotated images:
python prepare_manual_dataset.py
python train_manual.py

# Then test:
python test_api.py
```
