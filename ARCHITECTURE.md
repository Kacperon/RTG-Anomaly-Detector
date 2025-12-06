# Architektura systemu - RTG Anomaly Detector

## Przegląd architektury

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   demo.py    │  │  REST API    │  │  Web UI      │         │
│  │ (Interactive)│  │  (Flask)     │  │  (React)     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      DETECTION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────┐  ┌─────────────────────────┐    │
│  │ RTGAnomalySystem         │  │ YOLO Detection          │    │
│  │ (Comparison-based)       │  │ (Deep Learning)         │    │
│  │                          │  │                         │    │
│  │  ┌────────────────────┐  │  │  ┌──────────────────┐  │    │
│  │  │ ImageMatcher       │  │  │  │ YOLOv8 Model     │  │    │
│  │  │ - Feature extract  │  │  │  │ - Inference      │  │    │
│  │  │ - Similarity calc  │  │  │  │ - NMS            │  │    │
│  │  └────────────────────┘  │  │  └──────────────────┘  │    │
│  │           ↓              │  │                         │    │
│  │  ┌────────────────────┐  │  │                         │    │
│  │  │ ImageAligner       │  │  │                         │    │
│  │  │ - ECC alignment    │  │  │                         │    │
│  │  │ - Feature-based    │  │  │                         │    │
│  │  └────────────────────┘  │  │                         │    │
│  │           ↓              │  │                         │    │
│  │  ┌────────────────────┐  │  │                         │    │
│  │  │ AnomalyDetector    │  │  │                         │    │
│  │  │ - SSIM             │  │  │                         │    │
│  │  │ - Pixel diff       │  │  │                         │    │
│  │  │ - Filtering        │  │  │                         │    │
│  │  └────────────────────┘  │  │                         │    │
│  │           ↓              │  │                         │    │
│  │  ┌────────────────────┐  │  │                         │    │
│  │  │ ReportGenerator    │  │  │                         │    │
│  │  │ - Visual (PNG)     │  │  │                         │    │
│  │  │ - Data (JSON)      │  │  │                         │    │
│  │  └────────────────────┘  │  │                         │    │
│  └──────────────────────────┘  └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ data/czyste/ │  │ data/brudne/ │  │   uploads/   │         │
│  │ (References) │  │  (Test imgs) │  │  (Incoming)  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                           │
│  │   results/   │  │anomaly_reports/                          │
│  │ (YOLO out)   │  │(Comparison out)                          │
│  └──────────────┘  └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

## Przepływ danych - System porównywania

```
┌─────────────┐
│ Input Image │
└──────┬──────┘
       │
       ↓
┌─────────────────────────────────────┐
│ 1. FEATURE EXTRACTION               │
│    - Load & preprocess image        │
│    - Calculate histogram            │
│    - Calculate gradients            │
│    - Calculate intensity stats      │
│    - Extract moments                │
└──────┬──────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────┐
│ 2. REFERENCE MATCHING               │
│    - Compare with all references    │
│    - Calculate similarity scores    │
│    - Rank by similarity             │
│    - Select best match              │
└──────┬──────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────┐
│ 3. IMAGE ALIGNMENT (optional)       │
│    ┌─────────────┬─────────────┐   │
│    │ ECC Method  │ ORB Method  │   │
│    │ - Affine    │ - Keypoints │   │
│    │ - Transform │ - Matching  │   │
│    └─────────────┴─────────────┘   │
└──────┬──────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────┐
│ 4. DIFFERENCE CALCULATION           │
│    ┌─────────────┬─────────────┐   │
│    │ SSIM Method │ Pixel Diff  │   │
│    │ - Structural│ - Absolute  │   │
│    │ - Similarity│ - Difference│   │
│    └─────────────┴─────────────┘   │
└──────┬──────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────┐
│ 5. ANOMALY DETECTION                │
│    - Threshold difference map       │
│    - Morphological operations       │
│    - Contour detection              │
│    - Filter by size/shape           │
│    - Extract bounding boxes         │
└──────┬──────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────┐
│ 6. REPORT GENERATION                │
│    ┌─────────────┬─────────────┐   │
│    │ Visual (PNG)│ Data (JSON) │   │
│    │ - Grid 2x3  │ - Anomalies │   │
│    │ - Heatmap   │ - Metadata  │   │
│    │ - Annotated │ - Scores    │   │
│    └─────────────┴─────────────┘   │
└──────┬──────────────────────────────┘
       │
       ↓
┌─────────────┐
│   Results   │
│ - has_anomaly
│ - anomaly_count
│ - similarity
│ - ssim_score
│ - report_path
└─────────────┘
```

## Komponenty szczegółowo

### 1. ImageMatcher

**Cel:** Znalezienie najbardziej podobnego obrazu wzorcowego

**Wejście:**
- Query image (grayscale numpy array)
- Top K (liczba wyników)

**Proces:**
1. Extract features z query image:
   - Histogram (64 bins)
   - Gradient statistics (mean, std)
   - Intensity statistics (mean, std)
   - Image moments
   
2. Porównaj z każdym referencyjnym obrazem:
   - Histogram correlation
   - Gradient difference (normalized)
   - Intensity difference (normalized)
   - Weighted average (0.5, 0.25, 0.25)

3. Sortuj i zwróć top K

**Wyjście:**
```python
[
  {
    'path': Path,
    'image': np.ndarray,
    'similarity': float  # 0-1
  },
  ...
]
```

**Wydajność:** O(N) gdzie N = liczba obrazów referencyjnych

---

### 2. ImageAligner

**Cel:** Wyrównanie obrazu testowego do referencji

**Metody:**

#### ECC (Enhanced Correlation Coefficient)
- **Transformacja:** Affine (6 DOF)
- **Kryteria:** EPS=1e-6, MAX_ITER=5000
- **Zalety:** Szybka, dokładna dla niewielkich różnic
- **Wady:** Może zawieść dla dużych przesunięć

#### Feature-based (ORB)
- **Detektor:** ORB (5000 keypoints)
- **Matcher:** Brute Force Hamming
- **Filtr:** Lowe's ratio test (0.75)
- **Transformacja:** Affine (RANSAC)
- **Zalety:** Odporna na duże różnice
- **Wady:** Wolniejsza, wymaga wyraźnych cech

**Wyjście:**
```python
aligned_image: np.ndarray  # Wyrównany obraz
transform_matrix: np.ndarray  # 2x3 affine matrix
```

**Wydajność:** 
- ECC: ~1-3s
- ORB: ~2-5s

---

### 3. AnomalyDetector

**Cel:** Wykrycie anomalii na podstawie różnic

**Metody detekcji:**

#### SSIM (Structural Similarity Index)
```python
score, diff_map = ssim(ref, img, full=True)
diff_map = (1 - diff_map) * 255  # Convert to difference
```
- **Zakres SSIM:** 0-1 (1 = identyczne)
- **Zalety:** Uwzględnia strukturę, odporne na szum
- **Wady:** Wolniejsze od pixel diff

#### Pixel Difference
```python
diff_map = cv2.absdiff(ref_processed, img_processed)
```
- **Zakres:** 0-255
- **Zalety:** Szybkie, proste
- **Wady:** Wrażliwe na różnice w jasności

**Preprocessing:**
1. Histogram equalization
2. Fast NL Means Denoising (h=10)

**Filtrowanie anomalii:**
1. Threshold (default: 25)
2. Morphological opening (5x5 ellipse, 2 iter)
3. Morphological closing (9x9 ellipse, 2 iter)
4. Contour detection (RETR_EXTERNAL)
5. Filtry:
   - Area: 300 < area < 50000
   - Aspect ratio: 0.1 < ratio < 10
   - Solidity: > 0.3

**Wyjście:**
```python
{
  'difference_map': np.ndarray,
  'anomalies': [
    {
      'bbox': (x, y, w, h),
      'area': float,
      'solidity': float,
      'aspect_ratio': float,
      'contour': np.ndarray
    },
    ...
  ],
  'anomaly_count': int,
  'has_anomaly': bool,
  'ssim_score': float  # if SSIM used
}
```

---

### 4. AnomalyReportGenerator

**Cel:** Generowanie raportów wizualnych i danych

**Visual Report (PNG):**

Grid layout 2x3:
```
┌───────────┬───────────┬───────────┐
│  Original │ Reference │  Aligned  │
├───────────┼───────────┼───────────┤
│ Heatmap   │ Annotated │ Annotated │
└───────────┴───────────┴───────────┘
       Summary bar (status, count, SSIM)
```

Elementy:
- Labels na każdym obrazie
- Corner markers na anomaliach
- Bounding boxes (czerwone)
- Heatmap (JET colormap)
- Summary bar (czarne tło, białe/kolorowe teksty)

**JSON Report:**
```json
{
  "timestamp": "ISO-8601",
  "has_anomaly": true,
  "anomaly_count": 3,
  "ssim_score": 0.9234,
  "anomalies": [
    {
      "id": 1,
      "bbox": [x, y, w, h],
      "area": 10000.0,
      "solidity": 0.85,
      "aspect_ratio": 1.2
    }
  ],
  "metadata": {
    "input_image": "path",
    "reference_image": "path",
    "similarity": 0.8765,
    ...
  }
}
```

---

### 5. RTGAnomalySystem

**Cel:** Integracja wszystkich komponentów

**Inicjalizacja:**
```python
system = RTGAnomalySystem(
    reference_dir='data/czyste',
    output_dir='anomaly_reports'
)
```

**Pipeline:**
```
process_image(image_path)
    ↓
1. Load image
    ↓
2. matcher.find_best_match(img, top_k=1)
    ↓
3. [optional] aligner.align_images(ref, img)
    ↓
4. detector.detect_anomalies(ref, aligned)
    ↓
5. [optional] generator.generate_report(...)
    ↓
Return results dict
```

**Batch processing:**
```python
batch_process(image_dir, pattern='*.bmp')
    ↓
For each image:
    process_image(img)
    ↓
Aggregate statistics
    ↓
Return list of results
```

---

## API Architecture

### Flask Backend (app.py)

**Endpointy:**

```python
# Health & Status
GET  /api/health              # Basic health check
GET  /api/detector-status     # System status (YOLO + Comparison)

# YOLO Detection
POST /api/load-model          # Load YOLO model
POST /api/upload              # Upload image
POST /api/analyze             # Analyze with YOLO

# Comparison Detection (NEW)
POST /api/analyze-comparison  # Analyze with comparison
POST /api/batch-analyze       # Batch processing

# Reports
GET  /api/download-report/<id> # Download report
```

**Request/Response:**

```python
# analyze-comparison
Request:
{
  "file_id": "uuid",
  "use_alignment": true,
  "use_ssim": true
}

Response:
{
  "method": "comparison_based",
  "has_anomaly": true,
  "anomaly_count": 3,
  "anomalies": [...],
  "similarity": 0.8765,
  "ssim_score": 0.9234,
  "report_image": "base64...",
  "report_path": "path",
  ...
}
```

---

## Przepływ danych w systemie pełnym

```
┌──────────┐
│ User     │
└────┬─────┘
     │
     ↓ (upload BMP)
┌────────────────┐
│ Flask Backend  │
├────────────────┤
│ /api/upload    │
└────┬───────────┘
     │ save to uploads/
     ↓
┌────────────────────┐
│ File System        │
│ uploads/uuid.bmp   │
└────┬───────────────┘
     │
     ↓ (analyze request)
┌────────────────────────┐
│ RTGAnomalySystem       │
├────────────────────────┤
│ 1. ImageMatcher        │ → data/czyste/
│ 2. ImageAligner        │
│ 3. AnomalyDetector     │
│ 4. ReportGenerator     │
└────┬───────────────────┘
     │ save report
     ↓
┌──────────────────────────┐
│ File System              │
│ anomaly_reports/report.png│
│ anomaly_reports/report.json│
└────┬─────────────────────┘
     │
     ↓ (return results)
┌────────────────┐
│ Flask Backend  │
│ /api/analyze-  │
│  comparison    │
└────┬───────────┘
     │ (base64 + JSON)
     ↓
┌──────────┐
│ User     │
│ - View   │
│ - Download│
└──────────┘
```

---

## Skalowalność

### Obecna architektura

- **Single-threaded:** Jeden request na raz
- **CPU-bound:** Wszystkie obliczenia na CPU
- **File-based:** Dane na dysku

### Możliwości skalowania

#### 1. Horizontal (więcej instancji)
```
Load Balancer
    ↓
┌────────┬────────┬────────┐
│ App 1  │ App 2  │ App 3  │
└────────┴────────┴────────┘
    ↓
Shared Storage (NFS/S3)
```

#### 2. Vertical (więcej zasobów)
```
- GPU dla wyrównywania (CUDA)
- Multi-threading dla batch
- Większa RAM dla cache'owania
```

#### 3. Cache layer
```
Redis Cache
    ↓
┌─────────────────┐
│ Reference cache │
│ - Features      │
│ - Images (small)│
└─────────────────┘
```

#### 4. Queue system
```
Client → API → Queue → Workers → Storage
         ↓           ↓
      Request ID   Process
         ↑           ↓
      Poll status  Update
```

---

## Wydajność

### Bottlenecks

1. **Feature extraction:** O(N) dla N referencji
2. **Image alignment:** ECC iterations (1-3s)
3. **SSIM calculation:** Sliding window O(WH)
4. **File I/O:** Loading large BMP files

### Optymalizacje

1. **Cache reference features:**
```python
# Pre-compute on startup
with open('ref_features.pkl', 'rb') as f:
    cached_features = pickle.load(f)
```

2. **Resize for matching:**
```python
# Use smaller images for matching
small = cv2.resize(img, (256, 256))
```

3. **Skip alignment:**
```python
# Trade accuracy for speed
result = system.process_image(img, use_alignment=False)
```

4. **Parallel batch:**
```python
# Use multiprocessing
from multiprocessing import Pool
with Pool(4) as p:
    results = p.map(process_image, image_paths)
```

---

## Bezpieczeństwo

### Obecne zabezpieczenia

- ✅ File type validation (BMP, PNG, JPG)
- ✅ Path sanitization
- ✅ Error handling

### Potrzebne zabezpieczenia

- [ ] Authentication (JWT)
- [ ] Rate limiting
- [ ] File size limits (enforced)
- [ ] Input sanitization
- [ ] HTTPS
- [ ] CORS policy (stricter)

---

## Monitoring i Logging

### Aktualnie

```python
# Console output
print(f"Processing: {image_path}")
```

### Zalecane

```python
import logging

logger = logging.getLogger(__name__)
logger.info(f"Processing image: {image_path}")
logger.error(f"Failed to process: {e}")

# Metrics
from prometheus_client import Counter, Histogram

processing_time = Histogram('image_processing_seconds', 'Time spent processing')
anomalies_detected = Counter('anomalies_total', 'Total anomalies detected')
```

---

**Wersja architektury:** 2.0.0  
**Data:** 2025-12-06
