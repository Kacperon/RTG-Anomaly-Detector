# anomaly_detector.py - Zaawansowany system detekcji anomalii na RTG
"""
System detekcji anomalii poprzez porównywanie obrazów RTG:
1. Znajduje najbardziej podobny obraz wzorcowy (czysty)
2. Wyrównuje obrazy (image alignment)
3. Oblicza różnice i wykrywa anomalie
4. Generuje szczegółowy raport z wizualizacją
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
import pickle


class ImageMatcher:
    """Znajduje najbardziej podobny obraz wzorcowy"""
    
    def __init__(self, reference_dir: str):
        """
        Args:
            reference_dir: Katalog z obrazami wzorcowymi (czystymi)
        """
        self.reference_dir = Path(reference_dir)
        self.reference_images = []
        self.reference_features = []
        self._load_references()
    
    def _load_references(self):
        """Ładuje wszystkie obrazy wzorcowe"""
        print(f"📁 Ładowanie obrazów wzorcowych z: {self.reference_dir}")
        
        for root, dirs, files in os.walk(self.reference_dir):
            for file in files:
                if file.endswith('.bmp') and 'czarno' not in file.lower():
                    img_path = Path(root) / file
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Oblicz features dla szybszego dopasowywania
                        features = self._extract_features(img)
                        self.reference_images.append({
                            'path': img_path,
                            'image': img,
                            'features': features
                        })
        
        print(f"✅ Załadowano {len(self.reference_images)} obrazów wzorcowych")
    
    def _extract_features(self, img: np.ndarray) -> Dict:
        """Wyodrębnia cechy obrazu do porównywania"""
        # Zmniejsz dla szybszego przetwarzania
        small = cv2.resize(img, (256, 256))
        
        # Histogram
        hist = cv2.calcHist([small], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Gradient magnitude (krawędzie)
        sobelx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Momenty obrazu
        moments = cv2.moments(small)
        
        return {
            'histogram': hist,
            'gradient_mean': np.mean(gradient_mag),
            'gradient_std': np.std(gradient_mag),
            'intensity_mean': np.mean(small),
            'intensity_std': np.std(small),
            'moments': moments
        }
    
    def _compare_features(self, features1: Dict, features2: Dict) -> float:
        """Porównuje cechy dwóch obrazów, zwraca podobieństwo [0-1]"""
        # Histogram correlation
        hist_corr = cv2.compareHist(
            features1['histogram'], 
            features2['histogram'], 
            cv2.HISTCMP_CORREL
        )
        
        # Statystyki gradientu
        grad_diff = abs(features1['gradient_mean'] - features2['gradient_mean'])
        grad_diff_norm = 1 - min(grad_diff / 255.0, 1.0)
        
        # Statystyki intensywności
        int_diff = abs(features1['intensity_mean'] - features2['intensity_mean'])
        int_diff_norm = 1 - min(int_diff / 255.0, 1.0)
        
        # Łączone podobieństwo (ważona średnia)
        similarity = (
            0.5 * hist_corr +
            0.25 * grad_diff_norm +
            0.25 * int_diff_norm
        )
        
        return max(0, min(1, similarity))
    
    def find_best_match(self, query_img: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Znajdź najbardziej podobne obrazy wzorcowe
        
        Args:
            query_img: Obraz do dopasowania
            top_k: Ile najlepszych dopasowań zwrócić
            
        Returns:
            Lista słowników z informacjami o dopasowaniach
        """
        query_features = self._extract_features(query_img)
        
        matches = []
        for ref in self.reference_images:
            similarity = self._compare_features(query_features, ref['features'])
            matches.append({
                'path': ref['path'],
                'image': ref['image'],
                'similarity': similarity
            })
        
        # Sortuj po podobieństwie
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches[:top_k]


class ImageAligner:
    """Wyrównuje dwa obrazy dla dokładnego porównania"""
    
    @staticmethod
    def align_images(reference: np.ndarray, image: np.ndarray, 
                     method: str = 'ecc') -> Tuple[np.ndarray, np.ndarray]:
        """
        Wyrównuje obraz do referencji
        
        Args:
            reference: Obraz wzorcowy
            image: Obraz do wyrównania
            method: 'ecc' lub 'feature' (Enhanced Correlation Coefficient)
            
        Returns:
            (aligned_image, transformation_matrix)
        """
        # Upewnij się, że obrazy mają ten sam rozmiar
        if reference.shape != image.shape:
            image = cv2.resize(image, (reference.shape[1], reference.shape[0]))
        
        if method == 'ecc':
            return ImageAligner._align_ecc(reference, image)
        else:
            return ImageAligner._align_feature(reference, image)
    
    @staticmethod
    def _align_ecc(reference: np.ndarray, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Wyrównanie ECC (Enhanced Correlation Coefficient)"""
        # Konwertuj do float32
        ref_gray = reference.astype(np.float32)
        img_gray = image.astype(np.float32)
        
        # Zdefiniuj typ transformacji (affine)
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Kryteria zakończenia
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
        
        try:
            # Wyrównaj obrazy
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, img_gray, warp_matrix, warp_mode, criteria, 
                inputMask=None, gaussFiltSize=5
            )
            
            # Zastosuj transformację
            aligned = cv2.warpAffine(
                image, warp_matrix, (reference.shape[1], reference.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            
            return aligned, warp_matrix
            
        except Exception as e:
            print(f"⚠️ Wyrównanie ECC nie powiodło się: {e}, zwracam oryginalny obraz")
            return image, np.eye(2, 3, dtype=np.float32)
    
    @staticmethod
    def _align_feature(reference: np.ndarray, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Wyrównanie oparte na cechach (ORB)"""
        # Wykryj punkty kluczowe i deskryptory
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(reference, None)
        kp2, des2 = orb.detectAndCompute(image, None)
        
        if des1 is None or des2 is None:
            return image, np.eye(2, 3, dtype=np.float32)
        
        # Dopasuj cechy
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # Filtruj dobre dopasowania (Lowe's ratio test)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return image, np.eye(2, 3, dtype=np.float32)
        
        # Wyodrębnij punkty
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Znajdź transformację afiniczną
        warp_matrix, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts)
        
        if warp_matrix is None:
            return image, np.eye(2, 3, dtype=np.float32)
        
        # Zastosuj transformację
        aligned = cv2.warpAffine(
            image, warp_matrix, (reference.shape[1], reference.shape[0])
        )
        
        return aligned, warp_matrix


class AnomalyDetector:
    """Wykrywa anomalie przez porównanie obrazów"""
    
    def __init__(self, threshold: float = 25, min_area: int = 300, max_area: int = 50000):
        """
        Args:
            threshold: Próg różnicy pikseli
            min_area: Minimalna powierzchnia anomalii (piksele)
            max_area: Maksymalna powierzchnia anomalii (piksele)
        """
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
    
    def detect_anomalies(self, reference: np.ndarray, image: np.ndarray,
                        use_ssim: bool = True) -> Dict:
        """
        Wykryj anomalie porównując dwa obrazy
        
        Args:
            reference: Obraz wzorcowy (czysty)
            image: Obraz do sprawdzenia
            use_ssim: Czy użyć SSIM zamiast prostej różnicy
            
        Returns:
            Słownik z wynikami detekcji
        """
        # Upewnij się, że obrazy mają ten sam rozmiar
        if reference.shape != image.shape:
            image = cv2.resize(image, (reference.shape[1], reference.shape[0]))
        
        # Wstępne przetwarzanie
        ref_processed = self._preprocess(reference)
        img_processed = self._preprocess(image)
        
        if use_ssim:
            # SSIM - lepiej radzi sobie z niewielkimi różnicami w jasności
            score, diff_map = ssim(ref_processed, img_processed, full=True)
            diff_map = (1 - diff_map) * 255
            diff_map = diff_map.astype(np.uint8)
        else:
            # Prosta różnica bezwzględna
            diff_map = cv2.absdiff(ref_processed, img_processed)
        
        # Wykryj anomalie
        anomalies = self._find_anomalies(diff_map)
        
        return {
            'difference_map': diff_map,
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'has_anomaly': len(anomalies) > 0,
            'ssim_score': score if use_ssim else None
        }
    
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Wstępne przetwarzanie obrazu"""
        # Histogram equalization dla lepszego kontrastu
        img_eq = cv2.equalizeHist(img)
        
        # Denoising
        img_denoised = cv2.fastNlMeansDenoising(img_eq, h=10)
        
        return img_denoised
    
    def _find_anomalies(self, diff_map: np.ndarray) -> List[Dict]:
        """Znajdź regiony anomalii na mapie różnic"""
        # Progowanie
        _, binary = cv2.threshold(diff_map, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Operacje morfologiczne
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Znajdź kontury
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        anomalies = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtruj po obszarze
            if area < self.min_area or area > self.max_area:
                continue
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtruj po kształcie (aspect ratio)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                continue
            
            # Solidność (solidity)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.3:  # Zbyt nieregularne
                continue
            
            anomalies.append({
                'bbox': (x, y, w, h),
                'area': area,
                'solidity': solidity,
                'aspect_ratio': aspect_ratio,
                'contour': contour
            })
        
        return anomalies


class AnomalyReportGenerator:
    """Generuje raporty z wykrytymi anomaliami"""
    
    @staticmethod
    def generate_report(original_img: np.ndarray, reference_img: np.ndarray,
                       aligned_img: np.ndarray, detection_result: Dict,
                       output_path: str, metadata: Dict = None) -> str:
        """
        Generuje kompletny raport wizualny
        
        Args:
            original_img: Oryginalny obraz do sprawdzenia
            reference_img: Dopasowany obraz wzorcowy
            aligned_img: Wyrównany obraz
            detection_result: Wyniki detekcji
            output_path: Ścieżka do zapisu raportu
            metadata: Dodatkowe metadane
            
        Returns:
            Ścieżka do wygenerowanego raportu
        """
        # Przygotuj wizualizacje
        annotated = AnomalyReportGenerator._draw_anomalies(
            original_img, detection_result['anomalies']
        )
        diff_colored = AnomalyReportGenerator._colorize_diff(
            detection_result['difference_map']
        )
        
        # Utwórz grid z wizualizacjami
        report_img = AnomalyReportGenerator._create_report_grid(
            original_img, reference_img, aligned_img,
            diff_colored, annotated, detection_result
        )
        
        # Zapisz raport obrazowy
        cv2.imwrite(output_path, report_img)
        
        # Generuj raport JSON
        json_path = output_path.rsplit('.', 1)[0] + '_report.json'
        AnomalyReportGenerator._save_json_report(
            json_path, detection_result, metadata
        )
        
        return output_path
    
    @staticmethod
    def _draw_anomalies(img: np.ndarray, anomalies: List[Dict]) -> np.ndarray:
        """Rysuje wykryte anomalie na obrazie"""
        # Konwertuj do BGR dla kolorowych adnotacji
        if len(img.shape) == 2:
            annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            annotated = img.copy()
        
        for i, anomaly in enumerate(anomalies):
            x, y, w, h = anomaly['bbox']
            
            # Rysuj prostokąt - zawsze czerwony
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
            # Rysuj markery w rogach - także czerwone (wszystkie rogi)
            corner_size = 15
            # Lewy górny róg
            cv2.line(annotated, (x, y), (x + corner_size, y), (0, 0, 255), 4)
            cv2.line(annotated, (x, y), (x, y + corner_size), (0, 0, 255), 4)
            # Prawy górny róg
            cv2.line(annotated, (x+w, y), (x+w - corner_size, y), (0, 0, 255), 4)
            cv2.line(annotated, (x+w, y), (x+w, y + corner_size), (0, 0, 255), 4)
            # Lewy dolny róg
            cv2.line(annotated, (x, y+h), (x + corner_size, y+h), (0, 0, 255), 4)
            cv2.line(annotated, (x, y+h), (x, y+h - corner_size), (0, 0, 255), 4)
            # Prawy dolny róg
            cv2.line(annotated, (x+w, y+h), (x+w - corner_size, y+h), (0, 0, 255), 4)
            cv2.line(annotated, (x+w, y+h), (x+w, y+h - corner_size), (0, 0, 255), 4)
            
            # Etykieta - czerwone tło, biały tekst
            label = f"A{i+1}: {anomaly['area']:.0f}px"
            cv2.putText(annotated, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated
    
    @staticmethod
    def _colorize_diff(diff_map: np.ndarray) -> np.ndarray:
        """Koloruje mapę różnic dla lepszej wizualizacji"""
        # Normalizuj do 0-255
        diff_normalized = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX)
        
        # Zastosuj kolorową mapę (heatmap)
        colored = cv2.applyColorMap(diff_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        return colored
    
    @staticmethod
    def _create_report_grid(original: np.ndarray, reference: np.ndarray,
                           aligned: np.ndarray, diff_colored: np.ndarray,
                           annotated: np.ndarray, detection_result: Dict) -> np.ndarray:
        """Tworzy grid z wszystkimi wizualizacjami"""
        # Konwertuj grayscale do BGR jeśli potrzeba
        def to_bgr(img):
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        
        original = to_bgr(original)
        reference = to_bgr(reference)
        aligned = to_bgr(aligned)
        annotated = to_bgr(annotated)
        
        # Upewnij się, że wszystkie mają ten sam rozmiar
        h, w = original.shape[:2]
        reference = cv2.resize(reference, (w, h))
        aligned = cv2.resize(aligned, (w, h))
        diff_colored = cv2.resize(diff_colored, (w, h))
        annotated = cv2.resize(annotated, (w, h))
        
        # Dodaj etykiety
        def add_label(img, text, color=(255, 255, 255)):
            labeled = img.copy()
            cv2.rectangle(labeled, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(labeled, text, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            return labeled
        
        original = add_label(original, "Obraz testowy")
        reference = add_label(reference, "Obraz wzorcowy")
        aligned = add_label(aligned, "Wyrownany")
        diff_colored = add_label(diff_colored, "Mapa roznic (heatmap)")
        annotated = add_label(annotated, f"Wykryte anomalie: {len(detection_result['anomalies'])}", 
                            (0, 255, 0) if detection_result['has_anomaly'] else (255, 255, 255))
        
        # Grid 2x3
        row1 = np.hstack([original, reference, aligned])
        row2 = np.hstack([diff_colored, annotated, annotated])  # Duplikuj ostatni dla symetrii
        
        grid = np.vstack([row1, row2])
        
        # Dodaj podsumowanie na dole
        summary_height = 100
        summary = np.zeros((summary_height, grid.shape[1], 3), dtype=np.uint8)
        
        anomaly_count = len(detection_result['anomalies'])
        status = "ANOMALIA WYKRYTA!" if detection_result['has_anomaly'] else "BRAK ANOMALII"
        status_color = (0, 0, 255) if detection_result['has_anomaly'] else (0, 255, 0)
        
        cv2.putText(summary, status, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        cv2.putText(summary, f"Liczba anomalii: {anomaly_count}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        if detection_result.get('ssim_score') is not None:
            cv2.putText(summary, f"SSIM: {detection_result['ssim_score']:.4f}", 
                       (grid.shape[1] - 300, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        final = np.vstack([grid, summary])
        
        return final
    
    @staticmethod
    def _save_json_report(json_path: str, detection_result: Dict, metadata: Dict = None):
        """Zapisuje raport w formacie JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'has_anomaly': detection_result['has_anomaly'],
            'anomaly_count': detection_result['anomaly_count'],
            'ssim_score': detection_result.get('ssim_score'),
            'anomalies': []
        }
        
        for i, anomaly in enumerate(detection_result['anomalies']):
            report['anomalies'].append({
                'id': i + 1,
                'bbox': anomaly['bbox'],
                'area': float(anomaly['area']),
                'solidity': float(anomaly['solidity']),
                'aspect_ratio': float(anomaly['aspect_ratio'])
            })
        
        if metadata:
            report['metadata'] = metadata
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


class RTGAnomalySystem:
    """Główny system detekcji anomalii RTG"""
    
    def __init__(self, reference_dir: str, output_dir: str = 'anomaly_reports'):
        """
        Args:
            reference_dir: Katalog z obrazami wzorcowymi (czystymi)
            output_dir: Katalog do zapisywania raportów
        """
        self.matcher = ImageMatcher(reference_dir)
        self.aligner = ImageAligner()
        self.detector = AnomalyDetector()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🚀 System detekcji anomalii RTG zainicjalizowany")
        print(f"📂 Obrazy wzorcowe: {reference_dir}")
        print(f"📂 Raporty: {output_dir}")
    
    def process_image(self, image_path: str, use_alignment: bool = True,
                     use_ssim: bool = True, save_report: bool = True) -> Dict:
        """
        Przetwórz obraz i wykryj anomalie
        
        Args:
            image_path: Ścieżka do obrazu do sprawdzenia
            use_alignment: Czy wyrównywać obrazy
            use_ssim: Czy użyć SSIM
            save_report: Czy zapisać raport
            
        Returns:
            Słownik z wynikami analizy
        """
        print(f"\n🔍 Przetwarzanie: {image_path}")
        
        # Wczytaj obraz
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")
        
        # Znajdź najbardziej podobny obraz wzorcowy
        print("🔎 Szukanie najbardziej podobnego obrazu wzorcowego...")
        matches = self.matcher.find_best_match(img, top_k=1)
        
        if not matches:
            raise ValueError("Nie znaleziono obrazów wzorcowych")
        
        best_match = matches[0]
        reference_img = best_match['image']
        similarity = best_match['similarity']
        
        print(f"✅ Znaleziono dopasowanie: {best_match['path'].name}")
        print(f"   Podobieństwo: {similarity:.2%}")
        
        # Wyrównaj obrazy
        if use_alignment:
            print("⚙️ Wyrównywanie obrazów...")
            aligned_img, transform = self.aligner.align_images(reference_img, img)
        else:
            aligned_img = cv2.resize(img, (reference_img.shape[1], reference_img.shape[0]))
            transform = None
        
        # Wykryj anomalie
        print("🔬 Wykrywanie anomalii...")
        detection_result = self.detector.detect_anomalies(
            reference_img, aligned_img, use_ssim=use_ssim
        )
        
        print(f"{'❌' if detection_result['has_anomaly'] else '✅'} "
              f"Wykryto {detection_result['anomaly_count']} anomalii")
        
        # Generuj raport
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = Path(image_path).stem
            report_path = self.output_dir / f"report_{img_name}_{timestamp}.png"
            
            print(f"📊 Generowanie raportu...")
            AnomalyReportGenerator.generate_report(
                img, reference_img, aligned_img, detection_result,
                str(report_path),
                metadata={
                    'input_image': image_path,
                    'reference_image': str(best_match['path']),
                    'similarity': similarity,
                    'alignment_used': use_alignment,
                    'ssim_used': use_ssim
                }
            )
            print(f"💾 Raport zapisany: {report_path}")
        
        return {
            'has_anomaly': detection_result['has_anomaly'],
            'anomaly_count': detection_result['anomaly_count'],
            'anomalies': detection_result['anomalies'],
            'reference_match': str(best_match['path']),
            'similarity': similarity,
            'ssim_score': detection_result.get('ssim_score'),
            'report_path': str(report_path) if save_report else None
        }
    
    def batch_process(self, image_dir: str, pattern: str = '*.bmp') -> List[Dict]:
        """
        Przetwarzaj wiele obrazów w partii
        
        Args:
            image_dir: Katalog z obrazami do sprawdzenia
            pattern: Wzorzec nazw plików
            
        Returns:
            Lista wyników dla każdego obrazu
        """
        image_paths = list(Path(image_dir).rglob(pattern))
        print(f"\n📦 Przetwarzanie partiami: {len(image_paths)} obrazów")
        
        results = []
        for img_path in image_paths:
            try:
                result = self.process_image(str(img_path))
                results.append(result)
            except Exception as e:
                print(f"❌ Błąd przetwarzania {img_path}: {e}")
                results.append({'error': str(e), 'path': str(img_path)})
        
        # Podsumowanie
        anomaly_count = sum(1 for r in results if r.get('has_anomaly', False))
        print(f"\n📈 Podsumowanie przetwarzania partii:")
        print(f"   Przetworzono: {len(results)} obrazów")
        print(f"   Z anomaliami: {anomaly_count}")
        print(f"   Bez anomalii: {len(results) - anomaly_count}")
        
        return results


# Funkcja pomocnicza do szybkiego użycia
def quick_detect(image_path: str, reference_dir: str = 'data/czyste',
                output_dir: str = 'anomaly_reports') -> Dict:
    """
    Szybka detekcja anomalii dla pojedynczego obrazu
    
    Args:
        image_path: Ścieżka do obrazu
        reference_dir: Katalog z obrazami wzorcowymi
        output_dir: Katalog do zapisywania raportów
        
    Returns:
        Wyniki detekcji
    """
    system = RTGAnomalySystem(reference_dir, output_dir)
    return system.process_image(image_path)


if __name__ == "__main__":
    # Przykład użycia
    print("=" * 80)
    print("🔬 System Detekcji Anomalii RTG")
    print("=" * 80)
    
    # Inicjalizuj system
    system = RTGAnomalySystem(
        reference_dir='data/czyste',
        output_dir='anomaly_reports'
    )
    
    # Testuj na obrazach z anomaliami
    test_dir = 'data/brudne'
    if os.path.exists(test_dir):
        results = system.batch_process(test_dir, pattern='*.bmp')
        
        # Wyświetl podsumowanie
        print("\n" + "=" * 80)
        print("📊 SZCZEGÓŁOWE WYNIKI")
        print("=" * 80)
        for i, result in enumerate(results, 1):
            if 'error' not in result:
                print(f"\n{i}. {'🔴 ANOMALIA' if result['has_anomaly'] else '🟢 CZYSTE'}")
                print(f"   Wykryto: {result['anomaly_count']} anomalii")
                print(f"   Podobieństwo do wzorca: {result['similarity']:.2%}")
                if result.get('ssim_score'):
                    print(f"   SSIM: {result['ssim_score']:.4f}")
    else:
        print(f"⚠️ Katalog testowy nie istnieje: {test_dir}")
        print("Użyj: python anomaly_detector.py lub quick_detect('path/to/image.bmp')")
