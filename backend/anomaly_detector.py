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
    
    def __init__(self, reference_dir: str, processed_dir: str = None):
        """
        Args:
            reference_dir: Katalog z obrazami wzorcowymi (czystymi)
            processed_dir: Dodatkowy katalog z przetworzonymi obrazami (opcjonalny)
        """
        self.reference_dir = Path(reference_dir)
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.reference_images = []
        self.reference_features = []
        self._load_references()
    
    def _load_references(self):
        """Ładuje wszystkie obrazy wzorcowe z głównego i dodatkowego katalogu"""
        print(f"📁 Ładowanie obrazów wzorcowych z: {self.reference_dir}")
        
        # Lista katalogów do przeszukania
        directories_to_search = [self.reference_dir]
        
        # Dodaj katalog z przetworzonymi obrazami jeśli istnieje
        if self.processed_dir and self.processed_dir.exists():
            directories_to_search.append(self.processed_dir)
            print(f"📁 Dodatkowo przeszukuję: {self.processed_dir}")
        
        # Przeszukaj wszystkie katalogi
        for search_dir in directories_to_search:
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    # Akceptuj .bmp i inne formaty obrazów, ignoruj pliki z 'czarno' w nazwie
                    if (file.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')) and 
                        'czarno' not in file.lower()):
                        
                        img_path = Path(root) / file
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # Oblicz features dla szybszego dopasowywania
                            features = self._extract_features(img)
                            self.reference_images.append({
                                'path': img_path,
                                'image': img,
                                'features': features,
                                'source': 'processed' if search_dir == self.processed_dir else 'original'
                            })
        
        print(f"✅ Załadowano {len(self.reference_images)} obrazów wzorcowych")
        
        # Podsumowanie źródeł
        original_count = sum(1 for img in self.reference_images if img['source'] == 'original')
        processed_count = sum(1 for img in self.reference_images if img['source'] == 'processed')
        print(f"   📂 Oryginalne: {original_count}")
        print(f"   🔧 Przetworzone: {processed_count}")
    
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
        print("🔧 Rozpoczynanie wyrównywania ECC...")
        
        # Konwertuj do float32
        ref_gray = reference.astype(np.float32)
        img_gray = image.astype(np.float32)
        
        # Zdefiniuj typ transformacji (affine)
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Kryteria zakończenia - bardziej liberalne dla szybkości
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-4)
        
        try:
            print("🔄 Wykonywanie findTransformECC...")
            # Wyrównaj obrazy z timeout
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, img_gray, warp_matrix, warp_mode, criteria, 
                inputMask=None, gaussFiltSize=3
            )
            
            print("✅ ECC zakończone pomyślnie")
            # Zastosuj transformację
            aligned = cv2.warpAffine(
                image, warp_matrix, (reference.shape[1], reference.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            
            return aligned, warp_matrix
            
        except Exception as e:
            print(f"⚠️ Wyrównanie ECC nie powiodło się: {e}, zwracam oryginalny obraz")
            # Fallback - prosta korekta rozmiaru
            if reference.shape != image.shape:
                aligned = cv2.resize(image, (reference.shape[1], reference.shape[0]))
            else:
                aligned = image.copy()
            return aligned, np.eye(2, 3, dtype=np.float32)
    
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
    
    def __init__(self, threshold: float = 25, min_area: int = 300, max_area: int = 50000,
                 background_threshold: int = 240):
        """
        Args:
            threshold: Próg różnicy pikseli
            min_area: Minimalna powierzchnia anomalii (piksele)
            max_area: Maksymalna powierzchnia anomalii (piksele)
            background_threshold: Próg dla wykrywania białego tła (0-255)
        """
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.background_threshold = background_threshold
    
    def detect_anomalies(self, reference: np.ndarray, image: np.ndarray,
                        use_ssim: bool = True, ignore_background: bool = True,
                        background_method: str = 'otsu') -> Dict:
        """
        Wykryj anomalie porównując dwa obrazy
        
        Args:
            reference: Obraz wzorcowy (czysty)
            image: Obraz do sprawdzenia
            use_ssim: Czy użyć SSIM zamiast prostej różnicy
            ignore_background: Czy ignorować białe tło podczas detekcji
            background_method: Metoda wykrywania tła ('otsu', 'adaptive', 'threshold')
            
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
        reference_for_mask = ref_processed if ignore_background else None
        anomalies = self._find_anomalies(diff_map, reference_for_mask)
        
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
    
    def _create_background_mask(self, img: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """
        Tworzy maskę tła dla obrazu RTG
        
        Args:
            img: Obraz w skali szarości
            method: Metoda wykrywania tła ('otsu', 'adaptive', 'threshold')
            
        Returns:
            Maska binarna (True dla obszarów nie-tła)
        """
        if method == 'otsu':
            # Otsu thresholding - automatyczne wykrywanie progu
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Inwersja - chcemy maskę obszarów nie-tła
            mask = binary > 0
            
        elif method == 'adaptive':
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            mask = binary > 0
            
        elif method == 'threshold':
            # Stały próg dla białego tła
            mask = img < self.background_threshold
            
        else:
            # Fallback - prosty próg
            mask = img < self.background_threshold
        
        # Operacje morfologiczne do oczyszczenia maski
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        return mask_cleaned.astype(bool)
    
    def _find_anomalies(self, diff_map: np.ndarray, reference_img: np.ndarray = None) -> List[Dict]:
        """Znajdź regiony anomalii na mapie różnic, ignorując prawie białe tło"""
        # Progowanie
        _, binary = cv2.threshold(diff_map, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Maskowanie tła - ignoruj prawie białe piksele (tło RTG)
        if reference_img is not None:
            # Utwórz maskę obszarów nie-tła (ROI - Region of Interest)
            roi_mask = self._create_background_mask(reference_img, method='otsu')
            
            # Zastosuj maskę - usuń anomalie w obszarach białego tła
            binary = binary & roi_mask.astype(np.uint8) * 255
            
            print(f"🎯 Zastosowano maskę ROI (obszary nie-tła)")
            print(f"   Procent obszaru ROI: {np.sum(roi_mask) / roi_mask.size * 100:.1f}%")
        else:
            print("⚠️ Brak obrazu referencyjnego - pomijam maskowanie tła")
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
                       output_path: str, metadata: Dict = None,
                       reference_info: Dict = None) -> str:
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
            diff_colored, annotated, detection_result, reference_info
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
                           annotated: np.ndarray, detection_result: Dict,
                           reference_info: Dict = None) -> np.ndarray:
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
        
        # Dodaj etykiety z informacjami o dopasowaniu
        def add_label(img, text, color=(255, 255, 255)):
            labeled = img.copy()
            cv2.rectangle(labeled, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(labeled, text, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            return labeled
        
        # Przygotuj etykiety z dodatkowymi informacjami
        reference_label = "Obraz wzorcowy"
        if reference_info:
            ref_name = reference_info.get('name', 'nieznany')
            ref_similarity = reference_info.get('similarity', 0)
            ref_source = reference_info.get('source', 'original')
            source_emoji = "🔧" if ref_source == 'processed' else "📁"
            reference_label = f"{source_emoji} {ref_name[:20]} ({ref_similarity:.1%})"
        
        original = add_label(original, "Obraz testowy")
        reference = add_label(reference, reference_label, (0, 255, 255))  # Cyan dla wyróżnienia
        aligned = add_label(aligned, "Wyrownany")
        diff_colored = add_label(diff_colored, "Mapa roznic (heatmap)")
        annotated = add_label(annotated, f"Wykryte anomalie: {len(detection_result['anomalies'])}", 
                            (0, 255, 0) if detection_result['has_anomaly'] else (255, 255, 255))
        
        # Grid 2x3 - lepsze ułożenie
        row1 = np.hstack([original, reference, aligned])
        row2 = np.hstack([diff_colored, annotated, np.zeros_like(annotated)])  # Trzecia kolumna pusta
        
        grid = np.vstack([row1, row2])
        
        # Dodaj szczegółowe podsumowanie na dole
        summary_height = 120
        summary = np.zeros((summary_height, grid.shape[1], 3), dtype=np.uint8)
        
        anomaly_count = len(detection_result['anomalies'])
        status = "ANOMALIA WYKRYTA!" if detection_result['has_anomaly'] else "BRAK ANOMALII"
        status_color = (0, 0, 255) if detection_result['has_anomaly'] else (0, 255, 0)
        
        # Linia 1: Status
        cv2.putText(summary, status, (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        # Linia 2: Liczba anomalii
        cv2.putText(summary, f"Liczba anomalii: {anomaly_count}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Linia 3: Informacje o dopasowaniu
        if reference_info:
            match_text = f"Dopasowanie: {reference_info.get('name', 'nieznany')[:30]}"
            cv2.putText(summary, match_text, (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Po prawej: SSIM
        if detection_result.get('ssim_score') is not None:
            cv2.putText(summary, f"SSIM: {detection_result['ssim_score']:.4f}", 
                       (grid.shape[1] - 200, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Po prawej: Podobieństwo
        if reference_info:
            similarity_text = f"Podobienstwo: {reference_info.get('similarity', 0):.1%}"
            cv2.putText(summary, similarity_text,
                       (grid.shape[1] - 200, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
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
    
    def __init__(self, reference_dir: str, output_dir: str = 'anomaly_reports', 
                 processed_dir: str = None):
        """
        Args:
            reference_dir: Katalog z obrazami wzorcowymi (czystymi)
            output_dir: Katalog do zapisywania raportów
            processed_dir: Katalog z przetworzonymi obrazami (opcjonalny)
        """
        # Automatycznie dodaj katalog z przetworzonymi obrazami jeśli nie podano
        if processed_dir is None:
            processed_dir = 'data-processing/processed_clean_data'
        
        self.matcher = ImageMatcher(reference_dir, processed_dir)
        self.aligner = ImageAligner()
        self.detector = AnomalyDetector()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🚀 System detekcji anomalii RTG zainicjalizowany")
        print(f"📂 Obrazy wzorcowe: {reference_dir}")
        if processed_dir:
            print(f"🔧 Przetworzone: {processed_dir}")
        print(f"📂 Raporty: {output_dir}")
    
    def process_image(self, image_path: str, use_alignment: bool = True,
                     use_ssim: bool = True, save_report: bool = True,
                     ignore_background: bool = True) -> Dict:
        """
        Przetwórz obraz i wykryj anomalie
        
        Args:
            image_path: Ścieżka do obrazu do sprawdzenia
            use_alignment: Czy wyrównywać obrazy
            use_ssim: Czy użyć SSIM
            save_report: Czy zapisać raport
            ignore_background: Czy ignorować białe tło podczas detekcji
            
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
            try:
                # Spróbuj szybkiego wyrównania najpierw
                if reference_img.shape != img.shape:
                    print("📐 Dopasowywanie rozmiarów...")
                    img_resized = cv2.resize(img, (reference_img.shape[1], reference_img.shape[0]))
                else:
                    img_resized = img
                
                # Sprawdź czy obrazy są podobne - jeśli bardzo podobne, pomiń ECC
                similarity_quick = cv2.matchTemplate(reference_img.astype(np.float32), 
                                                   img_resized.astype(np.float32), 
                                                   cv2.TM_CCOEFF_NORMED)[0,0]
                
                if similarity_quick > 0.95:
                    print(f"🚀 Obrazy bardzo podobne ({similarity_quick:.3f}), pomijam dokładne wyrównywanie")
                    aligned_img = img_resized
                    transform = np.eye(2, 3, dtype=np.float32)
                else:
                    print(f"🔧 Podobieństwo: {similarity_quick:.3f}, uruchamiam dokładne wyrównywanie...")
                    aligned_img, transform = self.aligner.align_images(reference_img, img_resized)
                    
            except Exception as e:
                print(f"❌ Błąd podczas wyrównywania: {e}")
                print("🔄 Używam podstawowego dopasowania rozmiaru...")
                aligned_img = cv2.resize(img, (reference_img.shape[1], reference_img.shape[0]))
                transform = None
        else:
            aligned_img = cv2.resize(img, (reference_img.shape[1], reference_img.shape[0]))
            transform = None
        
        # Wykryj anomalie
        print("🔬 Wykrywanie anomalii...")
        detection_result = self.detector.detect_anomalies(
            reference_img, aligned_img, use_ssim=use_ssim, 
            ignore_background=ignore_background
        )
        
        print(f"{'❌' if detection_result['has_anomaly'] else '✅'} "
              f"Wykryto {detection_result['anomaly_count']} anomalii")
        
        # Generuj raport
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = Path(image_path).stem
            report_path = self.output_dir / f"report_{img_name}_{timestamp}.png"
            
            print(f"📊 Generowanie raportu...")
            
            # Przygotuj informacje o dopasowaniu dla raportu
            reference_info = {
                'name': best_match['path'].name,
                'similarity': similarity,
                'source': best_match.get('source', 'original'),
                'path': str(best_match['path'])
            }
            
            AnomalyReportGenerator.generate_report(
                img, reference_img, aligned_img, detection_result,
                str(report_path),
                metadata={
                    'input_image': image_path,
                    'reference_image': str(best_match['path']),
                    'reference_source': best_match.get('source', 'original'),
                    'similarity': similarity,
                    'alignment_used': use_alignment,
                    'ssim_used': use_ssim
                },
                reference_info=reference_info
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
                output_dir: str = 'anomaly_reports', 
                processed_dir: str = 'data-processing/processed_clean_data') -> Dict:
    """
    Szybka detekcja anomalii dla pojedynczego obrazu
    
    Args:
        image_path: Ścieżka do obrazu
        reference_dir: Katalog z obrazami wzorcowymi
        output_dir: Katalog do zapisywania raportów
        processed_dir: Katalog z przetworzonymi obrazami
        
    Returns:
        Wyniki detekcji
    """
    system = RTGAnomalySystem(reference_dir, output_dir, processed_dir)
    return system.process_image(image_path)


if __name__ == "__main__":
    # Przykład użycia
    print("=" * 80)
    print("🔬 System Detekcji Anomalii RTG")
    print("=" * 80)
    
    # Inicjalizuj system
    system = RTGAnomalySystem(
        reference_dir='data/czyste',
        output_dir='anomaly_reports',
        processed_dir='data-processing/processed_clean_data'
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
