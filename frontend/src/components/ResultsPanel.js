import React, { useEffect } from 'react';
import { AlertTriangle, CheckCircle, Download, FileText } from 'lucide-react';
import apiService from '../services/apiService';

const ResultsPanel = ({ results, uploadedFile, onAnalysisComplete, isAnalyzing }) => {
  // Usuwamy automatyczną analizę - będzie wykonana w App.js
  // useEffect do analizy jest teraz niepotrzebny

  if (!uploadedFile) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-6 overflow-hidden">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 break-words">
          Wyniki analizy
        </h3>
        
        {/* Pusta tabelka */}
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  ID
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Pewność
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Pozycja
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              <tr>
                <td colSpan="3" className="px-4 py-8 text-center text-gray-400 text-sm">
                  Brak danych - załaduj obraz i rozpocznij analizę
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  if (isAnalyzing) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-6 overflow-hidden">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 break-words">
          Wyniki analizy
        </h3>
        <div className="text-center py-8">
          <svg className="animate-spin h-12 w-12 text-primary-600 mx-auto mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <p className="text-lg font-medium text-gray-900">Analizowanie obrazu...</p>
          <p className="text-sm text-gray-500 mt-2">
            Proces analizy może potrwać kilka sekund
          </p>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-6 overflow-hidden">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 break-words">
          Wyniki analizy
        </h3>
        
        {/* Pusta tabelka z nagłówkiem */}
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  ID
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Pewność
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Pozycja
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              <tr>
                <td colSpan="3" className="px-4 py-8 text-center text-gray-400 text-sm">
                  Oczekiwanie na analizę...
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        
        <div className="mt-4 text-center text-sm text-gray-500">
          <p>Kliknij "Rozpocznij analizę" aby wykryć anomalie</p>
        </div>
      </div>
    );
  }

  const hasAnomalies = results.detection_count > 0;
  const detections = results.detections || results.anomalies || [];
  
  console.log('=== ResultsPanel Debug ===');
  console.log('results:', results);
  console.log('hasAnomalies:', hasAnomalies);
  console.log('detection_count:', results.detection_count);
  console.log('detections array:', detections);
  console.log('detections length:', detections.length);
  console.log('=========================');

  // Funkcja pobierania raportu
  const downloadReport = async (format) => {
    try {
      if (format === 'image' && results.annotated_image) {
        // Pobierz obraz z adnotacjami
        const link = document.createElement('a');
        link.href = `data:image/png;base64,${results.annotated_image}`;
        link.download = `raport_rtg_${new Date().getTime()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } else if (format === 'json') {
        // Pobierz dane JSON
        const dataStr = JSON.stringify(results, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `raport_rtg_${new Date().getTime()}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Błąd pobierania:', error);
      alert('Błąd podczas pobierania raportu');
    }
  };

  return (
    <div className="space-y-6 overflow-hidden">
      {/* Summary Card */}
      <div className="bg-white rounded-lg shadow-sm p-6 overflow-hidden">
        <div className="flex items-center justify-between mb-4 min-w-0">
          <h3 className="text-lg font-semibold text-gray-900 break-words">
            Podsumowanie
          </h3>
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium flex-shrink-0 ${
            hasAnomalies 
              ? 'bg-red-100 text-red-800' 
              : 'bg-green-100 text-green-800'
          }`}>
            {hasAnomalies ? (
              <AlertTriangle className="h-4 w-4" />
            ) : (
              <CheckCircle className="h-4 w-4" />
            )}
            <span className="whitespace-nowrap">
              {hasAnomalies ? 'Wykryto anomalie' : 'Brak anomalii'}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3">
          <div className="bg-gray-50 rounded-lg p-4 text-center">
            <div className="text-3xl font-bold text-gray-900">
              {results.detection_count}
            </div>
            <div className="text-sm text-gray-600 mt-1">Wykryte anomalie</div>
          </div>
          
          {detections && detections.length > 0 && (
            <div className="bg-gray-50 rounded-lg p-4 text-center">
              <div className="text-3xl font-bold text-blue-600">
                {Math.max(...detections.map(d => d.confidence || 0.5)).toFixed(2)}
              </div>
              <div className="text-sm text-gray-600 mt-1">Maksymalna pewność</div>
            </div>
          )}

          {/* Informacja o metodzie analizy i heatmapie */}
          {results.method === 'comparison_based' && (
            <div className="bg-blue-50 rounded-lg p-4 text-center">
              <div className="text-sm font-medium text-blue-800">
                {results.heatmap_image ? '🔥 Z Heatmapą' : '📊 Analiza Porównawcza'}
              </div>
              <div className="text-xs text-blue-600 mt-1">
                {results.ssim_score ? `SSIM: ${results.ssim_score.toFixed(3)}` : 'Porównanie wzorcowe'}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Results Table */}
      <div className="bg-white rounded-lg shadow-sm p-6 overflow-hidden">
        <h4 className="text-lg font-semibold text-gray-900 mb-4 break-words">
          Wykryte anomalie
        </h4>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  ID
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Pewność
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Pozycja (X, Y)
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {detections && detections.length > 0 ? (
                detections.map((detection, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                      #{detection.id || index + 1}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm">
                      <div className="flex items-center">
                        <div className={`w-2 h-2 rounded-full mr-2 ${
                          (detection.confidence || 0.5) > 0.7 ? 'bg-red-500' : 
                          (detection.confidence || 0.5) > 0.4 ? 'bg-orange-500' : 'bg-yellow-500'
                        }`}></div>
                        <span className="font-medium text-gray-900">
                          {((detection.confidence || 0.5) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                      {detection.bbox && detection.bbox.length >= 2 ? (
                        `(${detection.bbox[0]}, ${detection.bbox[1]})`
                      ) : detection.center && detection.center.length >= 2 ? (
                        `(${detection.center[0]}, ${detection.center[1]})`
                      ) : (
                        '(N/A, N/A)'
                      )}
                      {detection.area && (
                        <span className="text-xs text-gray-400 ml-2">
                          {detection.area}px²
                        </span>
                      )}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="3" className="px-4 py-8 text-center text-gray-500">
                    <CheckCircle className="h-12 w-12 mx-auto mb-2 text-green-400" />
                    <p className="font-medium">Nie wykryto żadnych anomalii</p>
                    <p className="text-sm text-gray-400 mt-1">Obraz wydaje się być czysty</p>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Download Actions */}
      <div className="bg-white rounded-lg shadow-sm p-6 overflow-hidden">
        <h4 className="text-lg font-semibold text-gray-900 mb-4 break-words">
          Pobierz raport
        </h4>
        
        <div className="space-y-3">
          <button 
            onClick={() => downloadReport('image')}
            className="w-full flex items-center justify-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
          >
            <Download className="h-5 w-5" />
            <span>Pobierz obraz z adnotacjami</span>
          </button>
          
          <button 
            onClick={() => downloadReport('json')}
            className="w-full flex items-center justify-center space-x-2 bg-gray-600 hover:bg-gray-700 text-white font-medium py-3 px-4 rounded-lg transition-colors"
          >
            <FileText className="h-5 w-5" />
            <span>Pobierz dane JSON</span>
          </button>
        </div>
        
        <div className="mt-4 pt-4 border-t text-xs text-gray-500">
          <p className="flex items-center justify-between">
            <span>Analiza zakończona:</span>
            <span className="font-medium">{new Date(results.timestamp).toLocaleString('pl-PL')}</span>
          </p>
          <p className="flex items-center justify-between mt-1">
            <span>Model:</span>
            <span className="font-medium">YOLOv8 RTG</span>
          </p>
        </div>
      </div>
    </div>
  );
};

export default ResultsPanel;
