import React, { useEffect } from 'react';
import { AlertTriangle, CheckCircle, Download, FileText } from 'lucide-react';
import apiService from '../services/apiService';

const ResultsPanel = ({ results, uploadedFile, onAnalysisComplete, isAnalyzing }) => {
  // Real analysis using API instead of mock
  useEffect(() => {
    if (isAnalyzing && uploadedFile) {
      const performAnalysis = async () => {
        try {
          // Use real API service
          const analysisResults = await apiService.uploadAndAnalyze(uploadedFile);
          onAnalysisComplete(analysisResults);
        } catch (error) {
          console.error('Analysis failed:', error);
          // On error, show error message
          onAnalysisComplete({
            analysis_complete: true,
            detection_count: 0,
            detections: [],
            error: error.message,
            timestamp: new Date().toISOString()
          });
        }
      };

      performAnalysis();
    }
  }, [isAnalyzing, uploadedFile, onAnalysisComplete]);

  if (!uploadedFile) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Wyniki analizy
        </h3>
        <div className="text-center text-gray-500 py-8">
          <FileText className="h-12 w-12 mx-auto mb-4 text-gray-300" />
          <p>Załaduj obraz aby rozpocząć analizę</p>
        </div>
      </div>
    );
  }

  if (isAnalyzing) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
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
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Wyniki analizy
        </h3>
        <div className="text-center text-gray-500 py-8">
          <button
            className="bg-primary-600 hover:bg-primary-700 text-white font-medium py-3 px-6 rounded-lg transition-colors"
            onClick={() => {/* This would trigger analysis */}}
          >
            Rozpocznij analizę
          </button>
        </div>
      </div>
    );
  }

  const hasAnomalies = results.detection_count > 0;

  return (
    <div className="space-y-6">
      {/* Summary Card */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Podsumowanie analizy
          </h3>
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
            hasAnomalies 
              ? 'bg-orange-100 text-orange-800' 
              : 'bg-green-100 text-green-800'
          }`}>
            {hasAnomalies ? (
              <AlertTriangle className="h-4 w-4" />
            ) : (
              <CheckCircle className="h-4 w-4" />
            )}
            <span>
              {hasAnomalies ? 'Wykryto anomalie' : 'Brak anomalii'}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">
              {results.detection_count}
            </div>
            <div className="text-sm text-gray-600">Wykryte anomalie</div>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="text-2xl font-bold text-gray-900">
              {results.detections.length > 0 
                ? Math.max(...results.detections.map(d => d.confidence)).toFixed(2)
                : '0.00'
              }
            </div>
            <div className="text-sm text-gray-600">Najwyższa pewność</div>
          </div>
        </div>
      </div>

      {/* Detections List */}
      {hasAnomalies && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">
            Szczegóły wykryć
          </h4>
          
          <div className="space-y-3">
            {results.detections.map((detection, index) => (
              <div 
                key={index}
                className="detection-box flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    detection.confidence > 0.7 ? 'bg-red-500' : 
                    detection.confidence > 0.4 ? 'bg-orange-500' : 'bg-yellow-500'
                  }`}></div>
                  <div>
                    <div className="font-medium text-gray-900">
                      {detection.class || 'Anomalia'} #{detection.id || index + 1}
                    </div>
                    <div className="text-sm text-gray-600">
                      Pozycja: ({detection.bbox[0]}, {detection.bbox[1]})
                      {detection.area && ` • Obszar: ${detection.area}px²`}
                    </div>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="font-medium text-gray-900">
                    {(detection.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">pewność</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4">
          Akcje
        </h4>
        
        <div className="space-y-3">
          <button className="w-full flex items-center justify-center space-x-2 bg-primary-600 hover:bg-primary-700 text-white font-medium py-3 px-4 rounded-lg transition-colors">
            <Download className="h-4 w-4" />
            <span>Pobierz raport PDF</span>
          </button>
          
          <button className="w-full flex items-center justify-center space-x-2 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-3 px-4 rounded-lg transition-colors">
            <FileText className="h-4 w-4" />
            <span>Eksportuj dane JSON</span>
          </button>
        </div>
        
        <div className="mt-4 pt-4 border-t text-xs text-gray-500">
          <p>Analiza zakończona: {new Date(results.timestamp).toLocaleString('pl-PL')}</p>
          <p>Model: YOLOv8 RTG Anomaly Detector</p>
        </div>
      </div>
    </div>
  );
};

export default ResultsPanel;
