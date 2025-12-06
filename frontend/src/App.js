import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import UploadArea from './components/UploadArea';
import ImageViewer from './components/ImageViewer';
import ResultsPanel from './components/ResultsPanel';
import StatusPanel from './components/StatusPanel';
import apiService from './services/apiService';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [statusMessages, setStatusMessages] = useState([]);

  // Check backend health on component mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        await apiService.healthCheck();
        addStatusMessage('Połączono z backendem', 'success');
      } catch (error) {
        addStatusMessage('Błąd połączenia z backendem', 'error');
      }
    };

    checkBackendHealth();
  }, []);

  const addStatusMessage = (message, type = 'info') => {
    const newMessage = {
      id: Date.now(),
      text: message,
      type,
      timestamp: new Date().toLocaleTimeString()
    };
    setStatusMessages(prev => [...prev, newMessage]);
  };

  const handleFileUpload = (file) => {
    setUploadedFile(file);
    setAnalysisResults(null);
    addStatusMessage(`Plik ${file.name} został załadowany`, 'success');
  };

  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results);
    setIsAnalyzing(false);
    
    if (results.error) {
      addStatusMessage(`Błąd analizy: ${results.error}`, 'error');
    } else {
      addStatusMessage(
        `Analiza zakończona. Wykryto ${results.detection_count} anomalii`, 
        results.detection_count > 0 ? 'warning' : 'success'
      );
    }
  };

  const startAnalysis = async () => {
    if (!modelLoaded) {
      addStatusMessage('Ładowanie modelu...', 'info');
      try {
        const modelResult = await apiService.loadModel();
        setModelLoaded(true);
        addStatusMessage(`Model załadowany: ${modelResult.model_type}`, 'success');
        setIsAnalyzing(true);
        addStatusMessage('Rozpoczynanie analizy...', 'info');
      } catch (error) {
        addStatusMessage(`Błąd ładowania modelu: ${error.message}`, 'error');
        return;
      }
    } else {
      setIsAnalyzing(true);
      addStatusMessage('Rozpoczynanie analizy...', 'info');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Panel - Upload and Controls */}
          <div className="lg:col-span-1 space-y-6">
            <UploadArea 
              onFileUpload={handleFileUpload}
              disabled={isAnalyzing}
            />
            
            {uploadedFile && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Kontrola analizy
                </h3>
                <button
                  onClick={startAnalysis}
                  disabled={isAnalyzing}
                  className="w-full bg-primary-600 hover:bg-primary-700 disabled:bg-gray-300 
                           text-white font-medium py-3 px-4 rounded-lg transition-colors"
                >
                  {isAnalyzing ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Analizowanie...
                    </span>
                  ) : (
                    'Rozpocznij analizę'
                  )}
                </button>
              </div>
            )}

            <StatusPanel 
              messages={statusMessages}
              modelLoaded={modelLoaded}
              isAnalyzing={isAnalyzing}
            />
          </div>

          {/* Center Panel - Image Viewer */}
          <div className="lg:col-span-1">
            <ImageViewer 
              uploadedFile={uploadedFile}
              analysisResults={analysisResults}
              isAnalyzing={isAnalyzing}
            />
          </div>

          {/* Right Panel - Results */}
          <div className="lg:col-span-1">
            <ResultsPanel 
              results={analysisResults}
              uploadedFile={uploadedFile}
              onAnalysisComplete={handleAnalysisComplete}
              isAnalyzing={isAnalyzing}
            />
          </div>
          
        </div>
      </main>
    </div>
  );
}

export default App;
