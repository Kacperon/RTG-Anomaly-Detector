import React, { useState } from 'react';
import { ZoomIn, ZoomOut, Download, FileImage } from 'lucide-react';

const ImageViewer = ({ uploadedFile, analysisResults, isAnalyzing }) => {
  const [zoom, setZoom] = useState(1);
  const [showOriginal, setShowOriginal] = useState(true);

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.2, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.2, 0.5));
  const resetZoom = () => setZoom(1);

  const getImageSrc = () => {
    if (analysisResults && !showOriginal) {
      return `data:image/png;base64,${analysisResults.annotated_image}`;
    }
    if (uploadedFile) {
      return uploadedFile.preview;
    }
    return null;
  };

  const imageSrc = getImageSrc();

  return (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden">
      <div className="p-4 border-b bg-gray-50">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">
            Podgląd obrazu
          </h3>
          
          {uploadedFile && (
            <div className="flex items-center space-x-2">
              {analysisResults && (
                <div className="flex bg-white rounded-lg border">
                  <button
                    onClick={() => setShowOriginal(true)}
                    className={`px-3 py-1 text-sm rounded-l-lg transition-colors ${
                      showOriginal 
                        ? 'bg-primary-600 text-white' 
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    Oryginalny
                  </button>
                  <button
                    onClick={() => setShowOriginal(false)}
                    className={`px-3 py-1 text-sm rounded-r-lg transition-colors ${
                      !showOriginal 
                        ? 'bg-primary-600 text-white' 
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    Z anomaliami
                  </button>
                </div>
              )}
              
              <div className="flex bg-white rounded-lg border">
                <button
                  onClick={handleZoomOut}
                  className="p-2 text-gray-600 hover:bg-gray-100 rounded-l-lg transition-colors"
                  title="Pomniejsz"
                >
                  <ZoomOut className="h-4 w-4" />
                </button>
                <button
                  onClick={resetZoom}
                  className="px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 transition-colors border-x"
                  title="Resetuj zoom"
                >
                  {Math.round(zoom * 100)}%
                </button>
                <button
                  onClick={handleZoomIn}
                  className="p-2 text-gray-600 hover:bg-gray-100 rounded-r-lg transition-colors"
                  title="Powiększ"
                >
                  <ZoomIn className="h-4 w-4" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="relative bg-gray-100" style={{ height: '700px' }}>
        {!uploadedFile ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="w-20 h-20 bg-gray-200 rounded-lg mx-auto mb-4 flex items-center justify-center">
                <FileImage className="h-10 w-10 text-gray-400" />
              </div>
              <p className="text-gray-500 text-lg">Brak załadowanego obrazu</p>
              <p className="text-gray-400 text-sm mt-2">Prześlij obraz RTG aby rozpocząć</p>
            </div>
          </div>
        ) : (
          <div className="absolute inset-0 overflow-auto p-4">
            <div className="flex items-center justify-center h-full">
              {isAnalyzing && showOriginal ? (
                <div className="relative">
                  <img
                    src={imageSrc}
                    alt="RTG scan"
                    className="max-w-full max-h-full object-contain rounded shadow-lg opacity-50"
                    style={{ transform: `scale(${zoom})` }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="bg-black bg-opacity-50 rounded-lg p-4">
                      <svg className="animate-spin h-8 w-8 text-white mx-auto mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <p className="text-white text-sm">Analizowanie...</p>
                    </div>
                  </div>
                </div>
              ) : (
                <img
                  src={imageSrc}
                  alt="RTG scan"
                  className="max-w-full max-h-full object-contain rounded shadow-lg transition-transform"
                  style={{ transform: `scale(${zoom})` }}
                />
              )}
            </div>
          </div>
        )}
      </div>

      {uploadedFile && (
        <div className="p-4 bg-gray-50 border-t">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div>
              <span className="font-medium">{uploadedFile.name}</span>
              <span className="ml-2">
                ({(uploadedFile.size / 1024 / 1024).toFixed(2)} MB)
              </span>
            </div>
            
            {analysisResults && !showOriginal && (
              <button
                className="flex items-center space-x-1 text-primary-600 hover:text-primary-700"
                title="Pobierz obraz z anomaliami"
              >
                <Download className="h-4 w-4" />
                <span>Pobierz</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageViewer;
