import React, { useState, useRef, useEffect } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Download, FileImage, Maximize, X } from 'lucide-react';

const ImageViewer = ({ uploadedFile, analysisResults, isAnalyzing }) => {
  const [showOriginal, setShowOriginal] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // Fullscreen state
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0, startX: 0, startY: 0 });

  // Automatycznie przełącz na widok z anomaliami gdy raport jest gotowy
  useEffect(() => {
    if (analysisResults && analysisResults.annotated_image) {
      setShowOriginal(false);
    }
  }, [analysisResults]);

  // Reset przy wyjściu z fullscreen
  useEffect(() => {
    if (!isFullscreen) {
      setZoom(1);
      setPosition({ x: 0, y: 0 });
    }
  }, [isFullscreen]);

  // Escape zamyka fullscreen
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isFullscreen]);

  // Mouse drag dla fullscreen - przepisane od nowa
  useEffect(() => {
    if (!isFullscreen) return;

    const handleMouseMove = (e) => {
      if (!isDragging) return;
      
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      
      setPosition({
        x: dragStart.startX + dx,
        y: dragStart.startY + dy,
      });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    if (isDragging) {
      document.body.style.cursor = 'grabbing';
      document.body.style.userSelect = 'none';
    }
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      if (isDragging) {
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
    };
  }, [isDragging, dragStart, isFullscreen]);

  const handleMouseDown = (e) => {
    // Ignoruj kliknięcia na przyciskach
    if (e.target.closest('button')) return;
    
    e.preventDefault();
    
    setDragStart({
      x: e.clientX,
      y: e.clientY,
      startX: position.x,
      startY: position.y,
    });
    setIsDragging(true);
  };

  const handleWheel = (e) => {
    e.preventDefault();
    
    // Oblicz nowy zoom
    const delta = e.deltaY > 0 ? -0.25 : 0.25;
    const newZoom = Math.min(Math.max(zoom + delta, 0.25), 8);
    
    if (newZoom <= 1) {
      // Wyzeruj pozycję przy zoom 1.0 lub mniejszym
      setZoom(newZoom);
      setPosition({ x: 0, y: 0 });
    } else {
      // Zoom względem pozycji myszy (jak w Windows)
      const rect = e.currentTarget.getBoundingClientRect();
      const mouseX = e.clientX - rect.left - rect.width / 2;
      const mouseY = e.clientY - rect.top - rect.height / 2;
      
      const zoomFactor = newZoom / zoom;
      
      setPosition(prev => ({
        x: mouseX - (mouseX - prev.x) * zoomFactor,
        y: mouseY - (mouseY - prev.y) * zoomFactor,
      }));
      
      setZoom(newZoom);
    }
  };

  const resetView = () => {
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  };

  const getImageSrc = () => {
    if (analysisResults && !showOriginal) {
      return `data:image/png;base64,${analysisResults.annotated_image}`;
    }
    if (uploadedFile) {
      return uploadedFile.preview;
    }
    return null;
  };

  const handleDownload = () => {
    if (analysisResults && !showOriginal) {
      const link = document.createElement('a');
      link.href = `data:image/png;base64,${analysisResults.annotated_image}`;
      link.download = `anomalie_${uploadedFile?.name || 'obraz'}.png`;
      link.click();
    }
  };

  const imageSrc = getImageSrc();

  // Przełącznik oryginalny/z anomaliami
  const ViewToggle = ({ dark = false }) => (
    analysisResults && (
      <div className={`flex rounded-lg overflow-hidden ${dark ? 'bg-black/70' : 'bg-white border shadow-sm'}`}>
        <button
          onClick={() => setShowOriginal(true)}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            showOriginal 
              ? 'bg-primary-600 text-white' 
              : dark ? 'text-white hover:bg-white/20' : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          Oryginalny
        </button>
        <button
          onClick={() => setShowOriginal(false)}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            !showOriginal 
              ? 'bg-primary-600 text-white' 
              : dark ? 'text-white hover:bg-white/20' : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          Z anomaliami
        </button>
      </div>
    )
  );

  // ==================== FULLSCREEN ====================
  if (isFullscreen) {
    return (
      <div className="fixed inset-0 z-50 bg-black">
        {/* Toolbar na górze */}
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-4">
          {/* Zoom controls */}
          <div className="flex items-center gap-2 bg-black/70 text-white rounded-lg p-2">
            <button
              onClick={() => {
                const newZoom = Math.max(zoom - 0.25, 0.25);
                if (newZoom <= 1) {
                  setZoom(newZoom);
                  setPosition({ x: 0, y: 0 });
                } else {
                  setZoom(newZoom);
                }
              }}
              className="p-2 rounded-lg hover:bg-white/20 transition-colors"
            >
              <ZoomOut className="h-5 w-5" />
            </button>
            <button
              onClick={resetView}
              className="px-3 py-1.5 text-sm font-medium rounded-lg hover:bg-white/20 transition-colors min-w-[60px]"
            >
              {Math.round(zoom * 100)}%
            </button>
            <button
              onClick={() => setZoom(prev => Math.min(prev + 0.25, 8))}
              className="p-2 rounded-lg hover:bg-white/20 transition-colors"
            >
              <ZoomIn className="h-5 w-5" />
            </button>
            <div className="w-px h-6 bg-white/30" />
            <button
              onClick={resetView}
              className="p-2 rounded-lg hover:bg-white/20 transition-colors"
            >
              <RotateCcw className="h-5 w-5" />
            </button>
          </div>
          
          <ViewToggle dark />
        </div>

        {/* Przycisk zamknij */}
        <button
          onClick={() => setIsFullscreen(false)}
          className="absolute top-4 right-4 z-20 p-3 bg-black/70 hover:bg-black/90 text-white rounded-lg transition-colors"
        >
          <X className="h-6 w-6" />
        </button>

        {/* Obraz */}
        <div 
          className="w-full h-full flex items-center justify-center overflow-hidden"
          onMouseDown={handleMouseDown}
          onWheel={handleWheel}
          style={{ 
            cursor: isDragging ? 'grabbing' : 'grab',
            userSelect: 'none',
          }}
        >
          <img
            src={imageSrc}
            alt="RTG scan"
            draggable={false}
            style={{ 
              transform: `translate(${position.x}px, ${position.y}px) scale(${zoom})`,
              maxWidth: '100%',
              maxHeight: '100%',
              transition: isDragging ? 'none' : 'transform 0.15s ease-out',
              pointerEvents: 'none',
              transformOrigin: 'center center',
            }}
          />
        </div>

        {/* Info */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 bg-black/70 text-white text-sm px-4 py-2 rounded-lg">
          Scroll = zoom • Przeciągnij = przesuń • Esc = zamknij
        </div>
      </div>
    );
  }

  // ==================== NORMALNY WIDOK ====================
  return (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b bg-gray-50">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Podgląd obrazu</h3>
          
          {uploadedFile && (
            <div className="flex items-center gap-3">
              <ViewToggle />
              <button
                onClick={() => setIsFullscreen(true)}
                className="p-2 text-gray-600 hover:bg-gray-200 rounded-lg transition-colors"
                title="Pełny ekran"
              >
                <Maximize className="h-5 w-5" />
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Obszar obrazu */}
      <div className="relative bg-gray-900" style={{ height: '600px' }}>
        {!uploadedFile ? (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
            <div className="text-center">
              <div className="w-24 h-24 bg-gray-200 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                <FileImage className="h-12 w-12 text-gray-400" />
              </div>
              <p className="text-gray-500 text-lg font-medium">Brak załadowanego obrazu</p>
              <p className="text-gray-400 text-sm mt-2">Prześlij obraz RTG aby rozpocząć analizę</p>
            </div>
          </div>
        ) : (
          <div className="w-full h-full flex items-center justify-center p-4">
            <div className="relative">
              {isAnalyzing && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10 rounded-lg">
                  <div className="text-center">
                    <svg className="animate-spin h-12 w-12 text-white mx-auto mb-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <p className="text-white font-medium">Analizowanie obrazu...</p>
                  </div>
                </div>
              )}
              <img
                src={imageSrc}
                alt="RTG scan"
                className="max-w-full max-h-[560px] object-contain rounded-lg shadow-2xl"
                draggable={false}
              />
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      {uploadedFile && (
        <div className="p-4 bg-gray-50 border-t">
          <div className="flex items-center justify-between text-sm">
            <div className="text-gray-600">
              <span className="font-medium">{uploadedFile.name}</span>
              <span className="ml-2 text-gray-400">
                ({(uploadedFile.size / 1024 / 1024).toFixed(2)} MB)
              </span>
            </div>
            
            {analysisResults && !showOriginal && (
              <button
                onClick={handleDownload}
                className="flex items-center gap-2 px-3 py-1.5 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
              >
                <Download className="h-4 w-4" />
                <span>Pobierz wynik</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageViewer;
