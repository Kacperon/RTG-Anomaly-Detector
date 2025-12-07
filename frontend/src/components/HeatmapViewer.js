import React, { useState, useRef, useEffect } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Download, Maximize, X, Thermometer, Image as ImageIcon, BarChart3 } from 'lucide-react';

const HeatmapViewer = ({ colorComparisonResults, uploadedFile, isAnalyzing }) => {
  const [currentView, setCurrentView] = useState('original'); // original, region, heatmap, comparison
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // Fullscreen state
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0, startX: 0, startY: 0 });

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

  // Mouse drag dla fullscreen
  useEffect(() => {
    if (!isFullscreen) return;

    const handleMouseMove = (e) => {
      if (!isDragging) return;
      
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      
      setPosition({
        x: dragStart.startX + dx,
        y: dragStart.startY + dy
      });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, dragStart]);

  const handleMouseDown = (e) => {
    if (!isFullscreen || zoom <= 1) return;
    e.preventDefault();
    setIsDragging(true);
    setDragStart({
      x: e.clientX,
      y: e.clientY,
      startX: position.x,
      startY: position.y
    });
  };

  const handleWheel = (e) => {
    if (!isFullscreen) return;
    e.preventDefault();
    
    const delta = e.deltaY > 0 ? -0.2 : 0.2;
    const newZoom = Math.min(Math.max(zoom + delta, 0.25), 8);
    
    if (newZoom <= 1) {
      setZoom(newZoom);
      setPosition({ x: 0, y: 0 });
    } else {
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
    if (!colorComparisonResults) {
      if (uploadedFile) {
        return uploadedFile.preview;
      }
      return null;
    }

    switch (currentView) {
      case 'original':
        return colorComparisonResults.original_base64 
          ? `data:image/png;base64,${colorComparisonResults.original_base64}`
          : uploadedFile?.preview;
      case 'region':
        return colorComparisonResults.object_region_base64 
          ? `data:image/png;base64,${colorComparisonResults.object_region_base64}`
          : null;
      case 'heatmap':
        return colorComparisonResults.heatmap_base64 
          ? `data:image/png;base64,${colorComparisonResults.heatmap_base64}`
          : null;
      case 'comparison':
        return colorComparisonResults.comparison_base64 
          ? `data:image/png;base64,${colorComparisonResults.comparison_base64}`
          : null;
      default:
        return null;
    }
  };

  const handleDownload = () => {
    const imageSrc = getImageSrc();
    if (imageSrc) {
      const link = document.createElement('a');
      link.href = imageSrc;
      link.download = `${currentView}_${uploadedFile?.name || 'heatmap'}.png`;
      link.click();
    }
  };

  const imageSrc = getImageSrc();
  const hasResults = colorComparisonResults && !colorComparisonResults.error;

  // View toggle buttons
  const ViewToggle = ({ dark = false }) => {
    if (!hasResults) return null;

    const buttonClass = dark 
      ? "px-3 py-2 rounded-lg text-sm font-medium transition-colors"
      : "px-3 py-2 rounded-lg text-sm font-medium transition-colors";
    
    const activeClass = dark
      ? "bg-white/20 text-white"
      : "bg-blue-100 text-blue-700";
    
    const inactiveClass = dark
      ? "text-white/80 hover:bg-white/10"
      : "text-gray-600 hover:bg-gray-100";

    return (
      <div className="flex bg-gray-100 rounded-lg p-1">
        <button
          onClick={() => setCurrentView('original')}
          className={`${buttonClass} ${currentView === 'original' ? activeClass : inactiveClass}`}
        >
          <ImageIcon className="h-4 w-4 mr-1 inline" />
          Oryginalny
        </button>
        {colorComparisonResults?.object_region_base64 && (
          <button
            onClick={() => setCurrentView('region')}
            className={`${buttonClass} ${currentView === 'region' ? activeClass : inactiveClass}`}
          >
            Obiekt
          </button>
        )}
        {colorComparisonResults?.heatmap_base64 && (
          <button
            onClick={() => setCurrentView('heatmap')}
            className={`${buttonClass} ${currentView === 'heatmap' ? activeClass : inactiveClass}`}
          >
            <Thermometer className="h-4 w-4 mr-1 inline" />
            Heatmapa
          </button>
        )}
        {colorComparisonResults?.comparison_base64 && (
          <button
            onClick={() => setCurrentView('comparison')}
            className={`${buttonClass} ${currentView === 'comparison' ? activeClass : inactiveClass}`}
          >
            <BarChart3 className="h-4 w-4 mr-1 inline" />
            Porównanie
          </button>
        )}
      </div>
    );
  };

  // ==================== FULLSCREEN VIEW ====================
  if (isFullscreen) {
    return (
      <div className="fixed inset-0 bg-black z-50 flex items-center justify-center">
        {/* Toolbar */}
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3 bg-black/70 backdrop-blur-sm rounded-lg px-4 py-2">
          <ViewToggle dark={true} />
          
          <div className="h-6 w-px bg-white/30 mx-2"></div>
          
          <button
            onClick={() => setZoom(prev => Math.min(prev + 0.5, 8))}
            className="p-2 text-white hover:bg-white/20 rounded-lg transition-colors"
          >
            <ZoomIn className="h-5 w-5" />
          </button>
          
          <button
            onClick={() => setZoom(prev => Math.max(prev - 0.5, 0.25))}
            className="p-2 text-white hover:bg-white/20 rounded-lg transition-colors"
          >
            <ZoomOut className="h-5 w-5" />
          </button>
          
          <button
            onClick={resetView}
            className="p-2 text-white hover:bg-white/20 rounded-lg transition-colors"
          >
            <RotateCcw className="h-5 w-5" />
          </button>
          
          <button
            onClick={handleDownload}
            className="p-2 text-white hover:bg-white/20 rounded-lg transition-colors"
          >
            <Download className="h-5 w-5" />
          </button>
          
          <div className="h-6 w-px bg-white/30 mx-2"></div>
          
          <button
            onClick={() => setIsFullscreen(false)}
            className="p-2 text-white hover:bg-white/20 rounded-lg transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Image container */}
        <div
          className="w-full h-full flex items-center justify-center cursor-grab active:cursor-grabbing"
          onMouseDown={handleMouseDown}
          onWheel={handleWheel}
        >
          <img
            src={imageSrc}
            alt={`Heatmap - ${currentView}`}
            className="max-w-none object-contain select-none"
            draggable={false}
            style={{
              transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`,
              cursor: zoom > 1 ? 'grab' : 'default',
              pointerEvents: 'none',
              transformOrigin: 'center center',
            }}
          />
        </div>

        {/* Info */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 bg-black/70 text-white text-sm px-4 py-2 rounded-lg">
          Scroll = zoom • Przeciągnij = przesuń • Esc = zamknij
          {hasResults && (
            <span className="ml-4">MSE: {colorComparisonResults.best_match_difference?.toFixed(2)}</span>
          )}
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
          <h3 className="text-lg font-semibold text-gray-900">Analiza porównania kolorów</h3>
          
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
                <Thermometer className="h-12 w-12 text-gray-400" />
              </div>
              <p className="text-gray-500 text-lg font-medium">Brak załadowanego obrazu</p>
              <p className="text-gray-400 text-sm mt-2">Prześlij obraz RTG aby rozpocząć analizę kolorów</p>
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
                    <p className="text-white font-medium">Generowanie heatmapy...</p>
                  </div>
                </div>
              )}
              {imageSrc && (
                <img
                  src={imageSrc}
                  alt={`Heatmap - ${currentView}`}
                  className="max-w-full max-h-[560px] object-contain rounded-lg shadow-2xl"
                  draggable={false}
                />
              )}
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
              {hasResults && (
                <span className="ml-4 text-blue-600 font-medium">
                  MSE: {colorComparisonResults.best_match_difference?.toFixed(2)}
                </span>
              )}
            </div>
            
            {hasResults && (
              <button
                onClick={handleDownload}
                className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                <Download className="h-4 w-4" />
                <span>Pobierz {currentView}</span>
              </button>
            )}
          </div>
        </div>
      )}

      {/* Legenda heatmapy */}
      {hasResults && currentView === 'heatmap' && (
        <div className="px-4 py-3 bg-blue-50 border-t">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-700 font-medium">Legenda kolorów:</span>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-blue-600 rounded"></div>
                <span className="text-gray-600">Małe różnice</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-yellow-500 rounded"></div>
                <span className="text-gray-600">Średnie różnice</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-600 rounded"></div>
                <span className="text-gray-600">Duże różnice</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HeatmapViewer;
