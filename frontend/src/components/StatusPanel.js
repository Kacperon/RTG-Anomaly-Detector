import React from 'react';
import { CheckCircle, AlertCircle, Info, Clock } from 'lucide-react';

const StatusPanel = ({ messages, modelLoaded, isAnalyzing }) => {
  const getStatusIcon = (type) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'warning':
        return <AlertCircle className="h-4 w-4 text-orange-500" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Info className="h-4 w-4 text-blue-500" />;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 overflow-hidden">
      <h3 className="text-lg font-semibold text-gray-900 mb-4 break-words">
        Status systemu
      </h3>
      
      {/* System Status Indicators */}
      <div className="space-y-3 mb-6">
        <div className="flex items-center justify-between min-w-0">
          <span className="text-sm text-gray-600 truncate">Model AI</span>
          <div className={`flex items-center space-x-2 flex-shrink-0 ${modelLoaded ? 'text-green-600' : 'text-gray-400'}`}>
            {modelLoaded ? (
              <CheckCircle className="h-4 w-4" />
            ) : (
              <div className="h-4 w-4 border-2 border-gray-300 rounded-full"></div>
            )}
            <span className="text-sm whitespace-nowrap">
              {modelLoaded ? 'Załadowany' : 'Niezaładowany'}
            </span>
          </div>
        </div>
        
        <div className="flex items-center justify-between min-w-0">
          <span className="text-sm text-gray-600 truncate">Status analizy</span>
          <div className={`flex items-center space-x-2 flex-shrink-0 ${isAnalyzing ? 'text-blue-600' : 'text-gray-400'}`}>
            {isAnalyzing ? (
              <Clock className="h-4 w-4" />
            ) : (
              <div className="h-4 w-4 border-2 border-gray-300 rounded-full"></div>
            )}
            <span className="text-sm whitespace-nowrap">
              {isAnalyzing ? 'W trakcie' : 'Gotowy'}
            </span>
          </div>
        </div>
      </div>

      {/* Activity Log */}
      <div className="min-w-0">
        <h4 className="text-sm font-medium text-gray-900 mb-3 break-words">
          Dziennik aktywności
        </h4>
        
        <div className="space-y-2 max-h-48 overflow-y-auto overflow-x-hidden">
          {messages.length === 0 ? (
            <p className="text-sm text-gray-500 italic">
              Brak aktywności
            </p>
          ) : (
            messages.slice(-10).reverse().map((message) => (
              <div 
                key={message.id}
                className="flex items-start space-x-2 text-sm min-w-0"
              >
                {getStatusIcon(message.type)}
                <div className="flex-1 min-w-0">
                  <p className="text-gray-900 break-words">
                    {message.text}
                  </p>
                  <p className="text-xs text-gray-500">
                    {message.timestamp}
                  </p>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="mt-6 pt-4 border-t">
        <div className="grid grid-cols-2 gap-4 text-center">
          <div>
            <div className="text-lg font-bold text-gray-900">
              {messages.filter(m => m.type === 'success').length}
            </div>
            <div className="text-xs text-gray-600">Sukcesy</div>
          </div>
          <div>
            <div className="text-lg font-bold text-gray-900">
              {messages.filter(m => m.type === 'warning').length}
            </div>
            <div className="text-xs text-gray-600">Ostrzeżenia</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatusPanel;
