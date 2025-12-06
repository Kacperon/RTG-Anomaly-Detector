import React from 'react';
import { Activity } from 'lucide-react';

const Header = () => {
  return (
    <header className="bg-white shadow-sm border-b">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-primary-600 p-2 rounded-lg">
              <Activity className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Vehicle Scan Anomaly Detector
              </h1>
              <p className="text-sm text-gray-600">
                AI-powered vehicle inspection system
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <div className="text-sm text-gray-600">Version</div>
              <div className="font-medium text-gray-900">1.0.0</div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
