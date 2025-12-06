import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileImage } from 'lucide-react';

const UploadArea = ({ onFileUpload, disabled }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      
      // Create file object with additional properties for our app
      const fileWithPreview = Object.assign(file, {
        preview: URL.createObjectURL(file),
        id: Date.now() + Math.random()
      });
      
      onFileUpload(fileWithPreview);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/bmp': ['.bmp'],
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg']
    },
    multiple: false,
    disabled
  });

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 overflow-hidden">
      <h2 className="text-xl font-semibold text-gray-900 mb-4 break-words">
        Załaduj skan pojazdu
      </h2>
      
      <div
        {...getRootProps()}
        className={`
          upload-area border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-colors duration-200
          ${isDragActive 
            ? 'border-primary-500 bg-primary-50' 
            : 'border-gray-300 hover:border-primary-400'
          }
          ${disabled ? 'cursor-not-allowed opacity-50' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className="space-y-4">
          {isDragActive ? (
            <Upload className="mx-auto h-12 w-12 text-primary-500" />
          ) : (
            <FileImage className="mx-auto h-12 w-12 text-gray-400" />
          )}
          
          <div className="break-words">
            <p className="text-lg font-medium text-gray-900">
              {isDragActive
                ? 'Upuść plik tutaj'
                : 'Przeciągnij i upuść skan pojazdu lub kliknij aby wybrać'
              }
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Obsługiwane formaty: BMP, PNG, JPG (max 10MB)
            </p>
          </div>
        </div>
      </div>
      
      <div className="mt-4 text-xs text-gray-500 break-words">
        <p>• Plik zostanie przesłany na serwer do analizy</p>
        <p>• Anomalie będą automatycznie zaznaczone na obrazie</p>
        <p>• Obsługiwane są skany pojazdów w odcieniach szarości</p>
      </div>
    </div>
  );
};

export default UploadArea;
