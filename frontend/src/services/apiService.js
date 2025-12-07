// API Service for Vehicle Anomaly Detection
const API_BASE_URL = 'http://localhost:5000/api';

class ApiService {
  constructor() {
    this.isModelLoaded = false;
  }

  // Health check
  async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw new Error('Backend nie odpowiada');
    }
  }

  // Load model
  async loadModel() {
    try {
      const response = await fetch(`${API_BASE_URL}/load-model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({})
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      this.isModelLoaded = true;
      return result;
    } catch (error) {
      console.error('Model loading failed:', error);
      throw new Error('Nie udało się załadować modelu');
    }
  }

  // Upload file
  async uploadFile(file) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Upload failed! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('File upload failed:', error);
      throw new Error('Nie udało się przesłać pliku');
    }
  }

  // Analyze image
  async analyzeImage(fileId) {
    try {
      if (!this.isModelLoaded) {
        throw new Error('Model nie jest załadowany');
      }

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_id: fileId
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Analysis failed! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Image analysis failed:', error);
      throw new Error(error.message || 'Nie udało się przeanalizować obrazu');
    }
  }

  // Analyze image using comparison method (with heatmap)
  async analyzeImageComparison(fileId) {
    try {
      if (!this.isModelLoaded) {
        throw new Error('Model nie jest załadowany');
      }

      const response = await fetch(`${API_BASE_URL}/analyze-comparison`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_id: fileId,
          use_alignment: false,  // Wyłącz dla szybkości
          use_ssim: true,
          fast_mode: true
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Analysis failed! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Image comparison analysis failed:', error);
      throw new Error(error.message || 'Nie udało się przeanalizować obrazu metodą porównawczą');
    }
  }

  // Full upload and analyze workflow using comparison method
  async uploadAndAnalyzeComparison(file) {
    try {
      // Step 1: Upload file
      const uploadResult = await this.uploadFile(file);
      
      // Step 2: Analyze using comparison method
      const analysisResult = await this.analyzeImageComparison(uploadResult.file_id);
      
      return {
        ...analysisResult,
        uploadInfo: uploadResult
      };
    } catch (error) {
      console.error('Upload and analyze comparison workflow failed:', error);
      throw error;
    }
  }

  // Backward compatibility - alias for old method name
  async uploadAndAnalyze(file) {
    // Redirect to new comparison method with heatmap
    return await this.uploadAndAnalyzeComparison(file);
  }

  // Get model status
  getModelStatus() {
    return this.isModelLoaded;
  }
}

// Export singleton instance
const apiService = new ApiService();
export default apiService;
