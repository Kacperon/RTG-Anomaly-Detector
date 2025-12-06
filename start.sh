#!/bin/bash

# Vehicle Scan Anomaly Detector - Start Script

echo "🚗 Vehicle Scan Anomaly Detector - Setup & Start"
echo "==============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "📋 Installing Python dependencies..."
pip install -r requirements.txt

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js to run the frontend."
    echo "   You can download it from: https://nodejs.org/"
    exit 1
fi

# Install frontend dependencies
echo "🌐 Installing frontend dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi
cd ..

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads results

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Starting application..."
echo ""

# Start backend in background
echo "🔧 Starting Flask backend (port 5000)..."
python app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🌐 Starting React frontend (port 3000)..."
cd frontend
npm start &
FRONTEND_PID=$!

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping application..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Trap signals to cleanup
trap cleanup SIGINT SIGTERM

echo ""
echo "🎉 Application is starting!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend:  http://localhost:5000"
echo ""
echo "📱 Open your browser and go to http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Wait for user to stop
wait
