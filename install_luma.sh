#!/bin/bash

echo "🔧 Setting up Luma Console environment..."

# Ensure Python is available
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 is not installed. Please install Python 3.8+ and rerun this script."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv luma_env

# Activate it
source luma_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install torch tokenizers flask flask-cors

# Optional: Check for CUDA support
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "🚀 CUDA is available. GPU acceleration enabled."
else
    echo "⚠️ CUDA not available. Running on CPU."
fi

echo "✅ Luma Console setup complete."
echo "➡️ To activate this environment later, run: source luma_env/bin/activate"
echo "➡️ Then launch: python server.py or python trainable_app.py"
