#!/bin/bash
set -e

echo "🚀 Setting up Nepali Distillation System..."

# Check macOS and M4 Max
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️ This setup is optimized for macOS with M4 Max"
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"
if [[ "$python_version" < "$required_version" ]]; then
    echo "❌ Python 3.10+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/{raw,processed,datasets}
mkdir -p models/{teacher,student,checkpoints}
mkdir -p outputs/{logs,metrics,exports}
mkdir -p config
mkdir -p src/{data,model,training,evaluation,deployment}
mkdir -p scripts
mkdir -p tests

# Create __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/model/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/deployment/__init__.py

# Setup Ollama if available
echo "🤖 Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama found"
else
    echo "📥 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. source venv/bin/activate"
echo "2. python scripts/setup_environment.py --init --verify --teacher-model ollama"
echo "3. python scripts/train.py --config config/training_config.yaml"
echo ""
echo "🎉 Ready to start training your Nepali model!"
