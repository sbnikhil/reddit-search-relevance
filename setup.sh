#!/bin/bash

set -e

echo "🚀 Reddit Search Relevance Engine - Setup Script"
echo "================================================"

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 is not installed. Please install it first."
        exit 1
    fi
}

echo "📋 Checking prerequisites..."
check_command docker
check_command docker-compose
check_command python3

echo "✅ All prerequisites met"

echo ""
echo "📦 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "⚙️  Creating environment file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file. Please edit it with your credentials."
else
    echo "ℹ️  .env file already exists"
fi

echo ""
echo "🐳 Starting Docker services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for Solr to be ready..."
for i in {1..30}; do
    if curl -sf http://localhost:8983/solr/technical_search/admin/ping > /dev/null 2>&1; then
        echo "✅ Solr is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Solr failed to start"
        exit 1
    fi
    sleep 2
done

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your GCP credentials path"
echo "2. Run data ingestion: make ingest"
echo "3. Train the model: make train"
echo "4. Export to ONNX: make export"
echo "5. Start the demo: make streamlit"
echo ""
echo "Or simply run: docker-compose up"
echo ""
echo "Access the UI at: http://localhost:8501"
