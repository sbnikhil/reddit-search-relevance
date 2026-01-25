"""
Health check endpoint for Docker and monitoring systems
"""
from flask import Flask, jsonify
import os
import torch
from pathlib import Path

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    status = {
        "status": "healthy",
        "service": "reddit-search-relevance",
        "checks": {}
    }
    
    # Check if model file exists
    model_path = Path("models/registry")
    model_files = list(model_path.glob("reddit_ranker_v*.pt"))
    status["checks"]["model_present"] = len(model_files) > 0
    
    # Check if config exists
    config_path = Path("config/settings.yaml")
    status["checks"]["config_present"] = config_path.exists()
    
    # Check PyTorch availability
    status["checks"]["pytorch_available"] = torch.cuda.is_available() or True
    
    # Overall health
    all_healthy = all(status["checks"].values())
    status["status"] = "healthy" if all_healthy else "degraded"
    
    return jsonify(status), 200 if all_healthy else 503

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check for k8s"""
    try:
        # Check critical dependencies
        import transformers
        import pysolr
        
        return jsonify({"status": "ready"}), 200
    except ImportError as e:
        return jsonify({"status": "not_ready", "error": str(e)}), 503

if __name__ == '__main__':
    port = int(os.getenv('HEALTH_CHECK_PORT', 8000))
    app.run(host='0.0.0.0', port=port)
