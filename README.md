# Reddit Search Relevance Engine

A production-ready semantic search system that combines traditional keyword search (BM25) with deep learning (BERT) to rank Reddit posts by relevance, expertise, and utility.

## Overview

Traditional keyword-based search systems struggle with semantic understanding and context. This project implements a two-stage retrieval system that:

1. **Fast Retrieval**: Apache Solr performs BM25 keyword search to retrieve top candidates
2. **Semantic Re-ranking**: BERT Cross-Encoder evaluates semantic relevance
3. **Hybrid Fusion**: Combines both signals with community expertise and utility scores

### Key Features

- **Two-Stage Retrieval Architecture**: Balances speed and accuracy
- **BERT-based Semantic Understanding**: Cross-encoder model for query-document relevance
- **Feature Fusion**: Integrates expertise and utility signals from community voting
- **Production-Ready**: Docker containerization, CI/CD, monitoring endpoints
- **Interactive Demo**: Streamlit UI for side-by-side comparison

## Architecture

```
Query → Solr (BM25) → Top-K Candidates → BERT Re-Ranker → Hybrid Fusion → Ranked Results
                                              ↑
                                    Expertise + Utility Scores
```

**Hybrid Score Formula**:
```
Score = α × BM25_normalized + (1-α) × BERT_score
```

## Tech Stack

- **Search Engine**: Apache Solr 9.4 (BM25 indexing)
- **ML Framework**: PyTorch 2.9.1
- **Model**: BERT-base-uncased (109M parameters)
- **Feature Engineering**: Apache Beam, BigQuery
- **Orchestration**: Apache Airflow
- **Containerization**: Docker Compose
- **CI/CD**: GitHub Actions
- **Demo**: Streamlit

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- 8GB RAM minimum (16GB recommended)
- CUDA-capable GPU (optional, CPU supported)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/reddit-search-relevance.git
cd reddit-search-relevance
```

### 2. Start Apache Solr

```bash
cd infra/solr
docker-compose up -d
cd ../..
```

Verify Solr is running:
```bash
curl http://localhost:8983/solr/admin/cores?action=STATUS
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Sample Data

```bash
python scripts/create_sample_data.py
```

This creates 500 synthetic Reddit posts and indexes them to Solr.

### 5. Train the Model

```bash
PYTHONPATH=. python scripts/train_ranker.py
```

Training takes approximately 2-3 minutes on CPU. The model will be saved to `models/registry/`.

### 6. Launch Demo

```bash
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501`

## Project Structure

```
reddit-search-relevance/
├── app.py                          # Flask API server
├── streamlit_app.py                # Interactive demo UI
├── config/
│   └── settings.yaml               # Configuration
├── data/
│   ├── pipelines/
│   │   └── utility_extractor.py    # Apache Beam pipeline
│   └── sample_posts.json           # Generated sample data
├── models/
│   ├── arch/
│   │   └── relevance_ranker.py     # BERT-based ranker architecture
│   └── registry/
│       ├── reddit_ranker_v1.0.1.pt # Trained model weights
│       └── feature_norm_params.json # Feature normalization params
├── scripts/
│   ├── create_sample_data.py       # Generate synthetic data
│   ├── train_ranker.py             # Model training script
│   ├── ingest_to_solr.py           # Bulk indexing script
│   └── evaluate.py                 # Model evaluation
├── infra/
│   └── solr/
│       ├── docker-compose.yml      # Solr service configuration
│       └── solr-data/              # Solr cores and schemas
├── orchestration/
│   └── dags/
│       └── ml_pipeline.py          # Airflow DAG
└── tests/
    ├── test_pipeline.py
    └── test_ranker.py
```

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
search_tuning:
  solr_url: "http://localhost:8983/solr/reddit_posts"
  k_recall: 50          # Number of candidates from Solr
  alpha: 0.2            # Weight for BM25 (0.8 for BERT)

training:
  epochs: 10
  batch_size: 16
  learning_rate: 2e-5
  model_name: "bert-base-uncased"

model_params:
  dropout_rate: 0.1
```

## API Usage

### Start the API Server

```bash
python app.py
```

### Search Endpoint

```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "python async programming", "top_k": 10}'
```

Response:
```json
{
  "results": [
    {
      "id": "post_123",
      "title": "Python async tutorial",
      "body": "...",
      "score": 0.856,
      "expertise": 0.92,
      "utility": 0.88
    }
  ],
  "latency_ms": 127.3
}
```

### Health Check

```bash
curl http://localhost:5000/health
```

## Training with Custom Data

### Using BigQuery

1. Set up Google Cloud credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

2. Update `config/settings.yaml`:
```yaml
database:
  project_id: "your-gcp-project"
  source_table: "project.dataset.reddit_posts"
```

3. Train:
```bash
PYTHONPATH=. python scripts/train_ranker.py
```

### Using Local Data

Format your data as JSON:
```json
[
  {
    "id": "post_1",
    "title": "...",
    "body": "...",
    "expertise_score": 0.85,
    "utility_score": 0.92,
    "label": 1
  }
]
```

Save to `data/sample_posts.json` and run training.

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black .
flake8 .
```

### Building Docker Image

```bash
docker build -t reddit-search-relevance:latest .
```

## Deployment

### Docker Compose (Full Stack)

```bash
docker-compose up -d
```

This starts:
- Solr (port 8983)
- API server (port 5000)
- Streamlit UI (port 8501)

## Performance

| Metric | Value |
|--------|-------|
| Latency (p50) | <100ms |
| Latency (p95) | <200ms |
| Throughput | ~50 QPS |
| Model Size | 420MB |
| Index Size | ~2GB / 1M docs |

## Model Details

- **Architecture**: BERT Cross-Encoder + MLP Fusion Layer
- **Parameters**: 109,589,125
- **Input**: Query-document pairs + expertise + utility scores
- **Output**: Relevance score (0-1)
- **Training**: Binary Cross-Entropy Loss with gradient clipping

## Background

This project addresses the challenge of information retrieval in community-driven platforms where:
- Users ask questions in varied language
- Keyword matching alone fails to capture semantic intent
- Community signals (expertise, utility) are strong relevance indicators

The two-stage architecture balances:
- **Speed**: Solr handles fast candidate retrieval
- **Accuracy**: BERT provides semantic understanding
- **Context**: Community features improve ranking quality

## License

MIT License - see LICENSE file for details.
