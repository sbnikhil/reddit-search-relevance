# Reddit Search Relevance Engine

A production-ready hybrid search system that combines Apache Solr's lexical search with BERT-based semantic ranking to improve Reddit post relevance for technical queries.

## Overview

Standard keyword-based search engines struggle with understanding context and intent. This project implements a **two-stage retrieval-and-ranking architecture** that:

1. **Retrieves** top candidates using Apache Solr (BM25)
2. **Re-ranks** using a fine-tuned BERT model with domain-specific features
3. **Combines** scores using a hybrid formula for optimal relevance

## Tech Stack

### Data & Feature Engineering
- **Google BigQuery** - Data warehouse for feature extraction
- **Apache Beam** - Distributed pipeline for utility signal processing
- **Pandas/NumPy** - Feature normalization and vectorization

### Search Infrastructure
- **Apache Solr** - Fast lexical retrieval engine (BM25)
- **Custom Schema** - Optimized for technical content tokenization

### ML & AI
- **PyTorch** - Deep learning framework
- **Hugging Face Transformers** - BERT-based cross-encoder (`bert-base-uncased`)
- **MPS (Metal Performance Shaders)** - GPU acceleration on Apple Silicon

### Orchestration (Planned)
- **Apache Airflow** - Workflow automation
- **Ray** - Distributed hyperparameter tuning

## Architecture

```
┌──────────────┐
│  BigQuery    │  Feature extraction (expertise, utility scores)
└──────┬───────┘
       │
       v
┌──────────────┐
│  Solr Index  │  BM25 retrieval → Top 50 candidates
└──────┬───────┘
       │
       v
┌──────────────┐
│ BERT Ranker  │  Semantic scoring + feature injection
└──────┬───────┘
       │
       v
┌──────────────┐
│ Hybrid Score │  α × Solr + (1-α) × BERT
└──────────────┘
```

## Model Architecture

**RedditRelevanceRanker**: Custom neural network combining:
- **BERT transformer** (768-dim) → Text understanding
- **MLP** (2-dim) → Feature injection (expertise + utility)
- **Fusion layer** → Combined relevance score

### Training Features
- **Expertise Score**: User karma and subreddit activity
- **Utility Score**: Presence of code blocks, solutions, helpful keywords

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/reddit-search-relevance.git
cd reddit-search-relevance

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Solr
cd infra/solr && docker-compose up -d
```

## Usage

### Train Model
```bash
PYTHONPATH=. python scripts/train_ranker.py
```

### Evaluate Performance
```bash
PYTHONPATH=. python scripts/evaluate.py
```

### Search Interface
```bash
PYTHONPATH=. python app.py
```

## Project Structure

```
reddit-search-relevance/
├── app.py                      # CLI search interface
├── config/
│   └── settings.yaml          # Central configuration
├── data/
│   ├── pipelines/             # Feature extraction pipelines
│   └── schemas/               # BigQuery SQL schemas
├── infra/
│   └── solr/                  # Solr configuration
├── models/
│   ├── arch/                  # Model architectures
│   └── registry/              # Trained weights
├── orchestration/
│   └── dags/                  # Airflow DAGs
├── scripts/
│   ├── train_ranker.py        # Training pipeline
│   ├── evaluate.py            # Evaluation metrics
│   ├── diagnose_model.py      # Debugging tools
│   └── ingest_to_solr.py      # Data ingestion
└── tests/                     # Unit tests
```

## Evaluation Metrics

- **nDCG@10**: Measures ranking quality
- **Recall@K**: Coverage of relevant documents
- **Latency (P99)**: 99th percentile response time
- **Lift**: Improvement over Solr baseline
