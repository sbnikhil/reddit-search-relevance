import pysolr
from sentence_transformers import SentenceTransformer
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    required_keys = ['project_id', 'dataset']
    for key in required_keys:
        if key not in config['database']:
            raise KeyError(f"MISSING CONFIG: 'database.{key}' not found in {path}")    
    return config

class RedditIndexer:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer(config['model']['encoder_name'])
        self.solr_url = f"http://localhost:8983/solr/{config['database']['dataset']}"
        self.solr = pysolr.Solr(self.solr_url, always_commit=True)

    def create_embeddings(self, texts):
        """Converts raw text into dense vectors."""
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def batch_index(self, documents, batch_size=100):
        """Indexes documents in chunks to manage memory."""
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc.get('body', '') for doc in batch]
            vectors = self.create_embeddings(texts)

            solr_docs = []
            for j, doc in enumerate(batch):
                solr_docs.append({
                    "id": doc['id'],
                    "title_t": doc.get('title', 'No Title'),  # Changed from post_title
                    "body_t": doc.get('body', ''),  # Changed from post_body
                    "expertise_score": doc.get('expertise_score', 0.5),  # Changed from topical_focus_score
                    "utility_score": doc.get('utility_score', 0.5),  # Added utility_score
                    "content_vector": vectors[j]
                })
            
            self.solr.add(solr_docs)
            logger.info(f"Successfully indexed batch of {len(solr_docs)} documents.")

if __name__ == "__main__":
    config = load_config()
    indexer = RedditIndexer(config)
    
    mock_data = [
        {"id": "v1", "title": "PyTorch Optimization", "body": "How to speed up training?", "topical_focus_score": 0.92}
    ]
    indexer.batch_index(mock_data)