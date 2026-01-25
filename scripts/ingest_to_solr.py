import pysolr
from sentence_transformers import SentenceTransformer
import yaml
import logging
import os
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path="config/settings.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    required_keys = ["project_id", "dataset"]
    for key in required_keys:
        if key not in config["database"]:
            raise KeyError(f"MISSING CONFIG: 'database.{key}' not found in {path}")
    return config


class RedditIndexer:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer(config["model"]["encoder_name"])
        self.solr_url = config["search_tuning"]["solr_url"]
        self.solr = pysolr.Solr(self.solr_url, always_commit=True, timeout=30)
        self.bq_client = bigquery.Client(project=config["database"]["project_id"])

    def create_embeddings(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def fetch_from_bigquery(self, limit=10000):
        query = f"""
            SELECT 
                id,
                title,
                body,
                expertise_score,
                utility_score
            FROM `{self.config['database']['source_table']}`
            WHERE body IS NOT NULL
            LIMIT {limit}
        """

        logger.info(f"Fetching data from BigQuery...")
        df = self.bq_client.query(query).to_dataframe()
        logger.info(f"Retrieved {len(df)} documents")

        documents = []
        for _, row in df.iterrows():
            documents.append(
                {
                    "id": str(row["id"]),
                    "title": str(row.get("title", "")),
                    "body": str(row["body"]),
                    "expertise_score": float(row.get("expertise_score", 0.0)),
                    "utility_score": float(row.get("utility_score", 0.0)),
                }
            )

        return documents

    def batch_index(self, documents, batch_size=100):
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc.get("body", "") for doc in batch]
            vectors = self.create_embeddings(texts)

            solr_docs = []
            for j, doc in enumerate(batch):
                solr_docs.append(
                    {
                        "id": doc["id"],
                        "title_t": doc.get("title", ""),
                        "body_t": doc.get("body", ""),
                        "expertise_score": doc.get("expertise_score", 0.0),
                        "utility_score": doc.get("utility_score", 0.0),
                        "content_vector": vectors[j],
                    }
                )

            self.solr.add(solr_docs)
            logger.info(f"Indexed batch {i//batch_size + 1} ({len(solr_docs)} docs)")


def main():
    config = load_config()
    indexer = RedditIndexer(config)

    documents = indexer.fetch_from_bigquery(limit=10000)
    indexer.batch_index(documents)

    logger.info(f"Indexing complete: {len(documents)} documents")


if __name__ == "__main__":
    main()
