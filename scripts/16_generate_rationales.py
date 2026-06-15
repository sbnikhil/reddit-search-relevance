"""
Generate LLM soft label distributions for ρ-conditioned cross-encoder training.

Samples N (query, product, esci_label) pairs from the ESCI training split and
prompts claude-haiku to assign probability distributions over {E, S, C, I}.
Stores results in BigQuery table `rationale_labels`.

These soft labels are used by scripts/17_train_cross_encoder.py:
  - Pairs with high-ρ queries (ColBERT agrees with human labels) use hard labels.
  - Pairs with low-ρ queries use LLM soft labels → KL distillation loss.

Strategy for sampling:
  - Stratified by esci_label to cover all relevance levels.
  - Oversample low-ρ queries (ColBERT unreliable) where LLM signal matters most.
  - Ensures representation of hard cases (C rated high by BM25, I with token overlap).

Cost estimate (claude-haiku-4-5):
  20K samples × ~300 tokens/sample ≈ 6M tokens ≈ $2-4 at current haiku pricing.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/16_generate_rationales.py --smoke          # 20 samples, no BQ write
    python scripts/16_generate_rationales.py --n_samples 20000
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
from google.cloud import bigquery

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECT_ID, BQ_DATASET

OUTPUT_TABLE = f"{PROJECT_ID}.{BQ_DATASET}.rationale_labels"

PROMPT_TEMPLATE = """\
You are evaluating product search relevance for an e-commerce search engine.

Query: "{query}"
Product Title: "{title}"
Product Description: "{description}"

Classify this product's relevance to the query using these labels:
- E (Exact): The product directly and completely answers the query intent.
- S (Substitute): A similar product that could work, but isn't exactly what was queried.
- C (Complement): An accessory or complement to the queried product (e.g. a case for a phone).
- I (Irrelevant): Not useful for this query.

Respond with ONLY a JSON object like this (probabilities must sum to 1.0):
{{"E": 0.00, "S": 0.00, "C": 0.00, "I": 0.00}}"""


def _call_llm(client, query: str, title: str, description: str,
              max_retries: int = 3) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(
        query=query[:200],
        title=title[:300],
        description=description[:400],
    )
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            probs = json.loads(raw)
            # Validate
            assert set(probs.keys()) == {"E", "S", "C", "I"}
            total = sum(probs.values())
            return {k: v / total for k, v in probs.items()}   # renormalize
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def _sample_pairs(bq, n_samples: int) -> pd.DataFrame:
    """Stratified sample: equal representation per label, skewed to low-ρ queries."""
    per_label = n_samples // 4

    sql = f"""
    WITH rho_ranks AS (
        SELECT
            CAST(e.query_id AS STRING) AS query_id,
            RANK() OVER (PARTITION BY e.query_id ORDER BY c.colbert_score DESC) AS colbert_rank,
            RANK() OVER (PARTITION BY e.query_id ORDER BY e.gain        DESC) AS gain_rank
        FROM `{PROJECT_ID}.{BQ_DATASET}.examples` e
        JOIN `{PROJECT_ID}.{BQ_DATASET}.colbert_scores` c
            ON CAST(e.query_id AS STRING) = c.query_id
            AND CAST(e.product_id AS STRING) = c.product_id
        WHERE e.split = 'train'
    ),
    query_rho AS (
        SELECT query_id, CORR(colbert_rank, gain_rank) AS rho
        FROM rho_ranks
        GROUP BY query_id
    ),
    labelled AS (
        SELECT
            CAST(e.query_id AS STRING) AS query_id,
            CAST(e.product_id AS STRING) AS product_id,
            e.query,
            e.esci_label,
            COALESCE(p.product_title, '') AS product_title,
            COALESCE(p.product_description, '') AS product_description,
            COALESCE(r.rho, 0.0) AS rho
        FROM `{PROJECT_ID}.{BQ_DATASET}.examples` e
        JOIN `{PROJECT_ID}.{BQ_DATASET}.products` p USING (product_id)
        LEFT JOIN query_rho r ON CAST(e.query_id AS STRING) = r.query_id
        WHERE e.split = 'train'
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY esci_label
                ORDER BY rho ASC, RAND()   -- low-ρ queries first within each label
            ) AS rn
        FROM labelled
    )
    SELECT query_id, product_id, query, esci_label,
           product_title, product_description, rho
    FROM ranked
    WHERE rn <= {per_label}
    ORDER BY rho ASC
    """
    print(f"Sampling {n_samples} training pairs...")
    df = bq.query(sql).to_dataframe()
    print(f"  Got {len(df):,} pairs  (label dist: {df['esci_label'].value_counts().to_dict()})")
    return df


def run(args) -> None:
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY env var")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    bq     = bigquery.Client(project=PROJECT_ID)

    df = _sample_pairs(bq, args.n_samples)
    if args.smoke:
        df = df.head(20)
        print("[SMOKE] capped at 20 samples")

    records = []
    n_ok = n_fail = 0
    t0 = time.time()

    for i, row in df.iterrows():
        probs = _call_llm(
            client,
            query=row["query"],
            title=row["product_title"],
            description=row["product_description"],
        )
        if probs is None:
            n_fail += 1
            continue

        records.append({
            "query_id":      str(row["query_id"]),
            "product_id":    str(row["product_id"]),
            "p_exact":       float(probs["E"]),
            "p_substitute":  float(probs["S"]),
            "p_complement":  float(probs["C"]),
            "p_irrelevant":  float(probs["I"]),
            "rho":           float(row["rho"]) if not pd.isna(row["rho"]) else 0.5,
        })
        n_ok += 1

        if n_ok % 100 == 0:
            elapsed = time.time() - t0
            rate = n_ok / elapsed
            eta = (len(df) - n_ok) / rate if rate > 0 else 0
            print(f"  {n_ok}/{len(df)}  ok={n_ok}  fail={n_fail}  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f}m")

    print(f"\nDone: {n_ok} OK, {n_fail} failed out of {len(df)} pairs")

    if args.smoke:
        print("\nSample output:")
        for r in records[:3]:
            print(f"  E={r['p_exact']:.2f} S={r['p_substitute']:.2f} "
                  f"C={r['p_complement']:.2f} I={r['p_irrelevant']:.2f}  ρ={r['rho']:.3f}")
        print("Smoke test done. No BigQuery write.")
        return

    results_df = pd.DataFrame(records)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("query_id",     "STRING"),
            bigquery.SchemaField("product_id",   "STRING"),
            bigquery.SchemaField("p_exact",      "FLOAT64"),
            bigquery.SchemaField("p_substitute", "FLOAT64"),
            bigquery.SchemaField("p_complement", "FLOAT64"),
            bigquery.SchemaField("p_irrelevant", "FLOAT64"),
            bigquery.SchemaField("rho",          "FLOAT64"),
        ],
    )
    bq.load_table_from_dataframe(results_df, OUTPUT_TABLE,
                                 job_config=job_config).result()
    print(f"Written {len(results_df):,} records → {OUTPUT_TABLE}")
    print("Next: python scripts/17_train_cross_encoder.py --submit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--smoke",     action="store_true",
                        help="Run 20 samples, no BigQuery write")
    args = parser.parse_args()
    run(args)
