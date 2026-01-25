import streamlit as st
import pysolr
import torch
import yaml
import os
import json
import numpy as np
import time
from transformers import AutoTokenizer
from models.arch.relevance_ranker import RedditRelevanceRanker

st.set_page_config(page_title="Reddit Search Relevance", layout="wide")


@st.cache_resource
def load_model():
    with open("config/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    norm_params_path = os.path.join(cfg["artifacts"]["model_path"], "feature_norm_params.json")
    if os.path.exists(norm_params_path):
        with open(norm_params_path, "r") as f:
            norm_params = json.load(f)
    else:
        norm_params = {"expertise_min": 0, "expertise_max": 1, "utility_min": 0, "utility_max": 1}

    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    except:
        device = torch.device("cpu")

    solr_url = os.getenv("SOLR_URL", cfg["search_tuning"]["solr_url"])
    solr = pysolr.Solr(solr_url, timeout=10)

    tokenizer = AutoTokenizer.from_pretrained(cfg["training"]["model_name"])

    model = RedditRelevanceRanker(
        cfg["training"]["model_name"],
        cfg["training"]["extra_feature_dim"],
        dropout=cfg["model_params"].get("dropout_rate", 0.1),
    ).to(device)

    model_file = f"reddit_ranker_v{cfg['artifacts']['version']}.pt"
    model_path = os.path.join(cfg["artifacts"]["model_path"], model_file)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        st.error(f"Model not found at {model_path}")
        return None, None, None, None, None

    return cfg, model, tokenizer, solr, device, norm_params


def search_hybrid(query_text, cfg, model, tokenizer, solr, device, norm_params, top_k=10):
    start_time = time.perf_counter()

    try:
        results = solr.search(f"body_t:({query_text})", rows=cfg["search_tuning"]["k_recall"], fl="*,score")
    except Exception as e:
        st.error(f"Solr error: {e}")
        return None, None, None

    if not results or len(results) == 0:
        return [], [], 0

    # Extract text from Solr's multiValued fields (returns lists)
    batch_docs = []
    for doc in results:
        body = doc.get("body_t", [""])
        body_text = body[0] if isinstance(body, list) and len(body) > 0 else str(body) if body else ""
        batch_docs.append(body_text)

    inputs = tokenizer(
        [query_text] * len(results),
        batch_docs,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg["data_processing"]["max_sequence_length"],
    ).to(device)

    raw_features = [[float(doc.get("expertise_score", 0.0)), float(doc.get("utility_score", 0.0))] for doc in results]

    normalized_features = []
    for exp, util in raw_features:
        exp_norm = (exp - norm_params["expertise_min"]) / (
            norm_params["expertise_max"] - norm_params["expertise_min"] + 1e-6
        )
        util_norm = (util - norm_params["utility_min"]) / (
            norm_params["utility_max"] - norm_params["utility_min"] + 1e-6
        )
        exp_norm = max(0.0, min(1.0, exp_norm))
        util_norm = max(0.0, min(1.0, util_norm))
        normalized_features.append([exp_norm, util_norm])

    features = torch.tensor(normalized_features, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"], features)
        bert_scores = torch.sigmoid(logits).cpu().numpy().flatten()

    solr_scores = np.array([float(doc.get("score", 0.0)) for doc in results])
    s_min, s_max = solr_scores.min(), solr_scores.max()
    solr_norm = (solr_scores - s_min) / (s_max - s_min + 1e-6)

    alpha = cfg["search_tuning"]["alpha"]
    hybrid_scores = (alpha * solr_norm) + ((1 - alpha) * bert_scores)

    solr_ranked = sorted(zip(solr_norm, batch_docs, raw_features), key=lambda x: x[0], reverse=True)[:top_k]
    hybrid_ranked = sorted(
        zip(hybrid_scores, batch_docs, raw_features, bert_scores, solr_norm), key=lambda x: x[0], reverse=True
    )[:top_k]

    latency = (time.perf_counter() - start_time) * 1000

    return solr_ranked, hybrid_ranked, latency


st.title("Reddit Expert Search")
st.markdown("""
### Compare Traditional vs AI-Powered Search
**Left (Baseline):** Pure keyword matching (BM25) | **Right (Our System):** Semantic understanding + expertise signals
""")

cfg, model, tokenizer, solr, device, norm_params = load_model()

if model is None:
    st.stop()

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Enter your search query:", placeholder="Django file upload example")
with col2:
    top_k = st.slider("Results", 3, 20, 10)

if st.button("Search", type="primary") or query:
    if query:
        with st.spinner("Searching..."):
            solr_results, hybrid_results, latency = search_hybrid(
                query, cfg, model, tokenizer, solr, device, norm_params, top_k
            )

        if not solr_results:
            st.warning("No results found")
        else:
            st.success(f"Retrieved {len(hybrid_results)} results in {latency:.1f}ms")

            st.info(
                "Notice how rankings differ: BERT promotes semantically relevant answers, not just keyword matches."
            )

            col_solr, col_hybrid = st.columns(2)

            with col_solr:
                st.markdown("### Traditional (Keyword Only)")
                st.caption("Pure BM25 ranking - matches exact words")
                for i, (score, body, features) in enumerate(solr_results, 1):
                    # Create preview (first 100 chars)
                    preview = body[:100].replace("\n", " ").strip()
                    if len(body) > 100:
                        preview += "..."

                    with st.expander(f"#{i} - {preview}", expanded=(i == 1)):
                        st.markdown(f"**Keyword Score:** `{score:.3f}` (normalized BM25)")
                        st.caption(f"Expertise: {features[0]:.2f} | Utility: {features[1]:.2f}")
                        st.markdown("---")
                        st.text(body[:400] + ("..." if len(body) > 400 else ""))

            with col_hybrid:
                st.markdown("### AI-Enhanced (Semantic + Expertise)")
                st.caption("BERT semantic understanding + community signals")
                for i, (h_score, body, features, b_score, s_score) in enumerate(hybrid_results, 1):
                    # Create preview
                    preview = body[:100].replace("\n", " ").strip()
                    if len(body) > 100:
                        preview += "..."

                    # Highlight if this moved up in ranking
                    rank_diff = ""
                    solr_bodies = [b for _, b, _ in solr_results]
                    if body in solr_bodies:
                        old_rank = solr_bodies.index(body) + 1
                        if old_rank > i:
                            rank_diff = f" (was #{old_rank})"

                    with st.expander(f"#{i} - {preview}{rank_diff}", expanded=(i == 1)):
                        st.markdown(f"**Hybrid Score:** `{h_score:.3f}` = 20% Keyword + 80% Semantic")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("BERT Semantic", f"{b_score:.3f}")
                        with col2:
                            st.metric("Keyword Match", f"{s_score:.3f}")
                        st.caption(f"Expertise: {features[0]:.2f} | Utility: {features[1]:.2f}")
                        st.markdown("---")
                        st.text(body[:400] + ("..." if len(body) > 400 else ""))

            with st.expander("Performance Metrics"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Latency", f"{latency:.1f}ms")
                col2.metric("Retrieved", len(solr_results))
                col3.metric("Alpha", cfg["search_tuning"]["alpha"])

else:
    st.info("Enter a query above to compare traditional keyword search vs AI-powered semantic search")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Try These Queries")
        st.code("docker deployment issues")
        st.code("improve app performance")
        st.code("best practices REST API")
        st.code("python async tutorial")

    with col2:
        st.markdown("### What You'll See")
        st.markdown("""
        - **Left:** Traditional keyword matching
        - **Right:** AI understands intent & context
        - **Rankings differ:** Better answers rise to top
        - **Scores differ:** We optimize for relevance, not score magnitude
        """)

    with col3:
        st.markdown("### Why It Matters")
        st.markdown("""
        - Finds answers even with different wording
        - Ranks by expertise & utility
        - Understands synonyms & context
        - 80% semantic + 20% keywords = better results
        """)

    with st.expander("Technical Details: How it works"):
        st.markdown("""
        **Stage 1: Fast Retrieval** - Apache Solr (BM25) retrieves top 50 candidates in <10ms
        
        **Stage 2: Semantic Re-Ranking** - BERT Cross-Encoder scores query-document relevance
        
        **Stage 3: Hybrid Fusion** - Weighted combination optimizes for both:
        ```python
        Hybrid_Score = (0.2 × Keyword_Score) + (0.8 × BERT_Score)
        ```
        
        **Why Hybrid Scores Can Be Lower:** Because we're optimizing RANKING, not raw scores. 
        A document with Solr=1.0 but BERT=0.6 gets: 0.2×1.0 + 0.8×0.6 = 0.68. 
        That's fine—what matters is the **order** improves!
        
        **Features Used:**
        - **Expertise Score**: User karma and subreddit activity
        - **Utility Score**: Code blocks, solution keywords, upvote ratio
        """)
