"""
RAG Reranker Evaluation Lab

A comprehensive tool for evaluating and comparing reranking approaches using MTEB datasets.
"""

import streamlit as st

st.set_page_config(page_title="RAG Reranker Lab", layout="wide", page_icon="🚀")

st.header("🚀 Welcome to RAG Reranker Evaluation Lab")
st.markdown("""
(1) **Standardized benchmarks** using [MTEB](https://pypi.org/project/mteb/) and [BEIR](https://pypi.org/project/beir/) datasets
(2) **Reproducible results** with configurable random seeds
(3) **Comprehensive metrics** following IR best practices
(4) **Publication-ready exports** with detailed statistics
(5) **Flexibility** to add custom models and datasets

""")

workflow_cols = st.columns(3)

with workflow_cols[0]:
    st.markdown("### 1️⃣ Data Manager")
    st.markdown("""
    **Load datasets:**
    - MTEB datasets (SciFact, TRECCOVID)
    - BEIR datasets
    - Synthetic data generation
    - View statistics and previews
    """)

with workflow_cols[1]:
    st.markdown("### 2️⃣ Run Benchmark")
    st.markdown("""
    **Execute evaluation:**
    - Supported ReRankers: 
        - **Maniscope** (this work) Based on Geodesic Distances on k-NN Manifolds
        - **LLM-Reranker** (via API)
        Flexible LLM scoring via Ollama or OpenRouter
        - **Jina Reranker v2** 
        - **BGE-M3** (BAAI/bge-reranker-v2-m3)
        Encoder-based cross-encoder reranker
    - Configure batch processing
    - Monitor progress in real-time
    - Save results automatically
    """)

with workflow_cols[2]:
    st.markdown("### 3️⃣ Analytics")
    st.markdown("""
    **Analyze results:**
    - Metric comparisons:
        - **MRR** - Mean Reciprocal Rank
        - **NDCG@K** - Normalized Discounted Cumulative Gain
        - **Precision@K** - Precision at K
        - **Recall@K** - Recall at K
        - **MAP** - Mean Average Precision
        - **Latency & Throughput** - Performance metrics
    - Per-query analysis
    - Latency analysis
    - Export to Excel/CSV
    """)


with st.expander("📖 User Guide", expanded=False):
    st.markdown("""
    ### Step 0: Setup Dependency
    `pip install -r requirements`
                
    ### Step 1: Prepare Dataset

    **Option A: Use MTEB Dataset (Recommended for Research)**
    1. Prepare dataset using `prep_scifact.py` or `prep_mteb_dataset.py`
    2. Upload JSON file in Data Manager page
    3. Review statistics and preview

    **Option B: Generate Synthetic Data (Quick Testing)**
    1. Enter OpenRouter API key
    2. Provide seed texts
    3. Generate and review

    ---

    ### Step 2: Run Benchmark

    1. Select models to evaluate (BGE-M3, Qwen-1.5B, LLM-Reranker)
    2. Configure execution settings (batch size, max queries)
    3. Click "Start Benchmark"
    4. Monitor progress and view live metrics
    5. Results are automatically saved

    ---

    ### Step 3: Analyze Results

    1. Navigate to Analytics page
    2. Explore metric comparisons
    3. Analyze per-query performance
    4. Review latency statistics
    5. Export data for publication

    ---

    ### Step 4: Export & Publish

    - Export to Excel with multiple sheets
    - Generate CSV for further analysis
    - Download raw JSON for custom processing
    """)

    st.markdown("---")

    st.markdown("## 📚 Resources")

    resource_cols = st.columns(3)

    with resource_cols[0]:
        st.markdown("### Dataset Tools")
        st.markdown("""
        - `data/prep_scifact.py` - SciFact preparation
        - `data/prep_mteb_dataset.py` - General MTEB prep
        - `data/README.md` - Full documentation
        """)

    with resource_cols[1]:
        st.markdown("### Documentation")
        st.markdown("""
        - `readme-fix-gemini.md` - Implementation plan
        - MTEB: embeddings-benchmark.github.io
        - FlagEmbedding: github.com/FlagOpen/FlagEmbedding
        """)

    with resource_cols[2]:
        st.markdown("### Architecture")
        st.markdown("""
        - `utils/metrics.py` - IR metrics
        - `utils/data_loader.py` - Dataset loading
        - `utils/models.py` - Model interface
        - `utils/visualization.py` - Charts
        """)

    st.markdown("---")

st.info("👈 **Use the sidebar to navigate** through the evaluation pipeline")


