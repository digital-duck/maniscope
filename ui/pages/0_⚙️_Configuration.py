"""
Configuration Management Page

Allows users to manage application settings from config.py without editing code:
- Dataset configurations
- ReRanker selections
- Default parameters
- Metrics settings
- UI preferences
"""

import streamlit as st
import json
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    DATASETS,
    RE_RANKERS,
    RERANKER_SHORT_MAP,
    DEFAULT_MANISCOPE_K,
    DEFAULT_MANISCOPE_ALPHA,
    DEFAULT_MANISCOPE_VERSION,
    MANISCOPE_VERSIONS,
    OPENROUTER_MODELS,
    OLLAMA_MODELS,
    METRICS_TO_PLOT,
    DEFAULT_METRICS,
    COLORS
)

st.set_page_config(page_title="Configuration", layout="wide", page_icon="⚙️")

st.header("⚙️ Configuration")
st.markdown("Manage application settings without editing code. Changes are temporary (session only).")

# Initialize session state for configuration
if 'config_datasets' not in st.session_state:
    st.session_state['config_datasets'] = DATASETS.copy()
if 'config_rerankers' not in st.session_state:
    st.session_state['config_rerankers'] = RE_RANKERS.copy()
if 'config_maniscope_k' not in st.session_state:
    st.session_state['config_maniscope_k'] = DEFAULT_MANISCOPE_K
if 'config_maniscope_alpha' not in st.session_state:
    st.session_state['config_maniscope_alpha'] = DEFAULT_MANISCOPE_ALPHA
if 'config_embedding_model' not in st.session_state:
    st.session_state['config_embedding_model'] = 'all-MiniLM-L6-v2'  # Default
if 'config_optimization_level' not in st.session_state:
    st.session_state['config_optimization_level'] = 'v2o'  # Default to recommended v2o
if 'config_rag_llm_provider' not in st.session_state:
    st.session_state['config_rag_llm_provider'] = 'OpenRouter'
if 'config_rag_llm_model' not in st.session_state:
    st.session_state['config_rag_llm_model'] = 'google/gemini-2.0-flash-lite-001'
if 'config_rag_top_k' not in st.session_state:
    st.session_state['config_rag_top_k'] = 3
if 'config_rag_api_key' not in st.session_state:
    st.session_state['config_rag_api_key'] = os.getenv("OPENROUTER_API_KEY", "")

# ============================================================================
# TABS
# ============================================================================

tab3, tab2, tab1, tab4, tab5 = st.tabs([
    "⚡ Defaults",
    "📊 Datasets",
    "🤖 ReRankers",
    "📈 Metrics",
    "🎨 UI Settings"
])

# ============================================================================
# TAB 1: DATASETS
# ============================================================================

with tab1:
    st.markdown("### Dataset Management")
    st.markdown("Configure which datasets are available in the application.")

    # Current datasets
    st.markdown("#### Current Datasets")

    col1, col2 = st.columns([3, 1])

    with col1:
        for i, dataset in enumerate(st.session_state['config_datasets']):
            with st.expander(f"{dataset['name']} ({dataset['queries']} queries)", expanded=False):
                cols = st.columns([2, 1, 1, 1, 2])

                with cols[0]:
                    st.text_input("Name", value=dataset['name'], key=f"ds_name_{i}", disabled=True)

                with cols[1]:
                    st.text_input("Short", value=dataset['short'], key=f"ds_short_{i}", disabled=True)

                with cols[2]:
                    st.number_input("Queries", value=dataset['queries'], key=f"ds_queries_{i}", disabled=True)

                with cols[3]:
                    priority_options = [0, 1, 2]
                    st.selectbox("Priority", priority_options, index=priority_options.index(dataset['priority']), key=f"ds_priority_{i}", disabled=True)

                with cols[4]:
                    st.text_input("File", value=dataset['file'], key=f"ds_file_{i}", disabled=True)

                st.text_area("Description", value=dataset['description'], key=f"ds_desc_{i}", height=60, disabled=True)

    with col2:
        st.markdown("#### Quick Stats")
        st.metric("Total Datasets", len(st.session_state['config_datasets']))
        st.metric("Priority 0 (Quick)", sum(1 for d in st.session_state['config_datasets'] if d['priority'] == 0))
        st.metric("Priority 1 (arXiv)", sum(1 for d in st.session_state['config_datasets'] if d['priority'] == 1))
        st.metric("Priority 2 (NeurIPS)", sum(1 for d in st.session_state['config_datasets'] if d['priority'] == 2))
        st.metric("Total Queries", sum(d['queries'] for d in st.session_state['config_datasets']))

    # Dataset order info
    st.markdown("---")
    st.info("""
    **ℹ️ Dataset Order:** Datasets are displayed in the order defined in `config.py`.
    To reorder, edit the DATASETS list in `/src/app/config.py`.

    **Priority Levels:**
    - Priority 0: Quick test datasets (10 queries, fast iteration)
    - Priority 1: arXiv/ICML paper (primary benchmarks)
    - Priority 2: NeurIPS/extended evaluation (secondary benchmarks)
    """)

    # Export current config
    st.markdown("#### Export Configuration")
    config_json = json.dumps(st.session_state['config_datasets'], indent=2)
    st.download_button(
        label="📥 Download Dataset Config (JSON)",
        data=config_json,
        file_name="datasets_config.json",
        mime="application/json"
    )

# ============================================================================
# TAB 2: RERANKERS
# ============================================================================

with tab2:
    st.markdown("### ReRanker Configuration")
    st.markdown("Manage available reranking models and their settings.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Available ReRankers")

        for i, reranker in enumerate(st.session_state['config_rerankers']):
            cols = st.columns([3, 2, 1])

            with cols[0]:
                st.text_input("ReRanker", value=reranker, key=f"rr_{i}", disabled=True)

            with cols[1]:
                short_name = RERANKER_SHORT_MAP.get(reranker, "unknown")
                st.text_input("Short Name", value=short_name, key=f"rr_short_{i}", disabled=True)

            with cols[2]:
                color = COLORS.get(reranker, "#999999")
                st.color_picker("Color", value=color, key=f"rr_color_{i}", disabled=True)

    with col2:
        st.markdown("#### Model Types")
        st.markdown("""
        **Maniscope** (Geodesic)
        - Manifold-based reranking
        - k-NN graph construction
        - Fast CPU inference

        **LLM-Reranker** (API)
        - Via OpenRouter or Ollama
        - Flexible model selection
        - High accuracy, high cost

        **Jina Reranker v2** (Cross-encoder)
        - Transformer-based
        - Joint query-doc encoding
        - Balanced performance

        **BGE-M3** (Bi-encoder)
        - BAAI embedding model
        - Multi-lingual support
        - Efficient inference
        """)

    st.markdown("---")

    # LLM Model Configuration
    st.markdown("#### LLM Model Options")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**OpenRouter Models**")
        for model in OPENROUTER_MODELS:
            st.code(model, language=None)

    with col2:
        st.markdown("**Ollama Models**")
        for model in OLLAMA_MODELS:
            st.code(model, language=None)

# ============================================================================
# TAB 3: DEFAULT PARAMETERS
# ============================================================================

with tab3:
    st.markdown("### Default Parameter Settings")
    st.markdown("Configure default values for benchmarking parameters.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Maniscope Parameters")

        new_k = st.slider(
            "Default k (neighbors)",
            min_value=3,
            max_value=20,
            value=st.session_state['config_maniscope_k'],
            step=1,
            help="Number of nearest neighbors for k-NN graph construction"
        )

        new_alpha = st.slider(
            "Default α (hybrid weight)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state['config_maniscope_alpha'],
            step=0.1,
            help="Balance between geodesic (0.0) and cosine (1.0) similarity"
        )

        st.markdown("---")

        st.markdown("#### Embedding Model")

        embedding_models = [
            'all-MiniLM-L6-v2',
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        ]
        embedding_labels = [
            'all-MiniLM-L6-v2 (Default, 22M params)',
            'Sentence-BERT Multilingual (MPNet, 278M params)'
        ]

        current_index = embedding_models.index(st.session_state['config_embedding_model']) if st.session_state['config_embedding_model'] in embedding_models else 0

        new_embedding = st.selectbox(
            "Embedding Model",
            options=embedding_models,
            format_func=lambda x: embedding_labels[embedding_models.index(x)],
            index=current_index,
            help="Model used to encode queries and documents"
        )

        st.markdown("---")

        st.markdown("#### Optimization Level")

        optimization_levels = ['v0', 'v1', 'v2', 'v3', 'v2o']
        optimization_labels = {
            'v0': 'v0 - Baseline (CPU, no caching)',
            'v1': 'v1 - GPU + Graph Caching (3× faster)',
            'v2': 'v2 - Full Optimization (5× faster)',
            'v3': 'v3 - Persistent Cache + Query Cache (variable)',
            'v2o': 'v2o - Ultimate ⭐ RECOMMENDED (20-235× faster)'
        }

        current_opt_index = optimization_levels.index(st.session_state['config_optimization_level']) if st.session_state['config_optimization_level'] in optimization_levels else 4  # Default to v2o

        new_opt_level = st.selectbox(
            "Optimization Level",
            options=optimization_levels,
            format_func=lambda x: optimization_labels[x],
            index=current_opt_index,
            help="Performance optimization level for Maniscope"
        )

        st.markdown("---")

        st.markdown("#### RAG Evaluation Settings")

        rag_provider = st.radio(
            "RAG LLM Provider",
            options=["OpenRouter", "Ollama"],
            index=0 if st.session_state['config_rag_llm_provider'] == 'OpenRouter' else 1,
            horizontal=True,
            help="Provider for RAG answer generation"
        )

        if rag_provider == "OpenRouter":
            # Check if env var exists
            env_key = os.getenv("OPENROUTER_API_KEY", "")
            default_key = st.session_state.get('config_rag_api_key', env_key)

            rag_api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                value=default_key,
                help="API key for OpenRouter (falls back to OPENROUTER_API_KEY env var if not set)"
            )

            # Show status if using env var
            if not rag_api_key and env_key:
                st.success("✅ Using OPENROUTER_API_KEY from environment variable")
            elif not rag_api_key and not env_key:
                st.warning("⚠️ No API key set. Please enter API key or set OPENROUTER_API_KEY environment variable.")

            rag_model = st.selectbox(
                "RAG LLM Model",
                options=OPENROUTER_MODELS,
                index=OPENROUTER_MODELS.index(st.session_state['config_rag_llm_model']) if st.session_state['config_rag_llm_model'] in OPENROUTER_MODELS else 0,
                help="LLM model for generating RAG responses"
            )
        else:
            rag_api_key = None
            rag_model = st.selectbox(
                "RAG LLM Model",
                options=OLLAMA_MODELS,
                index=OLLAMA_MODELS.index(st.session_state['config_rag_llm_model']) if st.session_state['config_rag_llm_model'] in OLLAMA_MODELS else 0,
                help="LLM model for generating RAG responses"
            )

        rag_top_k = st.slider(
            "RAG Top-K Documents",
            min_value=1,
            max_value=10,
            value=st.session_state['config_rag_top_k'],
            step=1,
            help="Number of top-ranked documents to use as context for RAG"
        )

        if st.button("💾 Save All Settings", type="primary"):
            st.session_state['config_maniscope_k'] = new_k
            st.session_state['config_maniscope_alpha'] = new_alpha
            st.session_state['config_embedding_model'] = new_embedding
            st.session_state['config_optimization_level'] = new_opt_level
            st.session_state['config_rag_llm_provider'] = rag_provider
            st.session_state['config_rag_llm_model'] = rag_model
            st.session_state['config_rag_top_k'] = rag_top_k
            if rag_api_key:
                st.session_state['config_rag_api_key'] = rag_api_key
            st.success(f"✅ Saved: k={new_k}, α={new_alpha}, model={new_embedding}, opt={new_opt_level}, RAG={rag_model}, top-k={rag_top_k}")
            st.rerun()

    with col2:
        st.markdown("#### Current Values")
        st.metric("Maniscope k", st.session_state['config_maniscope_k'])
        st.metric("Maniscope α", st.session_state['config_maniscope_alpha'])
        st.metric("Embedding Model", st.session_state['config_embedding_model'].split('/')[-1])
        st.metric("Optimization", st.session_state['config_optimization_level'])
        st.metric("RAG LLM Model", st.session_state['config_rag_llm_model'].split('/')[-1])
        st.metric("RAG Top-K", st.session_state['config_rag_top_k'])

        st.markdown("---")

        st.markdown("#### Recommendations")
        st.info("""
        **k (neighbors):**
        - k=3-5: Small, sparse manifolds
        - k=5-7: Balanced (recommended)
        - k=9-15: Dense, complex structures

        **α (hybrid weight):**
        - α=0.0: Pure geodesic (manifold structure only)
        - α=0.3-0.5: Balanced (recommended)
        - α=1.0: Pure cosine (baseline)

        **Embedding Model:**
        - all-MiniLM-L6-v2: Fast, lightweight (default)
        - Sentence-BERT Multilingual: Better quality, slower

        **Optimization Level:**
        - v0: Baseline for accuracy validation
        - v1: GPU + graph caching (3× speedup)
        - v2: Full optimization with FAISS (5× speedup)
        - v3: Persistent cache, CPU-friendly (1-10× variable)
        - v2o: ⭐ **RECOMMENDED** - All optimizations (20-235× speedup)
          * Best for: Production, maximum performance
          * Combines: GPU + FAISS + scipy + persistent cache
          * Benchmarks: MS MARCO 132ms → 0.58ms (229×)
        """)

    st.markdown("---")
    st.warning("⚠️ **Note:** Changes are session-specific. To persist changes, edit `config.py` directly.")

# ============================================================================
# TAB 4: METRICS
# ============================================================================

with tab4:
    st.markdown("### Metrics Configuration")
    st.markdown("Configure which metrics are calculated and displayed.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Metrics for Plotting")
        st.markdown("Displayed in charts and visualizations:")
        for metric in METRICS_TO_PLOT:
            st.code(metric, language=None)

        st.markdown("---")

        st.markdown("#### Metric Descriptions")
        st.markdown("""
        - **MRR**: Mean Reciprocal Rank (1/rank of first relevant)
        - **NDCG@K**: Normalized Discounted Cumulative Gain at K
        - **MAP**: Mean Average Precision
        - **P@K**: Precision at K (fraction of top-K that are relevant)
        - **R@K**: Recall at K (fraction of all relevant in top-K)
        """)

    with col2:
        st.markdown("#### All Calculated Metrics")
        st.markdown("Computed during evaluation:")
        for metric in DEFAULT_METRICS:
            st.code(metric, language=None)

        st.markdown("---")

        st.info("""
        **ℹ️ Metric Selection:**
        - All metrics in DEFAULT_METRICS are computed
        - Only METRICS_TO_PLOT are shown in charts
        - Full metrics available in exported files
        """)

# ============================================================================
# TAB 5: UI SETTINGS
# ============================================================================

with tab5:
    st.markdown("### UI Preferences")
    st.markdown("Customize the appearance of charts and visualizations.")

    st.markdown("#### ReRanker Colors")

    col1, col2 = st.columns(2)

    with col1:
        for reranker, color in COLORS.items():
            st.color_picker(
                f"{reranker}",
                value=color,
                key=f"color_{reranker}",
                disabled=True
            )

    with col2:
        st.markdown("#### Color Scheme")
        st.markdown("""
        Colors are used in:
        - Bar charts comparing rerankers
        - Line plots showing metric trends
        - Scatter plots for per-query analysis
        - Heatmaps for correlation matrices

        **Default Palette:**
        - Maniscope: Red (#FF6B6B)
        - LLM-Reranker: Teal (#4ECDC4)
        - Jina: Blue (#45B7D1)
        - BGE-M3: Green (#96CEB4)
        """)

    st.markdown("---")

    st.markdown("#### Layout Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Current Layout**")
        st.code("layout='wide'", language="python")
        st.markdown("Pages use wide layout for better chart visibility")

    with col2:
        st.markdown("**Page Icon**")
        st.code("page_icon='📊'", language="python")
        st.markdown("Each page has a custom emoji icon")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📁 Config File")
    st.code("src/app/config.py", language=None)
    st.markdown("Edit directly for persistent changes")

with col2:
    st.markdown("#### 🔄 Reset to Defaults")
    if st.button("Reset All Settings", type="secondary"):
        st.session_state['config_datasets'] = DATASETS.copy()
        st.session_state['config_rerankers'] = RE_RANKERS.copy()
        st.session_state['config_maniscope_k'] = DEFAULT_MANISCOPE_K
        st.session_state['config_maniscope_alpha'] = DEFAULT_MANISCOPE_ALPHA
        st.success("✅ Reset to default configuration")
        st.rerun()

with col3:
    st.markdown("#### 💾 Export All")
    all_config = {
        "datasets": st.session_state['config_datasets'],
        "rerankers": st.session_state['config_rerankers'],
        "maniscope": {
            "k": st.session_state['config_maniscope_k'],
            "alpha": st.session_state['config_maniscope_alpha'],
            "embedding_model": st.session_state['config_embedding_model'],
            "optimization_level": st.session_state['config_optimization_level']
        },
        "metrics": {
            "to_plot": METRICS_TO_PLOT,
            "default": DEFAULT_METRICS
        }
    }
    st.download_button(
        label="📥 Download Full Config",
        data=json.dumps(all_config, indent=2),
        file_name="app_config.json",
        mime="application/json"
    )
