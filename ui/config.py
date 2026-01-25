"""
Global Configuration for RAG-ReRanker Application

Shared constants and configurations across all pages.
"""

# ============================================================================
# Dataset Configurations
# ============================================================================

DATASETS = [
    # Quick Test Datasets (10 queries each)
    {
        "name": "AorB (Quick)",
        "file": "dataset-aorb-10.json",
        "short": "aorb",
        "queries": 10,
        "priority": 0,  # Quick test
        "description": "🚀 Quick test - Semantic disambiguation"
    },
    {
        "name": "SciFact (Quick)",
        "file": "dataset-scifact-10.json",
        "short": "sci",
        "queries": 10,
        "priority": 0,  # Quick test
        "description": "🚀 Quick test - Scientific claim verification"
    },
    {
        "name": "MS MARCO (Quick)",
        "file": "dataset-msmarco-10.json",
        "short": "marco",
        "queries": 10,
        "priority": 0,  # Quick test
        "description": "🚀 Quick test - Web search queries"
    },
    {
        "name": "TREC-COVID (Quick)",
        "file": "dataset-trec-covid-10.json",
        "short": "covid",
        "queries": 10,
        "priority": 0,  # Quick test
        "description": "🚀 Quick test - COVID-19 research"
    },
    {
        "name": "ArguAna (Quick)",
        "file": "dataset-arguana-10.json",
        "short": "arg",
        "queries": 10,
        "priority": 0,  # Quick test
        "description": "🚀 Quick test - Counter-argument retrieval"
    },
    {
        "name": "FiQA (Quick)",
        "file": "dataset-fiqa-10.json",
        "short": "fiqa",
        "queries": 10,
        "priority": 0,  # Quick test
        "description": "🚀 Quick test - Financial question answering"
    },
    # Full Datasets
    {
        "name": "AorB",
        "file": "dataset-aorb.json",
        "short": "aorb",
        "queries": 50,
        "priority": 1,  # arXiv/ICML - Novel disambiguation benchmark
        "description": "Semantic disambiguation (Python/Apple/Java/Mercury/Jaguar/Flow)"
    },
    {
        "name": "SciFact",
        "file": "dataset-scifact.json",
        "short": "sci",
        "queries": 100,
        "priority": 1,  # arXiv/ICML
        "description": "Scientific claim verification"
    },
    {
        "name": "MS MARCO",
        "file": "dataset-msmarco.json",
        "short": "marco",
        "queries": 200,
        "priority": 1,  # arXiv/ICML
        "description": "Web search queries"
    },
    {
        "name": "TREC-COVID",
        "file": "dataset-trec-covid.json",
        "short": "covid",
        "queries": 50,
        "priority": 1,  # arXiv/ICML
        "description": "COVID-19 research"
    },
    {
        "name": "ArguAna",
        "file": "dataset-arguana.json",
        "short": "arg",
        "queries": 100,
        "priority": 2,  # NeurIPS
        "description": "Counter-argument retrieval"
    },
    {
        "name": "FiQA",
        "file": "dataset-fiqa.json",
        "short": "fiqa",
        "queries": 100,
        "priority": 2,  # NeurIPS
        "description": "Financial question answering"
    },
    # New BEIR Datasets (Weekend Expansion - Multilingual Focus)
    {
        "name": "NFCorpus",
        "file": "dataset-nfcorpus.json",
        "short": "nfc",
        "queries": 323,
        "priority": 1,  # arXiv/ICML v4.0
        "description": "Medical/nutrition information retrieval (English)"
    },
    {
        "name": "FEVER",
        "file": "dataset-fever.json",
        "short": "fever",
        "queries": 200,
        "priority": 1,  # arXiv/ICML v4.0
        "description": "Fact extraction and verification (English)"
    },
    # Multilingual datasets - Not yet implemented
    # {
    #     "name": "MIRACL",
    #     "file": "dataset-miracl.json",
    #     "short": "miracl",
    #     "queries": 200,
    #     "priority": 1,  # arXiv/ICML v4.0 - Multilingual
    #     "description": "🌍 Multilingual Information Retrieval Across Continents and Languages"
    # },
    # {
    #     "name": "Mr. TyDi",
    #     "file": "dataset-mrtydi.json",
    #     "short": "mrtydi",
    #     "queries": 200,
    #     "priority": 1,  # arXiv/ICML v4.0 - Multilingual
    #     "description": "🌍 Multilingual Typologically Diverse Question Answering"
    # },
]

# ============================================================================
# ReRanker Configurations
# ============================================================================

RE_RANKERS = [
    "Maniscope",  # Uses version from sidebar dropdown
    "Maniscope_v0",  # Explicit baseline (CPU, no caching) - 115ms avg
    "Maniscope_v2o",  # Explicit v2o (GPU + all optimizations) - 0.4-20ms avg
    "HNSW",  # Baseline - CPU only, no caching
    "HNSW_v2o",  # Optimized - GPU + caching (3-10× faster)
    "Jina Reranker v2",  # Baseline - CPU only, no caching
    "Jina Reranker v2_v2o",  # Optimized - GPU + caching (3-5× faster)
    "BGE-M3",  # Baseline - no explicit optimizations
    "BGE-M3_v2o",  # Optimized - GPU + caching (2-3× faster)
    "LLM-Reranker",
    # "Qwen-1.5B"  # Commented out - poor performance (MRR 0.3785)
]

# Short names for filenames
RERANKER_SHORT_MAP = {
    'Maniscope': 'mani',
    'Maniscope_v0': 'mani_v0',
    'Maniscope_v2o': 'mani_v2o',
    'HNSW': 'hnsw',
    'HNSW_v2o': 'hnsw_v2o',
    'Jina Reranker v2': 'jina',
    'Jina Reranker v2_v2o': 'jina_v2o',
    'BGE-M3': 'bge',
    'BGE-M3_v2o': 'bge_v2o',
    'LLM-Reranker': 'llm',
    'Qwen-1.5B': 'qwen'
}

# ============================================================================
# Default Maniscope Configuration
# ============================================================================

DEFAULT_MANISCOPE_K = 5
DEFAULT_MANISCOPE_ALPHA = 0.5

# Maniscope Optimization Level Selection
# Options: "v0", "v1", "v2", "v3", "v2o"
# - v0: Baseline (CPU, no caching) - 115ms avg
# - v1: GPU + Graph Caching (3x faster) - 40ms avg
# - v2: Full Optimization (5x faster) - 20-25ms avg
# - v3: Persistent Cache + Query Cache (1-10x faster, cache-dependent) - 10-115ms avg
# - v2o: RECOMMENDED - Ultimate (20-235x faster) - 0.4-20ms avg
DEFAULT_MANISCOPE_VERSION = "v2o"

# Version descriptions for UI
MANISCOPE_VERSIONS = {
    "v0": {
        "name": "v0 - Baseline (CPU, no caching)",
        "description": "CPU-only, rebuilds graph every query",
        "latency": "115ms",
        "speedup": "1.0x",
        "best_for": "Baseline reference"
    },
    "v1": {
        "name": "v1 - GPU + Graph Caching (3x faster)",
        "description": "GPU embeddings + cached graph between queries",
        "latency": "40ms",
        "speedup": "3.0x",
        "best_for": "GPU available, multiple queries"
    },
    "v2": {
        "name": "v2 - Full Optimization (5x faster)",
        "description": "GPU + scipy + FAISS + vectorized scoring",
        "latency": "20-25ms",
        "speedup": "5.0x",
        "best_for": "Cold cache, first-run performance"
    },
    "v3": {
        "name": "v3 - Persistent Cache + Query Cache",
        "description": "Disk cache (persistent) + LRU query cache + heap optimization",
        "latency": "10-115ms (variable)",
        "speedup": "1-10x (cache-dependent)",
        "best_for": "Repeated experiments, grid search, CPU-only"
    },
    "v2o": {
        "name": "v2o - Ultimate Optimization ⭐ RECOMMENDED",
        "description": "ALL optimizations: GPU + FAISS + scipy + persistent cache + query cache",
        "latency": "0.4-20ms",
        "speedup": "20-235x",
        "best_for": "Production, maximum performance, all scenarios"
    }
}

# ============================================================================
# Embedding Models Configuration (Weekend Expansion - Robustness Testing)
# ============================================================================

EMBEDDING_MODELS = [
    {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "short": "ml-minilm",
        "dimensions": 384,
        "description": "🌍 Multilingual MiniLM - 50+ languages, compact and fast",
        "languages": "50+ languages",
        "priority": 1  # New baseline - multilingual
    },
    {
        "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "short": "ml-mpnet",
        "dimensions": 768,
        "description": "🌍 Multilingual MPNet - 50+ languages, higher quality",
        "languages": "50+ languages",
        "priority": 1  # Weekend expansion - multilingual
    },
    {
        "name": "intfloat/multilingual-e5-large",
        "short": "ml-e5",
        "dimensions": 1024,
        "description": "🌍 Multilingual E5 Large - 100+ languages, SOTA performance",
        "languages": "100+ languages",
        "priority": 1  # Weekend expansion - multilingual
    },
    {
        "name": "BAAI/bge-m3",
        "short": "bge-m3",
        "dimensions": 1024,
        "description": "🌍 BGE-M3 - 100+ languages, multi-functionality, multi-granularity",
        "languages": "100+ languages",
        "priority": 1  # Weekend expansion - multilingual
    },
]

# Default embedding model for Maniscope (now multilingual)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ============================================================================
# LLM ReRanker Configuration
# ============================================================================

OPENROUTER_MODELS = [
    "google/gemini-2.0-flash-lite-001",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3.5-sonnet",
    "openai/chatgpt-4o-latest",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct"
]

OLLAMA_MODELS = [
    "llama3.1:latest",
    "deepseek-r1:7b",
    "qwen2.5:latest"
]

DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"

# ============================================================================
# Metrics Configuration
# ============================================================================

METRICS_TO_PLOT = ["MRR", "NDCG@3", "NDCG@10", "MAP", "P@3", "R@10"]
DEFAULT_METRICS = ["MRR", "NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10", "P@1", "P@3", "P@5", "P@10", "R@10", "MAP"]

# ============================================================================
# Output Directory Configuration
# ============================================================================

from pathlib import Path

# Base output directory (root/output)
OUTPUT_BASE_DIR = Path(__file__).parent.parent / "output"

# Subdirectories for organized output
OUTPUT_DIRS = {
    "eval_rag": OUTPUT_BASE_DIR / "eval-rag",        # Eval ReRanker page
    "benchmark": OUTPUT_BASE_DIR / "benchmark",      # Benchmark & Batch Benchmark pages
    "grid_search": OUTPUT_BASE_DIR / "grid-search",  # Grid Search page
    "metrics": OUTPUT_BASE_DIR / "metrics",          # Analytics page
}

def ensure_output_dirs():
    """Create all output subdirectories if they don't exist."""
    OUTPUT_BASE_DIR.mkdir(exist_ok=True)
    for dir_path in OUTPUT_DIRS.values():
        dir_path.mkdir(exist_ok=True, parents=True)

# ============================================================================
# UI Configuration
# ============================================================================

PAGE_ICON = "📊"
PAGE_LAYOUT = "wide"

# Color scheme for charts (optional)
COLORS = {
    "Maniscope": "#FF6B6B",
    "Maniscope_v0": "#FF9999",  # Light red - baseline
    "Maniscope_v2o": "#CC0000",  # Dark red - ultimate optimization
    "LLM-Reranker": "#4ECDC4",
    "Jina Reranker v2": "#45B7D1",
    "Jina Reranker v2_v2o": "#1E90FF",  # Dodger blue - v2o variant
    "BGE-M3": "#96CEB4",
    "BGE-M3_v2o": "#4CAF50",  # Green - v2o variant
    "HNSW": "#FFA07A",  # Light salmon - graph-based baseline
    "HNSW_v2o": "#FF6347"  # Tomato - graph-based v2o
}
