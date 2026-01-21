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
]

# ============================================================================
# ReRanker Configurations
# ============================================================================

RE_RANKERS = [
    "Maniscope",
    "LLM-Reranker",
    "Jina Reranker v2",
    "BGE-M3",
    # "Qwen-1.5B"  # Commented out - poor performance (MRR 0.3785)
]

# Short names for filenames
RERANKER_SHORT_MAP = {
    'Maniscope': 'mani',
    'BGE-M3': 'bge',
    'LLM-Reranker': 'llm',
    'Jina Reranker v2': 'jina',
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
# LLM ReRanker Configuration
# ============================================================================

OPENROUTER_MODELS = [
    "google/gemini-2.0-flash-lite-001",
    "anthropic/claude-3.5-haiku"
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
DEFAULT_METRICS = ["MRR", "NDCG@1", "NDCG@3", "NDCG@10", "P@1", "P@3", "P@10", "R@10", "MAP"]

# ============================================================================
# UI Configuration
# ============================================================================

PAGE_ICON = "📊"
PAGE_LAYOUT = "wide"

# Color scheme for charts (optional)
COLORS = {
    "Maniscope": "#FF6B6B",
    "LLM-Reranker": "#4ECDC4",
    "Jina Reranker v2": "#45B7D1",
    "BGE-M3": "#96CEB4"
}
