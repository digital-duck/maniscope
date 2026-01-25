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
    # === LIGHTWEIGHT & FAST (Priority 0 - Quick Testing) ===
    {
        "name": "all-MiniLM-L6-v2",
        "short": "minilm-l6",
        "dimensions": 384,
        "description": "⚡ All-MiniLM L6 v2 - 22M params, fastest inference",
        "languages": "English",
        "priority": 0,
        "params": "22M",
        "speed": "fastest"
    },
    {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "short": "ml-minilm",
        "dimensions": 384,
        "description": "🌍 Multilingual MiniLM - 50+ languages, compact and fast",
        "languages": "50+ languages",
        "priority": 1,
        "params": "118M",
        "speed": "fast"
    },
    {
        "name": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "short": "distiluse-ml",
        "dimensions": 512,
        "description": "🌍 Universal Sentence Encoder Multilingual - 135M params",
        "languages": "15+ languages",
        "priority": 1,
        "params": "135M",
        "speed": "fast"
    },

    # === MULTILINGUAL LEADERS (Priority 1 - Production Ready) ===
    {
        "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "short": "ml-mpnet",
        "dimensions": 768,
        "description": "🌍 Sentence-BERT Multilingual - MPNet, 50+ languages, proven baseline",
        "languages": "50+ languages",
        "priority": 1,
        "params": "278M",
        "speed": "medium"
    },
    {
        "name": "sentence-transformers/LaBSE",
        "short": "labse",
        "dimensions": 768,
        "description": "🌍 Language-agnostic BERT - 109 languages, sentence similarity expert",
        "languages": "109 languages",
        "priority": 1,
        "params": "471M",
        "speed": "medium"
    },
    {
        "name": "intfloat/multilingual-e5-large-instruct",
        "short": "e5-instruct",
        "dimensions": 1024,
        "description": "🌍 E5-Large-Instruct - 560M params, instruction-following, 100+ languages",
        "languages": "100+ languages",
        "priority": 1,
        "params": "560M",
        "speed": "medium"
    },

    # === CURRENT SOTA MODELS (Priority 2 - 2025 Leaders) ===
    {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "short": "qwen3-06b",
        "dimensions": 1024,
        "description": "🚀 Qwen3-Embedding-0.6B - MTEB #1 series, 600M params, 100+ languages",
        "languages": "100+ languages",
        "priority": 2,
        "params": "600M",
        "speed": "medium",
        "mteb_rank": "Top 5"
    },
    {
        "name": "google/embeddinggemma-300m",
        "short": "gemma-300m",
        "dimensions": 768,
        "description": "🚀 EmbeddingGemma - Google's 300M param model, 100+ languages, on-device AI",
        "languages": "100+ languages",
        "priority": 2,
        "params": "300M",
        "speed": "medium",
        "mteb_rank": "SOTA class"
    },
    {
        "name": "intfloat/e5-base-v2",
        "short": "e5-base",
        "dimensions": 768,
        "description": "🚀 E5-Base-v2 - 278M params, balanced accuracy-speed, no prefix required",
        "languages": "100+ languages",
        "priority": 2,
        "params": "278M",
        "speed": "fast",
        "mteb_rank": "High"
    },

    # === SPECIALIZED MODELS (Priority 3 - Research & Specific Use Cases) ===
    {
        "name": "bert-base-multilingual-cased",
        "short": "mbert",
        "dimensions": 768,
        "description": "🔬 mBERT - Multilingual BERT, 104 languages, research baseline",
        "languages": "104 languages",
        "priority": 3,
        "params": "178M",
        "speed": "medium"
    },
    {
        "name": "distilbert-base-multilingual-cased",
        "short": "distilbert-ml",
        "dimensions": 768,
        "description": "🔬 DistilBERT Multilingual - Lightweight mBERT variant",
        "languages": "104 languages",
        "priority": 3,
        "params": "134M",
        "speed": "fast"
    },
    {
        "name": "xlm-roberta-base",
        "short": "xlm-roberta",
        "dimensions": 768,
        "description": "🔬 XLM-RoBERTa - Cross-lingual model, 100+ languages",
        "languages": "100+ languages",
        "priority": 3,
        "params": "270M",
        "speed": "medium"
    },

    # === HIGH-END MODELS (Priority 4 - Advanced Research) ===
    {
        "name": "intfloat/multilingual-e5-large",
        "short": "e5-large",
        "dimensions": 1024,
        "description": "💎 E5-Large - 560M params, SOTA multilingual performance",
        "languages": "100+ languages",
        "priority": 4,
        "params": "560M",
        "speed": "slow",
        "mteb_rank": "Top 10"
    },
    {
        "name": "BAAI/bge-m3",
        "short": "bge-m3",
        "dimensions": 1024,
        "description": "💎 BGE-M3 - Multi-functionality, multi-granularity, 100+ languages",
        "languages": "100+ languages",
        "priority": 4,
        "params": "568M",
        "speed": "slow",
        "mteb_rank": "Top 5"
    },
]



# ============================================================================
# LLM ReRanker Configuration
# ============================================================================

# Validated OpenRouter models (sorted alphabetically)
OPENROUTER_MODELS = [
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3.5-sonnet",
    "cohere/command-r",
    "cohere/command-r-plus",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-coder",
    "google/gemini-2.0-flash-exp",
    "google/gemini-flash-1.5",
    "google/gemini-pro-1.5",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.2-3b-instruct:free",
    "microsoft/phi-3-medium-128k-instruct",
    "microsoft/phi-3-mini-128k-instruct",
    "microsoft/phi-3-mini-128k-instruct:free",
    "mistralai/mistral-7b-instruct",
    "mistralai/mistral-large",
    "mistralai/mistral-medium",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4",
    "openai/gpt-4-turbo",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "perplexity/llama-3.1-sonar-large-128k-online",
    "perplexity/llama-3.1-sonar-small-128k-online",
    "qwen/qwen-2-7b-instruct:free",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-7b-instruct",
]

# Ollama models (sorted alphabetically)
OLLAMA_MODELS = [
    "deepseek-r1:7b",
    "llama3.1:latest",
    "llama3.2:latest",
    "llama3.3:latest",
    "qwen2.5:latest"
]

DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"

# Default embedding model for Maniscope (balanced speed and quality)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/LaBSE" #"all-MiniLM-L6-v2"  # Fastest, most compatible
DEFAULT_LLM_MODEL = "openai/gpt-3.5-turbo"

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
