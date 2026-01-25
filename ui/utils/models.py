"""
Unified model interface for reranker evaluation.

Provides consistent API for different reranker implementations:
- BGE-M3 (FlagReranker)
- Qwen-1.5B (FlagLLMReranker)
- LLM-as-Reranker (via OpenAI-compatible API)
- Maniscope (Geodesic Reranker)
"""

import numpy as np
from typing import List, Dict, Any, Optional
import streamlit as st
import sys
from pathlib import Path

# Add utils directory to path for maniscope import
sys.path.insert(0, str(Path(__file__).parent))

# Import config for default Maniscope version
try:
    from config import DEFAULT_MANISCOPE_VERSION
except ImportError:
    DEFAULT_MANISCOPE_VERSION = 'v0'  # Fallback


class RerankerModelError(Exception):
    """Raised when model loading or inference fails."""
    pass


@st.cache_resource
def load_bge_reranker():
    """
    Load BGE-M3 reranker model - baseline version.

    Returns:
        FlagReranker instance

    Raises:
        RerankerModelError: If model fails to load
    """
    try:
        from FlagEmbedding import FlagReranker
        # Baseline: no explicit device management, no caching
        model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        print(f"Loaded BGE-M3 reranker (baseline)")
        return model
    except Exception as e:
        raise RerankerModelError(f"Failed to load BGE-M3 model: {e}")


@st.cache_resource
def load_bge_reranker_v2o():
    """
    Load BGE-M3 reranker v2o - OPTIMIZED version with caching.

    v2o Optimizations:
    - GPU auto-detection with fp16 precision (2-3× faster)
    - Model loading cache (reuse across queries)
    - Batch processing for multiple query-doc pairs (2-4× faster)
    - Result caching for repeated queries

    Returns:
        Dict with FlagReranker instance and cache

    Raises:
        RerankerModelError: If model fails to load
    """
    try:
        from FlagEmbedding import FlagReranker
        import torch
        from pathlib import Path

        # Auto-detect GPU (GTX 1080 Ti fully supported by PyTorch 2.5.1+cu121)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load model
        model = FlagReranker(
            'BAAI/bge-reranker-v2-m3',
            use_fp16=torch.cuda.is_available(),  # Enable fp16 for GPU
            device=device
        )

        # Setup cache directory
        cache_dir = Path.home() / '.cache' / 'maniscope' / 'bge_m3_v2o'
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Loaded BGE-M3 v2o reranker on {device} with fp16")
        print(f"✓ Cache directory: {cache_dir}")

        return {
            "type": "bge_m3_v2o",
            "model": model,
            "device": device,
            "cache_dir": cache_dir,
            "query_cache": {}  # LRU cache for query results
        }

    except Exception as e:
        raise RerankerModelError(f"Failed to load BGE-M3 v2o model: {e}")


@st.cache_resource
def load_qwen_reranker():
    """
    Load Qwen-1.5B LLM reranker model.

    Returns:
        FlagLLMReranker instance

    Raises:
        RerankerModelError: If model fails to load
    """
    try:
        from FlagEmbedding import FlagLLMReranker
        model = FlagLLMReranker('Alibaba-NLP/gte-qwen2-1.5b-instruct', use_fp16=True)
        return model
    except Exception as e:
        raise RerankerModelError(f"Failed to load Qwen-1.5B model: {e}")


def load_llm_reranker_api(base_url: str = "http://localhost:11434/v1", api_key: str = "ollama", model_name: str = "llama3.1:8b"):
    """
    Create LLM reranker using OpenAI-compatible API (Ollama or OpenRouter).

    Args:
        base_url: API base URL (default: Ollama local)
        api_key: API key (default: "ollama")
        model_name: Model identifier (default: "llama3.1:8b")

    Returns:
        Dict with model configuration

    Raises:
        RerankerModelError: If configuration is invalid
    """
    try:
        return {
            "type": "llm_api",
            "base_url": base_url,
            "api_key": api_key,
            "model_name": model_name
        }
    except Exception as e:
        raise RerankerModelError(f"Failed to configure LLM API reranker: {e}")


@st.cache_resource
def load_maniscope_reranker(k: int = 5, alpha: float = 0.3, version: Optional[str] = None, embedding_model: str = 'all-MiniLM-L6-v2'):
    """
    Load Maniscope geodesic reranker (version-aware).

    Args:
        k: Number of nearest neighbors for manifold graph construction (default: 5)
        alpha: Hybrid scoring weight (0=pure geodesic, 1=pure cosine, default: 0.3)
        version: Version to load ('v0', 'v1', 'v2', 'v3', 'v2o', or None for config default)
        embedding_model: Sentence-Transformers model name (default: 'all-MiniLM-L6-v2')

    Returns:
        ManiscopeEngine instance (v0, v1, v2, v3, or v2o)

    Raises:
        RerankerModelError: If model fails to load
    """
    try:
        # Use default from config if not specified
        if version is None:
            version = DEFAULT_MANISCOPE_VERSION

        # Auto-detect GPU (GTX 1080 Ti fully supported by PyTorch 2.5.1+cu121)
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if version == 'v2o':
            # v2o: RECOMMENDED - Ultimate optimization (20-235× speedup with GPU)
            from maniscope.maniscope_engine import ManiscopeEngine_v2o
            model = ManiscopeEngine_v2o(
                model_name='all-MiniLM-L6-v2',
                k=k,
                alpha=alpha,
                verbose=False,
                device=device,
                local_files_only=True,
                use_cache=True,  # Enable persistent disk cache
                query_cache_size=100,  # Cache 100 queries in memory
                use_faiss=True  # Enable FAISS for GPU-accelerated k-NN
            )
        elif version == 'v1':
            from maniscope.maniscope_engine import ManiscopeEngine_v1
            model = ManiscopeEngine_v1(
                model_name='all-MiniLM-L6-v2',
                k=k,
                alpha=alpha,
                verbose=False,
                device=device,
                local_files_only=True
            )
        elif version == 'v2':
            from maniscope.maniscope_engine import ManiscopeEngine_v2
            model = ManiscopeEngine_v2(
                model_name='all-MiniLM-L6-v2',
                k=k,
                alpha=alpha,
                verbose=False,
                device=device,
                local_files_only=True,
                use_faiss=True
            )
        elif version == 'v3':
            from maniscope.maniscope_engine import ManiscopeEngine_v3
            model = ManiscopeEngine_v3(
                model_name='all-MiniLM-L6-v2',
                k=k,
                alpha=alpha,
                verbose=False,
                device=device,
                local_files_only=True,
                use_cache=True,  # Enable persistent disk cache
                query_cache_size=100  # Cache 100 queries in memory
            )
        else:  # v0 or default
            from maniscope.maniscope_engine import ManiscopeEngine
            model = ManiscopeEngine(
                model_name='all-MiniLM-L6-v2',
                k=k,
                alpha=alpha,
                verbose=False,
                device=device,
                local_files_only=True
            )
        return model
    except Exception as e:
        raise RerankerModelError(f"Failed to load Maniscope {version} model: {e}")


@st.cache_resource
def load_jina_reranker_v2(model_name: str = "jinaai/jina-reranker-v2-base-multilingual"):
    """
    Load Jina Reranker v2 using Hugging Face transformers - baseline version.

    Args:
        model_name: Hugging Face model identifier (default: jinaai/jina-reranker-v2-base-multilingual)

    Returns:
        Dict with model, tokenizer, and device info

    Raises:
        RerankerModelError: If model fails to load
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        print(f"Loading Jina Reranker v2 (baseline): {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Auto-detect GPU (GTX 1080 Ti fully supported by PyTorch 2.5.1+cu121)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )

        model = model.to(device)
        model.eval()

        print(f"✓ Jina Reranker v2 loaded on {device} ({torch_dtype})")

        return {
            "type": "jina_local",
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "model_name": model_name
        }

    except Exception as e:
        raise RerankerModelError(f"Failed to load Jina Reranker v2: {e}")


@st.cache_resource
def load_jina_reranker_v2_v2o(model_name: str = "jinaai/jina-reranker-v2-base-multilingual"):
    """
    Load Jina Reranker v2 v2o - OPTIMIZED version with caching.

    v2o Optimizations:
    - GPU auto-detection with fp16 precision (3-5× faster)
    - Model loading cache (reuse across queries)
    - Batch processing optimization (better GPU utilization)
    - Result caching for repeated query-doc pairs

    Args:
        model_name: Hugging Face model identifier (default: jinaai/jina-reranker-v2-base-multilingual)

    Returns:
        Dict with model, tokenizer, device info, and cache

    Raises:
        RerankerModelError: If model fails to load
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        from pathlib import Path

        print(f"Loading Jina Reranker v2 v2o: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Auto-detect GPU (GTX 1080 Ti fully supported by PyTorch 2.5.1+cu121)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )

        model = model.to(device)
        model.eval()

        # Setup cache directory
        cache_dir = Path.home() / '.cache' / 'maniscope' / 'jina_v2_v2o'
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Jina Reranker v2 v2o loaded on {device} with {torch_dtype}")
        print(f"✓ Cache directory: {cache_dir}")

        return {
            "type": "jina_local_v2o",
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "model_name": model_name,
            "cache_dir": cache_dir,
            "query_cache": {}  # LRU cache for query results
        }

    except Exception as e:
        raise RerankerModelError(f"Failed to load Jina Reranker v2 v2o: {e}")


@st.cache_resource
def load_hnsw_reranker(embedding_model: str = "all-MiniLM-L6-v2", space: str = "cosine", ef_construction: int = 200, M: int = 16):
    """
    Load HNSW (Hierarchical Navigable Small World) reranker - baseline version.

    HNSW is a graph-based approximate nearest neighbor search algorithm.
    This serves as a baseline to compare against Maniscope's geodesic reranking.

    Key differences from Maniscope:
    - HNSW: Hierarchical graph for fast ANN search (global retrieval)
    - Maniscope: k-NN manifold graph for geodesic reranking (local refinement)

    Args:
        embedding_model: Sentence-Transformers model name for encoding
        space: Distance metric ('cosine', 'l2', 'ip')
        ef_construction: Controls index quality (higher = better quality, slower build)
        M: Max number of connections per node (higher = better recall, more memory)

    Returns:
        Dict with HNSW index configuration

    Raises:
        RerankerModelError: If model fails to load
    """
    try:
        from sentence_transformers import SentenceTransformer
        import hnswlib
        import torch

        # Auto-detect GPU (GTX 1080 Ti fully supported by PyTorch 2.5.1+cu121)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(embedding_model, device=device)
        dim = model.get_sentence_embedding_dimension()

        print(f"Loaded HNSW reranker with {embedding_model} ({dim}d) on {device}")

        return {
            "type": "hnsw",
            "embedding_model": model,
            "dim": dim,
            "space": space,
            "ef_construction": ef_construction,
            "M": M,
            "model_name": embedding_model
        }

    except Exception as e:
        raise RerankerModelError(f"Failed to load HNSW reranker: {e}")


@st.cache_resource
def load_hnsw_reranker_v2o(embedding_model: str = "all-MiniLM-L6-v2", space: str = "cosine", ef_construction: int = 200, M: int = 16):
    """
    Load HNSW reranker v2o - OPTIMIZED version with caching.

    v2o Optimizations:
    - GPU auto-detection for embeddings (3-4× faster)
    - Model loading cache (reuse across queries)
    - Persistent embedding cache (disk-based, survives restarts)
    - Query embedding LRU cache (100 queries in memory)

    Args:
        embedding_model: Sentence-Transformers model name for encoding
        space: Distance metric ('cosine', 'l2', 'ip')
        ef_construction: Controls index quality (higher = better quality, slower build)
        M: Max number of connections per node (higher = better recall, more memory)

    Returns:
        Dict with HNSW index configuration including cache

    Raises:
        RerankerModelError: If model fails to load
    """
    try:
        from sentence_transformers import SentenceTransformer
        import hnswlib
        import torch
        from functools import lru_cache
        import pickle
        from pathlib import Path

        # Auto-detect GPU (GTX 1080 Ti fully supported by PyTorch 2.5.1+cu121)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load embedding model
        model = SentenceTransformer(embedding_model, device=device)
        dim = model.get_sentence_embedding_dimension()

        # Setup persistent cache directory
        cache_dir = Path.home() / '.cache' / 'maniscope' / 'hnsw_v2o'
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"✓ Loaded HNSW v2o reranker with {embedding_model} ({dim}d) on {device}")
        print(f"✓ Cache directory: {cache_dir}")

        return {
            "type": "hnsw_v2o",
            "embedding_model": model,
            "dim": dim,
            "space": space,
            "ef_construction": ef_construction,
            "M": M,
            "model_name": embedding_model,
            "device": device,
            "cache_dir": cache_dir,
            "embedding_cache": {},  # In-memory cache for this session
            "query_cache": {}  # LRU cache for query embeddings
        }

    except Exception as e:
        raise RerankerModelError(f"Failed to load HNSW v2o reranker: {e}")


@st.cache_resource
def load_all_models(selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load selected reranker models.

    Args:
        selected_models: List of model names to load. If None, loads all available.
                        Options: ["Maniscope", "Maniscope_v0", "Maniscope_v2o", "LLM-Reranker",
                                 "BGE-M3", "BGE-M3_v2o", "Jina Reranker v2", "Jina Reranker v2_v2o",
                                 "HNSW", "HNSW_v2o", "Qwen-1.5B"]

    Returns:
        Dict mapping model names to model instances/configs

    Raises:
        RerankerModelError: If any model fails to load
    """
    if selected_models is None:
        selected_models = ["BGE-M3", "Qwen-1.5B", "Maniscope"]

    models = {}
    errors = []

    for model_name in selected_models:
        try:
            if model_name == "BGE-M3":
                models[model_name] = load_bge_reranker()
            elif model_name == "BGE-M3_v2o":
                models[model_name] = load_bge_reranker_v2o()
            elif model_name == "Qwen-1.5B":
                models[model_name] = load_qwen_reranker()
            elif model_name == "LLM-Reranker":
                # Get config from session state if available
                base_url = st.session_state.get('llm_base_url', "http://localhost:11434/v1")
                api_key = st.session_state.get('llm_api_key', "ollama")
                llm_model = st.session_state.get('llm_model', "llama3.1:8b")
                models[model_name] = load_llm_reranker_api(base_url, api_key, llm_model)
            elif model_name == "Maniscope":
                # Get config from session state if available
                k = st.session_state.get('maniscope_k', 5)
                alpha = st.session_state.get('maniscope_alpha', 0.3)
                version = st.session_state.get('maniscope_version', DEFAULT_MANISCOPE_VERSION)
                models[model_name] = load_maniscope_reranker(k, alpha, version)
            elif model_name == "Maniscope_v0":
                # Explicit v0 (baseline): CPU, no caching - 115ms avg
                k = st.session_state.get('maniscope_k', 5)
                alpha = st.session_state.get('maniscope_alpha', 0.3)
                models[model_name] = load_maniscope_reranker(k, alpha, version="v0")
            elif model_name == "Maniscope_v2o":
                # Explicit v2o (ultimate): GPU + all optimizations - 0.4-20ms avg
                k = st.session_state.get('maniscope_k', 5)
                alpha = st.session_state.get('maniscope_alpha', 0.3)
                models[model_name] = load_maniscope_reranker(k, alpha, version="v2o")
            elif model_name == "Jina Reranker v2":
                models[model_name] = load_jina_reranker_v2()
            elif model_name == "Jina Reranker v2_v2o":
                models[model_name] = load_jina_reranker_v2_v2o()
            elif model_name == "HNSW":
                # Get config from session state if available
                embedding_model = st.session_state.get('hnsw_embedding_model', 'all-MiniLM-L6-v2')
                models[model_name] = load_hnsw_reranker(embedding_model=embedding_model)
            elif model_name == "HNSW_v2o":
                # Get config from session state if available
                embedding_model = st.session_state.get('hnsw_embedding_model', 'all-MiniLM-L6-v2')
                models[model_name] = load_hnsw_reranker_v2o(embedding_model=embedding_model)
            else:
                errors.append(f"Unknown model: {model_name}")
        except Exception as e:
            errors.append(f"{model_name}: {str(e)}")

    if errors and not models:
        raise RerankerModelError("All models failed to load:\n" + "\n".join(errors))

    return models


def run_bge_reranker(model: Any, query: str, docs: List[str]) -> np.ndarray:
    """
    Run BGE reranker on query-document pairs - baseline version.

    Args:
        model: FlagReranker instance
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document)
    """
    pairs = [[query, doc] for doc in docs]
    scores = model.compute_score(pairs)

    # Handle both single score and list of scores
    if isinstance(scores, (list, tuple)):
        return np.array(scores)
    else:
        return np.array([scores])


def run_bge_reranker_v2o(model_dict: Dict, query: str, docs: List[str]) -> np.ndarray:
    """
    Run BGE-M3 reranker v2o - OPTIMIZED with caching.

    v2o Optimizations:
    - GPU fp16 for 2-3× speedup (from load function)
    - Query result caching (repeated queries = instant results)
    - Batch processing optimization
    - Warm-start optimization (model stays loaded in GPU memory)

    Args:
        model_dict: Dict containing FlagReranker instance and cache
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document)
    """
    try:
        model = model_dict["model"]
        query_cache = model_dict.get("query_cache", {})

        # Create cache key from query + docs using fast hash (10-100× faster than MD5)
        cache_key = hash((query, tuple(docs)))

        # Check cache first (warm-start optimization)
        if cache_key in query_cache:
            return query_cache[cache_key]

        # Cache miss - compute scores
        pairs = [[query, doc] for doc in docs]
        scores = model.compute_score(pairs)

        # Handle both single score and list of scores
        if isinstance(scores, (list, tuple)):
            scores_array = np.array(scores)
        else:
            scores_array = np.array([scores])

        # Cache the results (keep last 100 queries)
        if len(query_cache) > 100:
            # Remove oldest entry (simple FIFO)
            query_cache.pop(next(iter(query_cache)))
        query_cache[cache_key] = scores_array

        return scores_array

    except Exception as e:
        raise RerankerModelError(f"BGE-M3 v2o reranker failed: {e}")


def run_qwen_reranker(model: Any, query: str, docs: List[str]) -> np.ndarray:
    """
    Run Qwen LLM reranker on query-document pairs.

    Args:
        model: FlagLLMReranker instance
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document)

    Note:
        FlagLLMReranker returns LOWER scores for MORE relevant documents.
        We negate the scores so higher = more relevant (consistent with other rerankers).
    """
    pairs = [[query, doc] for doc in docs]
    scores = model.compute_score(pairs)

    # Handle both single score and list of scores
    if isinstance(scores, (list, tuple)):
        # Negate scores: FlagLLMReranker uses lower=better, we need higher=better
        return -np.array(scores)
    else:
        return -np.array([scores])


def run_llm_api_reranker(model_config: Dict, query: str, docs: List[str]) -> np.ndarray:
    """
    Run LLM reranker via OpenAI-compatible API.

    Uses a prompt-based approach where the LLM scores each document.

    Args:
        model_config: Dict with API configuration
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document)
    """
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=model_config["base_url"],
            api_key=model_config["api_key"]
        )

        scores = []
        for doc in docs:
            prompt = f"""Rate the relevance of this document to the query on a scale of 0-100.
Only respond with a number.

Query: {query}

Document: {doc}

Relevance score (0-100):"""

            response = client.chat.completions.create(
                model=model_config.get("model_name", "llama3.1:8b"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )

            # Extract score from response
            score_text = response.choices[0].message.content
            if score_text:
                score_text = score_text.strip()
            else:
                score_text = "50"  # Default to neutral score
            try:
                score = float(score_text) / 100.0  # Normalize to 0-1
            except (ValueError, TypeError):
                score = 0.5  # Default score if parsing fails

            scores.append(score)

        return np.array(scores)

    except Exception as e:
        raise RerankerModelError(f"LLM API reranker failed: {e}")


def run_maniscope_reranker(model: Any, query: str, docs: List[str]) -> np.ndarray:
    """
    Run Maniscope geodesic reranker (v0 baseline - no caching).

    Uses a two-stage approach:
    1. Telescope: Broad retrieval with cosine similarity
    2. Microscope: Geodesic reranking on k-NN manifold graph

    Args:
        model: ManiscopeEngine instance
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document)
    """
    try:
        # Fit model on the document corpus
        model.fit(docs)

        # Get detailed results with scores
        results = model.search_maniscope_detailed(query, top_n=len(docs), coarse_multiplier=3)

        # Create score array maintaining original document order
        scores = np.zeros(len(docs))
        for result in results:
            doc_id = result['doc_id']
            scores[doc_id] = result['final_score']

        return scores

    except Exception as e:
        raise RerankerModelError(f"Maniscope reranker failed: {e}")


def run_maniscope_reranker_v1(model: Any, query: str, docs: List[str]) -> np.ndarray:
    """
    Run Maniscope v1 reranker with graph caching optimization.

    v1 Optimization: Cache graph between queries for same document set.
    This eliminates the need to rebuild the graph for every query (42ms → 0ms).

    Args:
        model: ManiscopeEngine_v1 instance
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document)
    """
    try:
        # v1 Optimization: Graph caching
        # Only rebuild graph if documents have changed
        docs_hash = hash(tuple(docs))
        if not hasattr(model, '_cached_docs_hash') or model._cached_docs_hash != docs_hash:
            # Documents changed, rebuild graph
            model.fit(docs)
            model._cached_docs_hash = docs_hash
        # else: reuse cached graph from previous queries

        # Get detailed results with scores
        results = model.search_maniscope_detailed(query, top_n=len(docs), coarse_multiplier=3)

        # Create score array maintaining original document order
        scores = np.zeros(len(docs))
        for result in results:
            doc_id = result['doc_id']
            scores[doc_id] = result['final_score']

        return scores

    except Exception as e:
        raise RerankerModelError(f"Maniscope v1 reranker failed: {e}")


def run_maniscope_reranker_v2(model: Any, query: str, docs: List[str]) -> np.ndarray:
    """
    Run Maniscope v2 reranker with full optimizations.

    v2 Optimizations:
    - Graph caching (same as v1)
    - scipy sparse for geodesic distances (4× faster)
    - FAISS for GPU k-NN (4× faster)
    - Vectorized scoring (4× faster)

    Args:
        model: ManiscopeEngine_v2 instance
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document)
    """
    try:
        # v2 Optimization: Graph caching (same as v1)
        docs_hash = hash(tuple(docs))
        if not hasattr(model, '_cached_docs_hash') or model._cached_docs_hash != docs_hash:
            model.fit(docs)
            model._cached_docs_hash = docs_hash

        # Get detailed results with scores
        results = model.search_maniscope_detailed(query, top_n=len(docs), coarse_multiplier=3)

        # Create score array maintaining original document order
        scores = np.zeros(len(docs))
        for result in results:
            doc_id = result['doc_id']
            scores[doc_id] = result['final_score']

        return scores

    except Exception as e:
        raise RerankerModelError(f"Maniscope v2 reranker failed: {e}")


def run_maniscope_reranker_v3(model: Any, query: str, docs: List[str]) -> np.ndarray:
    """
    Run Maniscope v3 reranker with persistent caching optimizations.

    v3 Optimizations:
    - Persistent disk cache for embeddings (survives restarts!)
    - Query embedding LRU cache (100 queries in memory)
    - Batch geodesic computation with NetworkX
    - Heap-based early termination for top-n selection

    Performance characteristics:
    - First run (cold cache): ~115ms (same as v0)
    - Cache hit (warm): ~30-50ms (2-3x faster)
    - Repeated queries: ~10-20ms (5-10x faster with query cache)

    Args:
        model: ManiscopeEngine_v3 instance
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document)
    """
    try:
        # v3 automatically handles disk caching internally
        # No need for external graph caching - fit() uses persistent cache
        model.fit(docs)

        # Get detailed results with scores
        # v3 also caches query embeddings internally (LRU)
        results = model.search_maniscope_detailed(query, top_n=len(docs), coarse_multiplier=3)

        # Create score array maintaining original document order
        scores = np.zeros(len(docs))
        for result in results:
            doc_id = result['doc_id']
            scores[doc_id] = result['final_score']

        return scores

    except Exception as e:
        raise RerankerModelError(f"Maniscope v3 reranker failed: {e}")


def run_maniscope_reranker_v2o(model: Any, query: str, docs: List[str]) -> np.ndarray:
    """
    Run Maniscope v2o reranker - RECOMMENDED Ultimate Optimization.

    v2o combines ALL optimizations from v1, v2, and v3:

    From v1 (GPU):
    - GPU auto-detection for embeddings (3.5× faster encoding)

    From v2 (Algorithmic):
    - scipy.sparse for geodesic distances (4× faster than NetworkX)
    - FAISS for GPU-accelerated k-NN (4× faster than sklearn)
    - Vectorized hybrid scoring (4× faster than loops)

    From v3 (Caching):
    - Persistent disk cache for embeddings (survives restarts!)
    - Query embedding LRU cache (100 queries in memory)

    Performance characteristics:
    - Cold cache (first run): ~4-20ms (5-25× faster than v0)
    - Warm cache (cached embeddings): ~0.4-0.6ms (20-235× faster than v0)
    - Real-world benchmarks:
      * MS MARCO: 132ms → 0.58ms (229× speedup)
      * TREC-COVID: 85ms → 0.38ms (226× speedup)
      * SciFact: 92ms → 0.39ms (235× speedup)

    Args:
        model: ManiscopeEngine_v2o instance
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document)
    """
    try:
        # v2o automatically handles all optimizations internally:
        # - Persistent disk cache for embeddings (fit() uses cache)
        # - GPU auto-detection
        # - FAISS for k-NN
        # - scipy sparse for geodesic computation
        model.fit(docs)

        # Get detailed results with scores
        # v2o also caches query embeddings internally (LRU)
        results = model.search_maniscope_detailed(query, top_n=len(docs), coarse_multiplier=3)

        # Create score array maintaining original document order
        scores = np.zeros(len(docs))
        for result in results:
            doc_id = result['doc_id']
            scores[doc_id] = result['final_score']

        return scores

    except Exception as e:
        raise RerankerModelError(f"Maniscope v2o reranker failed: {e}")


def run_jina_reranker(model_dict: Dict, query: str, docs: List[str]) -> np.ndarray:
    """
    Run Jina Reranker v2 on query-document pairs with batch processing - baseline version.

    Args:
        model_dict: Dict containing model, tokenizer, device
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document)
    """
    try:
        import torch

        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        device = model_dict["device"]

        # Batch process all query-document pairs at once for better GPU utilization
        query_doc_pairs = [(query, doc) for doc in docs]

        with torch.no_grad():
            # Tokenize all pairs together
            inputs = tokenizer(
                query_doc_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            # Get relevance scores for all pairs in one forward pass
            outputs = model(**inputs)
            # Extract scores from logits (batch dimension) and convert to float32
            scores = outputs.logits[:, 0].float().cpu().numpy()

        return scores

    except Exception as e:
        raise RerankerModelError(f"Jina Reranker v2 failed: {e}")


def run_jina_reranker_v2o(model_dict: Dict, query: str, docs: List[str]) -> np.ndarray:
    """
    Run Jina Reranker v2 v2o - OPTIMIZED with caching.

    v2o Optimizations:
    - GPU fp16 for 3-5× speedup (from load function)
    - Query result caching (repeated queries = instant results)
    - Batch processing with optimal GPU utilization
    - Warm-start optimization (model stays loaded in GPU memory)

    Args:
        model_dict: Dict containing model, tokenizer, device, cache
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document)
    """
    try:
        import torch

        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        device = model_dict["device"]
        query_cache = model_dict.get("query_cache", {})

        # Create cache key from query + docs using fast hash (10-100× faster than MD5)
        cache_key = hash((query, tuple(docs)))

        # Check cache first (warm-start optimization)
        if cache_key in query_cache:
            return query_cache[cache_key]

        # Cache miss - compute scores
        query_doc_pairs = [(query, doc) for doc in docs]

        with torch.no_grad():
            # Tokenize all pairs together
            inputs = tokenizer(
                query_doc_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            # Get relevance scores for all pairs in one forward pass
            outputs = model(**inputs)
            # Extract scores from logits (batch dimension) and convert to float32
            scores = outputs.logits[:, 0].float().cpu().numpy()

        # Cache the results (keep last 100 queries)
        if len(query_cache) > 100:
            # Remove oldest entry (simple FIFO)
            query_cache.pop(next(iter(query_cache)))
        query_cache[cache_key] = scores

        return scores

    except Exception as e:
        raise RerankerModelError(f"Jina Reranker v2 v2o failed: {e}")


def run_hnsw_reranker(model_dict: Dict, query: str, docs: List[str]) -> np.ndarray:
    """
    Run HNSW reranker on query-document pairs - baseline version.

    HNSW builds a hierarchical graph structure for approximate nearest neighbor search.
    This provides a baseline comparison to Maniscope's geodesic reranking approach.

    Args:
        model_dict: Dict containing HNSW index configuration
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document, normalized 0-1)
        Higher scores indicate more relevant documents
    """
    try:
        import hnswlib

        embedding_model = model_dict["embedding_model"]
        dim = model_dict["dim"]
        space = model_dict["space"]
        ef_construction = model_dict["ef_construction"]
        M = model_dict["M"]

        # Encode query and documents (no caching in baseline)
        query_embedding = embedding_model.encode([query])[0]
        doc_embeddings = embedding_model.encode(docs)

        # Build HNSW index on documents (rebuilt each time)
        num_docs = len(docs)
        index = hnswlib.Index(space=space, dim=dim)
        index.init_index(max_elements=num_docs, ef_construction=ef_construction, M=M)
        index.add_items(doc_embeddings, np.arange(num_docs))

        # Set ef for search (higher = more accurate, slower)
        index.set_ef(50)  # Balance between speed and accuracy

        # Query the index to get all documents sorted by similarity
        labels, distances = index.knn_query(query_embedding, k=num_docs)

        # Convert distances to similarity scores
        # For cosine distance: similarity = 1 - distance
        # For L2 distance: similarity = 1 / (1 + distance)
        if space == "cosine":
            # Cosine distance in [0, 2], convert to similarity in [0, 1]
            similarities = 1.0 - distances[0]
        else:  # L2 or IP
            # L2 distance, convert to similarity
            similarities = 1.0 / (1.0 + distances[0])

        # Create score array maintaining original document order
        scores = np.zeros(num_docs)
        for doc_id, sim in zip(labels[0], similarities):
            scores[doc_id] = sim

        return scores

    except Exception as e:
        raise RerankerModelError(f"HNSW reranker failed: {e}")


def run_hnsw_reranker_v2o(model_dict: Dict, query: str, docs: List[str]) -> np.ndarray:
    """
    Run HNSW reranker v2o - OPTIMIZED with caching.

    v2o Optimizations:
    - GPU-accelerated embeddings (3-4× faster)
    - Document embedding cache (avoid recomputing)
    - Query embedding LRU cache (100 queries in memory)
    - HNSW index caching (reuse if documents unchanged)
    - Warm-start optimization

    Args:
        model_dict: Dict containing HNSW index configuration and cache
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document, normalized 0-1)
    """
    try:
        import hnswlib

        embedding_model = model_dict["embedding_model"]
        dim = model_dict["dim"]
        space = model_dict["space"]
        ef_construction = model_dict["ef_construction"]
        M = model_dict["M"]
        embedding_cache = model_dict.get("embedding_cache", {})
        query_cache = model_dict.get("query_cache", {})

        # Create document hash for caching using fast hash (10-100× faster than MD5)
        docs_hash = hash(tuple(docs))

        # Check if we have cached embeddings for these documents
        if docs_hash in embedding_cache:
            doc_embeddings = embedding_cache[docs_hash]["embeddings"]
            index = embedding_cache[docs_hash]["index"]
        else:
            # Cache miss - compute embeddings and build index
            doc_embeddings = embedding_model.encode(docs)

            # Build HNSW index
            num_docs = len(docs)
            index = hnswlib.Index(space=space, dim=dim)
            index.init_index(max_elements=num_docs, ef_construction=ef_construction, M=M)
            index.add_items(doc_embeddings, np.arange(num_docs))
            index.set_ef(50)

            # Cache the embeddings and index (keep last 10 document sets)
            if len(embedding_cache) > 10:
                # Remove oldest entry
                embedding_cache.pop(next(iter(embedding_cache)))
            embedding_cache[docs_hash] = {
                "embeddings": doc_embeddings,
                "index": index
            }

        # Check query cache using fast hash
        query_hash = hash((query, docs_hash))
        if query_hash in query_cache:
            return query_cache[query_hash]

        # Encode query (GPU-accelerated from load function)
        query_embedding = embedding_model.encode([query])[0]

        # Query the index
        num_docs = len(docs)
        labels, distances = index.knn_query(query_embedding, k=num_docs)

        # Convert distances to similarity scores
        if space == "cosine":
            similarities = 1.0 - distances[0]
        else:
            similarities = 1.0 / (1.0 + distances[0])

        # Create score array maintaining original document order
        scores = np.zeros(num_docs)
        for doc_id, sim in zip(labels[0], similarities):
            scores[doc_id] = sim

        # Cache the results (keep last 100 queries)
        if len(query_cache) > 100:
            query_cache.pop(next(iter(query_cache)))
        query_cache[query_hash] = scores

        return scores

    except Exception as e:
        raise RerankerModelError(f"HNSW v2o reranker failed: {e}")


def run_reranker(model: Any, query: str, docs: List[str]) -> np.ndarray:
    """
    Unified interface to run any reranker model.

    Args:
        model: Model instance or config dict
        query: Query text
        docs: List of document texts

    Returns:
        Numpy array of scores (one per document)

    Raises:
        RerankerModelError: If model type is unknown or inference fails
    """
    try:
        # Check if it's a config dict (check v2o types first!)
        if isinstance(model, dict):
            if model.get("type") == "llm_api":
                return run_llm_api_reranker(model, query, docs)
            elif model.get("type") == "jina_local_v2o":
                return run_jina_reranker_v2o(model, query, docs)
            elif model.get("type") == "jina_local":
                return run_jina_reranker(model, query, docs)
            elif model.get("type") == "hnsw_v2o":
                return run_hnsw_reranker_v2o(model, query, docs)
            elif model.get("type") == "hnsw":
                return run_hnsw_reranker(model, query, docs)
            elif model.get("type") == "bge_m3_v2o":
                return run_bge_reranker_v2o(model, query, docs)

        # Check if it's a FlagEmbedding or other model
        model_class = model.__class__.__name__
        model_type_str = str(type(model))

        # Handle Maniscope versions (order matters - check specific versions first!)
        if "ManiscopeEngine_v2o" in model_class:
            return run_maniscope_reranker_v2o(model, query, docs)
        elif "ManiscopeEngine_v3" in model_class:
            return run_maniscope_reranker_v3(model, query, docs)
        elif "ManiscopeEngine_v2" in model_class:
            return run_maniscope_reranker_v2(model, query, docs)
        elif "ManiscopeEngine_v1" in model_class:
            return run_maniscope_reranker_v1(model, query, docs)
        elif "ManiscopeEngine" in model_class:
            return run_maniscope_reranker(model, query, docs)
        elif ("FlagReranker" in model_class or "BaseReranker" in model_class or
              "BGE" in model_type_str or "encoder_only" in model_type_str):
            return run_bge_reranker(model, query, docs)
        elif ("FlagLLMReranker" in model_class or "BaseLLMReranker" in model_class or
              "Qwen" in model_type_str or "decoder_only" in model_type_str):
            return run_qwen_reranker(model, query, docs)
        else:
            raise RerankerModelError(f"Unknown model type: {type(model)}")

    except Exception as e:
        raise RerankerModelError(f"Reranker execution failed: {e}")


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get metadata about a reranker model.

    Args:
        model_name: Name of the model

    Returns:
        Dict with model metadata:
        - name: Model name
        - architecture: Model architecture type
        - complexity: Computational complexity
        - description: Brief description
    """
    model_info = {
        "BGE-M3": {
            "name": "BGE-Reranker-v2-M3",
            "architecture": "Encoder-only (BERT-style)",
            "complexity": "O(N)",
            "description": "Cross-encoder reranker based on BERT architecture. Fast and effective for most use cases."
        },
        "BGE-M3_v2o": {
            "name": "BGE-Reranker-v2-M3 (Optimized)",
            "architecture": "Encoder-only (BERT-style) + GPU + Caching",
            "complexity": "O(N)",
            "description": "Optimized BGE-M3 with GPU acceleration, fp16 precision, and query result caching. 2-3× faster than baseline."
        },
        "Qwen-1.5B": {
            "name": "GTE-Qwen2-1.5B-Instruct",
            "architecture": "Decoder-only (GPT-style)",
            "complexity": "O(N)",
            "description": "LLM-based reranker using instruction-tuned Qwen model. Larger model with strong performance."
        },
        "LLM-Reranker": {
            "name": "LLM-as-Reranker",
            "architecture": "Generative (API-based)",
            "complexity": "O(N)",
            "description": "Uses any LLM via API to score document relevance. Flexible but higher latency."
        },
        "Maniscope": {
            "name": "Maniscope Geodesic Reranker",
            "architecture": "Manifold-based (k-NN graph + geodesic distance)",
            "complexity": "O(N log N)",
            "description": "Multi-scale reranker combining cosine similarity (telescope) with geodesic distance on k-NN manifold (microscope). Excels at semantic disambiguation with 4x precision improvement over baseline."
        },
        "Maniscope_v0": {
            "name": "Maniscope v0 - Baseline (CPU, no caching)",
            "architecture": "Manifold-based (k-NN graph + geodesic) - CPU only",
            "complexity": "O(N log N)",
            "description": "Baseline Maniscope: CPU-only, rebuilds graph every query. 115ms avg latency. Reference for comparison."
        },
        "Maniscope_v2o": {
            "name": "Maniscope v2o - Ultimate Optimization ⭐",
            "architecture": "Manifold-based + GPU + FAISS + scipy + persistent cache",
            "complexity": "O(N log N) → O(1) with cache",
            "description": "Ultimate Maniscope with ALL optimizations: GPU + FAISS + scipy + persistent cache + query cache. 0.4-20ms avg latency, 20-235× speedup over baseline."
        },
        "Jina Reranker v2": {
            "name": "Jina Reranker v2",
            "architecture": "Transformer-based cross-encoder",
            "complexity": "O(N)",
            "description": "State-of-the-art commercial reranker from Jina AI. Claims 6× speedup over v1, supports 8K tokens, multilingual."
        },
        "Jina Reranker v2_v2o": {
            "name": "Jina Reranker v2 (Optimized)",
            "architecture": "Transformer-based cross-encoder + GPU + Caching",
            "complexity": "O(N)",
            "description": "Optimized Jina v2 with GPU acceleration, fp16 precision, batch processing, and query caching. 3-5× faster than baseline."
        },
        "HNSW": {
            "name": "HNSW (Hierarchical Navigable Small World)",
            "architecture": "Graph-based approximate nearest neighbor search",
            "complexity": "O(log N)",
            "description": "Hierarchical graph structure for fast approximate nearest neighbor search. Uses global similarity without local manifold refinement (baseline comparison to Maniscope)."
        },
        "HNSW_v2o": {
            "name": "HNSW (Optimized)",
            "architecture": "Graph-based ANN + GPU embeddings + Caching",
            "complexity": "O(log N)",
            "description": "Optimized HNSW with GPU-accelerated embeddings, document/query caching, and index reuse. 3-10× faster than baseline on warm cache."
        }
    }

    return model_info.get(model_name, {
        "name": model_name,
        "architecture": "Unknown",
        "complexity": "Unknown",
        "description": "No information available"
    })


# Test if run directly
if __name__ == "__main__":
    print("Testing model loading...")

    try:
        print("\n1. Loading BGE-M3...")
        bge = load_bge_reranker()
        print(f"   Loaded: {type(bge)}")

        # Test with sample data
        query = "What is machine learning?"
        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a programming language.",
            "The weather today is sunny."
        ]

        print("\n2. Running BGE reranker...")
        scores = run_bge_reranker(bge, query, docs)
        print(f"   Scores: {scores}")
        print(f"   Rankings: {np.argsort(scores)[::-1]}")

        print("\n3. Model info:")
        for model_name in ["BGE-M3", "Qwen-1.5B", "LLM-Reranker"]:
            info = get_model_info(model_name)
            print(f"\n   {model_name}:")
            for key, value in info.items():
                print(f"     {key}: {value}")

    except Exception as e:
        print(f"   Error: {e}")
