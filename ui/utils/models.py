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
    Load BGE-M3 reranker model.

    Returns:
        FlagReranker instance

    Raises:
        RerankerModelError: If model fails to load
    """
    try:
        from FlagEmbedding import FlagReranker
        model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        return model
    except Exception as e:
        raise RerankerModelError(f"Failed to load BGE-M3 model: {e}")


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
def load_maniscope_reranker(k: int = 5, alpha: float = 0.3, version: Optional[str] = None):
    """
    Load Maniscope geodesic reranker (version-aware).

    Args:
        k: Number of nearest neighbors for manifold graph construction (default: 5)
        alpha: Hybrid scoring weight (0=pure geodesic, 1=pure cosine, default: 0.3)
        version: Version to load ('v0', 'v1', 'v2', 'v3', 'v2o', or None for config default)

    Returns:
        ManiscopeEngine instance (v0, v1, v2, v3, or v2o)

    Raises:
        RerankerModelError: If model fails to load
    """
    try:
        # Use default from config if not specified
        if version is None:
            version = DEFAULT_MANISCOPE_VERSION

        if version == 'v2o':
            # v2o: RECOMMENDED - Ultimate optimization (20-235× speedup)
            from maniscope import ManiscopeEngine_v2o
            model = ManiscopeEngine_v2o(
                model_name='all-MiniLM-L6-v2',
                k=k,
                alpha=alpha,
                verbose=False,
                device=None,  # Auto-detect GPU
                local_files_only=True,
                use_cache=True,  # Enable persistent disk cache
                query_cache_size=100,  # Cache 100 queries in memory
                use_faiss=True  # Enable FAISS for GPU-accelerated k-NN
            )
        elif version == 'v1':
            from maniscope import ManiscopeEngine_v1
            model = ManiscopeEngine_v1(
                model_name='all-MiniLM-L6-v2',
                k=k,
                alpha=alpha,
                verbose=False,
                device=None,  # Auto-detect GPU
                local_files_only=True
            )
        elif version == 'v2':
            from maniscope import ManiscopeEngine_v2
            model = ManiscopeEngine_v2(
                model_name='all-MiniLM-L6-v2',
                k=k,
                alpha=alpha,
                verbose=False,
                device=None,  # Auto-detect GPU
                local_files_only=True,
                use_faiss=True
            )
        elif version == 'v3':
            from maniscope import ManiscopeEngine_v3
            model = ManiscopeEngine_v3(
                model_name='all-MiniLM-L6-v2',
                k=k,
                alpha=alpha,
                verbose=False,
                device='cpu',  # v3 is CPU-friendly (can change to 'cuda' if GPU available)
                local_files_only=True,
                use_cache=True,  # Enable persistent disk cache
                query_cache_size=100  # Cache 100 queries in memory
            )
        else:  # v0 or default
            from maniscope import ManiscopeEngine
            model = ManiscopeEngine(
                model_name='all-MiniLM-L6-v2',
                k=k,
                alpha=alpha,
                verbose=False,
                device='cpu',
                local_files_only=True
            )
        return model
    except Exception as e:
        raise RerankerModelError(f"Failed to load Maniscope {version} model: {e}")


@st.cache_resource
def load_jina_reranker_v2(model_name: str = "jinaai/jina-reranker-v2-base-multilingual"):
    """
    Load Jina Reranker v2 using Hugging Face transformers.

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

        print(f"Loading Jina Reranker v2: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Load model with appropriate precision based on device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use float16 on GPU for speed, float32 on CPU for compatibility
        if device == "cuda":
            torch_dtype = torch.float16  # Faster than float32, more compatible than bfloat16
        else:
            torch_dtype = torch.float32

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
def load_all_models(selected_models: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load selected reranker models.

    Args:
        selected_models: List of model names to load. If None, loads all available.
                        Options: ["Maniscope", "LLM-Reranker", "BGE-M3", "Jina Reranker v2", "Qwen-1.5B"]

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
            elif model_name == "Jina Reranker v2":
                models[model_name] = load_jina_reranker_v2()
            else:
                errors.append(f"Unknown model: {model_name}")
        except Exception as e:
            errors.append(f"{model_name}: {str(e)}")

    if errors and not models:
        raise RerankerModelError("All models failed to load:\n" + "\n".join(errors))

    return models


def run_bge_reranker(model: Any, query: str, docs: List[str]) -> np.ndarray:
    """
    Run BGE reranker on query-document pairs.

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
    Run Jina Reranker v2 on query-document pairs with batch processing.

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
        # Check if it's a config dict
        if isinstance(model, dict):
            if model.get("type") == "llm_api":
                return run_llm_api_reranker(model, query, docs)
            elif model.get("type") == "jina_local":
                return run_jina_reranker(model, query, docs)

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
        "Jina Reranker v2": {
            "name": "Jina Reranker v2",
            "architecture": "Transformer-based cross-encoder",
            "complexity": "O(N)",
            "description": "State-of-the-art commercial reranker from Jina AI. Claims 6× speedup over v1, supports 8K tokens, multilingual."
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
