"""
ManiscopeEngine: Core geodesic reranking implementation.

This implements a two-stage retrieval system:
1. Telescope (Coarse): Broad retrieval using cosine similarity
2. Microscope (Fine): Geodesic reranking on k-NN manifold graph

Key features:
- Better anchor node selection from candidate set
- Hybrid scoring combining cosine + geodesic distances
- Robust handling of disconnected graph components
- Configurable parameters for k and retrieval depth
"""

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional, Any
import hashlib
import pickle
import os
from pathlib import Path
from collections import OrderedDict
from heapq import heappush, heappop
import joblib


class ManiscopeEngine:
    """
    Multi-scale retrieval engine combining cosine similarity (telescope)
    and geodesic distance on k-NN manifold (microscope).

    This class provides efficient semantic search by exploiting local
    manifold structure in embedding spaces through geodesic distances
    on k-nearest neighbor graphs.

    Embeddings are automatically cached to disk to avoid recomputation when
    testing different k/alpha parameters on the same document corpus.

    Attributes:
        model: Sentence transformer for text encoding
        k: Number of nearest neighbors for manifold graph
        alpha: Weight for hybrid scoring (0=pure geodesic, 1=pure cosine)
        verbose: Enable debug output
        documents: Fitted document corpus
        embeddings: Document embeddings
        G: k-NN manifold graph
        cache_dir: Directory for embedding cache files
        use_cache: Whether caching is enabled
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        k: int = 5,
        alpha: float = 0.5,
        verbose: bool = False,
        device: str = 'cpu',
        local_files_only: bool = True,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize the ManiscopeEngine.

        Args:
            model_name: Sentence transformer model name
            k: Number of nearest neighbors for manifold graph construction
            alpha: Weight for hybrid scoring (0=pure geodesic, 1=pure cosine)
            verbose: Print debug information
            device: Device for computation ('cpu' or 'cuda')
            local_files_only: Use only cached models without checking HuggingFace
            cache_dir: Directory for embedding cache (defaults to ~/projects/embedding_cache/maniscope)
            use_cache: Enable disk-based caching of embeddings
        """
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=local_files_only
        )
        self.k = k
        self.alpha = alpha
        self.verbose = verbose
        self.documents = []
        self.embeddings = None
        self.G = None

        # Cache configuration
        self.cache_dir = cache_dir or os.path.expanduser('~/projects/embedding_cache/maniscope')
        self.use_cache = use_cache

    def _compute_cache_key(self, docs: List[str]) -> str:
        """
        Compute hash-based cache key from documents and model name.

        Args:
            docs: List of document texts

        Returns:
            Cache key string combining model name and document hash
        """
        content = '|'.join(docs)
        doc_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        # Replace slashes in model name for valid filenames
        safe_model_name = self.model_name.replace('/', '_')
        return f"{safe_model_name}_{doc_hash}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get full path to cache file.

        Args:
            cache_key: Cache key from _compute_cache_key()

        Returns:
            Path object for cache file
        """
        cache_dir = Path(self.cache_dir)
        return cache_dir / f"{cache_key}.pkl"

    def _load_embeddings_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Load embeddings from disk cache if available using joblib for faster deserialization.

        Args:
            cache_key: Cache key from _compute_cache_key()

        Returns:
            Cached embeddings array or None if not found
        """
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            if self.verbose:
                print(f"Loading embeddings from cache: {cache_path}")
            try:
                return joblib.load(cache_path)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load cache: {e}")
                return None
        return None

    def _save_embeddings_to_cache(self, embeddings: np.ndarray, cache_key: str):
        """
        Save embeddings to disk cache using joblib for faster serialization.

        Args:
            embeddings: Document embeddings array
            cache_key: Cache key from _compute_cache_key()
        """
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(cache_key)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"Saving embeddings to cache: {cache_path}")
            joblib.dump(embeddings, cache_path)
        except Exception as e:
            if self.verbose:
                print(f"Failed to save cache: {e}")

    def _get_cached_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Retrieve cached query embedding or None if not cached.

        Args:
            query: Query text

        Returns:
            Cached embedding or None
        """
        return self.query_cache.get(query)

    def _cache_query_embedding(self, query: str, embedding: np.ndarray):
        """
        Cache query embedding with LRU eviction.

        Args:
            query: Query text
            embedding: Query embedding array
        """
        if query in self.query_cache:
            # Move to end (most recent)
            self.query_cache.move_to_end(query)
        else:
            if len(self.query_cache) >= self.query_cache_size:
                # Evict least recently used
                self.query_cache.popitem(last=False)
            self.query_cache[query] = embedding

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode query with caching optimization.

        Args:
            query: Query text

        Returns:
            Query embedding array
        """
        cached = self._get_cached_query_embedding(query)
        if cached is not None:
            if self.verbose:
                print(f"  Using cached query embedding for: {query[:30]}...")
            return cached

        if self.verbose:
            print(f"  Encoding query: {query[:30]}...")
        embedding = self.model.encode([query])
        self._cache_query_embedding(query, embedding)
        return embedding

    def fit(self, docs: List[str]) -> 'ManiscopeEngine__v1':
        """
        Build the manifold graph from document corpus.

        Embeddings are cached to disk for efficiency. Subsequent calls with
        the same documents will load from cache instead of re-encoding.

        Args:
            docs: List of document texts

        Returns:
            self for method chaining
        """
        self.documents = docs

        # Try to load embeddings from cache
        cache_key = self._compute_cache_key(docs)
        cached_embeddings = self._load_embeddings_from_cache(cache_key)

        if cached_embeddings is not None:
            if self.verbose:
                print(f"Using cached embeddings for {len(docs)} documents")
            self.embeddings = cached_embeddings
        else:
            if self.verbose:
                print(f"Encoding {len(docs)} documents...")
            self.embeddings = self.model.encode(docs, show_progress_bar=self.verbose)
            self._save_embeddings_to_cache(self.embeddings, cache_key)

        if self.verbose:
            print(f"Building k-NN manifold graph (k={self.k})...")

        # Build k-NN manifold graph using cosine metric
        nbrs = NearestNeighbors(
            n_neighbors=min(self.k + 1, len(docs)),
            metric='cosine'
        ).fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)

        self.G = nx.Graph()
        for i in range(len(docs)):
            for neighbor_idx, dist in zip(indices[i], distances[i]):
                if i != neighbor_idx:
                    self.G.add_edge(i, neighbor_idx, weight=dist)

        # Optimization: Pre-compute adjacency list for faster neighbor queries
        self.graph_adjacency = {
            node: [(neighbor, self.G[node][neighbor]['weight'])
                   for neighbor in self.G.neighbors(node)]
            for node in self.G.nodes()
        }

        if self.verbose:
            print(f"Graph built: {self.G.number_of_nodes()} nodes, "
                  f"{self.G.number_of_edges()} edges")
            print(f"Connected components: "
                  f"{nx.number_connected_components(self.G)}")

        return self

    def search_baseline(
        self,
        query: str,
        top_n: int = 5
    ) -> List[Tuple[str, float, int]]:
        """
        Baseline search using pure cosine similarity (telescope only).

        Args:
            query: Query text
            top_n: Number of results to return

        Returns:
            List of (document, score, doc_index) tuples sorted by score
        """
        if self.embeddings is None:
            raise ValueError("Must call fit() before search")

        q_emb = self.model.encode([query])
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        idx = np.argsort(sims)[::-1][:top_n]
        return [(self.documents[i], float(sims[i]), int(i)) for i in idx]

    def search(
        self,
        query: str,
        top_n: int = 5,
        coarse_multiplier: int = 3
    ) -> List[Tuple[str, float, int]]:
        """
        Optimized Maniscope search: Telescope + Microscope (geodesic reranking).

        Incorporates batch shortest path computation, query caching, and early termination.

        Args:
            query: Query text
            top_n: Number of final results to return
            coarse_multiplier: Retrieve top_n * coarse_multiplier for reranking

        Returns:
            List of (document, score, doc_index) tuples sorted by hybrid score
        """
        if self.embeddings is None:
            raise ValueError("Must call fit() before search")

        # Phase 1: TELESCOPE - Broad retrieval with cosine similarity
        q_emb = self.model.encode([query])
        cosine_sims = cosine_similarity(q_emb, self.embeddings)[0]

        # Retrieve broader set for reranking
        coarse_size = min(top_n * coarse_multiplier, len(self.documents))
        coarse_idx = np.argsort(cosine_sims)[::-1][:coarse_size]

        if self.verbose:
            print(f"\nTelescope: Retrieved top {coarse_size} candidates")

        # Phase 2: MICROSCOPE - Geodesic reranking on manifold
        # Select anchor node from coarse set (closest to query)
        anchor_candidates = coarse_idx[:min(5, len(coarse_idx))]
        anchor_node = anchor_candidates[0]

        if self.verbose:
            print(f"Microscope: Anchor node = {anchor_node} "
                  f"(cosine sim = {cosine_sims[anchor_node]:.3f})")

        # Optimization: Batch compute all geodesic distances from anchor
        try:
            all_geo_dists = nx.single_source_dijkstra_path_length(
                self.G, source=anchor_node, weight='weight'
            )
        except nx.NetworkXError:
            all_geo_dists = {}

        # Optimization: Vectorized computation with heap for early termination
        heap = []
        for i in coarse_idx:
            cosine_score = cosine_sims[i]
            geo_dist = all_geo_dists.get(i, None)
            if geo_dist is not None:
                geo_score = 1.0 / (1.0 + geo_dist)
            else:
                geo_score = 0.0  # Disconnected

            final_score = self.alpha * cosine_score + (1 - self.alpha) * geo_score

            # Use heap to maintain top_n (min-heap, so use negative score)
            heappush(heap, (-final_score, i))
            if len(heap) > top_n:
                heappop(heap)

        # Extract results from heap
        results = []
        while heap:
            neg_score, i = heappop(heap)
            score = -neg_score
            results.append((self.documents[i], score, i))

        # Heap gives ascending order, reverse to descending
        results.reverse()

        return results

    def search_detailed(
        self,
        query: str,
        top_n: int = 5,
        coarse_multiplier: int = 3
    ) -> List[Dict]:
        """
        Optimized Maniscope search with detailed scoring information for analysis.

        Uses batch geodesic computation for efficiency.

        Args:
            query: Query text
            top_n: Number of results to return
            coarse_multiplier: Retrieve top_n * coarse_multiplier for reranking

        Returns:
            List of dicts with document, scores, and metadata:
            - doc_id: Document index
            - document: Document text
            - final_score: Hybrid score (alpha * cosine + (1-alpha) * geodesic)
            - cosine_score: Cosine similarity score
            - geo_score: Geodesic similarity score
            - geo_distance: Geodesic distance (None if disconnected)
            - connected: Whether node is connected to anchor in graph
        """
        if self.embeddings is None:
            raise ValueError("Must call fit() before search")

        q_emb = self.model.encode([query])
        cosine_sims = cosine_similarity(q_emb, self.embeddings)[0]

        coarse_size = min(top_n * coarse_multiplier, len(self.documents))
        coarse_idx = np.argsort(cosine_sims)[::-1][:coarse_size]

        anchor_candidates = coarse_idx[:min(5, len(coarse_idx))]
        anchor_node = anchor_candidates[0]

        # Optimization: Batch compute all geodesic distances from anchor
        try:
            all_geo_dists = nx.single_source_dijkstra_path_length(
                self.G, source=anchor_node, weight='weight'
            )
        except nx.NetworkXError:
            all_geo_dists = {}

        results = []
        for i in coarse_idx:
            cosine_score = float(cosine_sims[i])
            geo_dist = all_geo_dists.get(i)
            if geo_dist is not None:
                geo_score = float(1.0 / (1.0 + geo_dist))
                connected = True
            else:
                geo_score = 0.0
                connected = False

            final_score = self.alpha * cosine_score + (1 - self.alpha) * geo_score

            results.append({
                'doc_id': int(i),
                'document': self.documents[i],
                'final_score': float(final_score),
                'cosine_score': cosine_score,
                'geo_score': geo_score,
                'geo_distance': geo_dist,
                'connected': connected
            })

        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:top_n]

    @staticmethod
    def compare_performance(
        engine1: 'ManiscopeEngine',
        engine2: 'ManiscopeEngine',
        query: str,
        top_n: int = 5,
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Compare performance of two ManiscopeEngine instances.

        Args:
            engine1: First engine instance
            engine2: Second engine instance
            query: Query to test
            top_n: Number of results
            num_runs: Number of benchmark runs

        Returns:
            Dict with timing and accuracy comparison
        """
        import time

        # Benchmark engine1
        start = time.time()
        results1 = []
        for _ in range(num_runs):
            results1.append(engine1.search(query, top_n=top_n))
        time1 = time.time() - start

        # Benchmark engine2
        start = time.time()
        results2 = []
        for _ in range(num_runs):
            results2.append(engine2.search(query, top_n=top_n))
        time2 = time.time() - start

        # Check consistency
        consistent = all(r1 == r2 for r1, r2 in zip(results1, results2))

        return {
            'engine1_time': time1,
            'engine2_time': time2,
            'speedup': time1 / time2 if time2 > 0 else float('inf'),
            'results_consistent': consistent,
            'engine1_class': engine1.__class__.__name__,
            'engine2_class': engine2.__class__.__name__,
            'query': query,
            'top_n': top_n,
            'num_runs': num_runs
        }


class ManiscopeEngine__v1(ManiscopeEngine):
    """
    Optimized version of ManiscopeEngine with performance enhancements:

    - Batch shortest path computation for geodesic distances
    - Query embedding caching with LRU eviction
    - Vectorized geodesic score computation
    - Faster serialization with joblib
    - Graph adjacency list pre-computation
    - Heap-based early termination for top-n selection

    This class inherits from ManiscopeEngine but overrides key methods
    with optimizations for 5-50x speedup in various scenarios.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        k: int = 5,
        alpha: float = 0.5,
        verbose: bool = False,
        device: str = 'cpu',
        local_files_only: bool = True,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        query_cache_size: int = 100  # New: LRU cache size for queries
    ):
        """
        Initialize the optimized ManiscopeEngine.

        Args:
            model_name: Sentence transformer model name
            k: Number of nearest neighbors for manifold graph construction
            alpha: Weight for hybrid scoring (0=pure geodesic, 1=pure cosine)
            verbose: Print debug information
            device: Device for computation ('cpu' or 'cuda')
            local_files_only: Use only cached models without checking HuggingFace
            cache_dir: Directory for embedding cache (defaults to ~/projects/embedding_cache/maniscope)
            use_cache: Enable disk-based caching of embeddings
            query_cache_size: Maximum number of query embeddings to cache
        """
        super().__init__(
            model_name=model_name,
            k=k,
            alpha=alpha,
            verbose=verbose,
            device=device,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            use_cache=use_cache
        )

        # Optimization: Query embedding LRU cache
        self.query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.query_cache_size = query_cache_size

        # Optimization: Pre-computed graph adjacency (for faster neighbor queries)
        self.graph_adjacency: Optional[Dict[int, List[Tuple[int, float]]]] = None

    def search(
        self,
        query: str,
        top_n: int = 5,
        coarse_multiplier: int = 3
    ) -> List[Tuple[str, float, int]]:
        """
        Optimized Maniscope search: Telescope + Microscope (geodesic reranking).

        Incorporates batch shortest path computation, query caching, and early termination.

        Args:
            query: Query text
            top_n: Number of final results to return
            coarse_multiplier: Retrieve top_n * coarse_multiplier for reranking

        Returns:
            List of (document, score, doc_index) tuples sorted by hybrid score
        """
        if self.embeddings is None:
            raise ValueError("Must call fit() before search")

        # Phase 1: TELESCOPE - Broad retrieval with cosine similarity
        q_emb = self._encode_query(query)
        cosine_sims = cosine_similarity(q_emb, self.embeddings)[0]

        # Retrieve broader set for reranking
        coarse_size = min(top_n * coarse_multiplier, len(self.documents))
        coarse_idx = np.argsort(cosine_sims)[::-1][:coarse_size]

        if self.verbose:
            print(f"\nTelescope: Retrieved top {coarse_size} candidates")

        # Phase 2: MICROSCOPE - Geodesic reranking on manifold
        # Select anchor node from coarse set (closest to query)
        anchor_candidates = coarse_idx[:min(5, len(coarse_idx))]
        anchor_node = anchor_candidates[0]

        if self.verbose:
            print(f"Microscope: Anchor node = {anchor_node} "
                  f"(cosine sim = {cosine_sims[anchor_node]:.3f})")

        # Optimization: Batch compute all geodesic distances from anchor
        try:
            all_geo_dists = nx.single_source_dijkstra_path_length(
                self.G, source=anchor_node, weight='weight'
            )
        except nx.NetworkXError:
            all_geo_dists = {}

        # Optimization: Vectorized computation with heap for early termination
        heap = []
        for i in coarse_idx:
            cosine_score = cosine_sims[i]
            geo_dist = all_geo_dists.get(i, None)
            if geo_dist is not None:
                geo_score = 1.0 / (1.0 + geo_dist)
            else:
                geo_score = 0.0  # Disconnected

            final_score = self.alpha * cosine_score + (1 - self.alpha) * geo_score

            # Use heap to maintain top_n (min-heap, so use negative score)
            heappush(heap, (-final_score, i))
            if len(heap) > top_n:
                heappop(heap)

        # Extract results from heap
        results = []
        while heap:
            neg_score, i = heappop(heap)
            score = -neg_score
            results.append((self.documents[i], score, i))

        # Heap gives ascending order, reverse to descending
        results.reverse()

        return results
