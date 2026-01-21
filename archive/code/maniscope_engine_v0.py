"""
ManiscopeEngine: Multi-scale Information Retrieval with Geodesic Reranking

This implements a two-stage retrieval system:
1. Telescope (Coarse): Broad retrieval using cosine similarity
2. Microscope (Fine): Geodesic reranking on k-NN manifold graph

Key improvements over baseline:
- Better anchor node selection (from candidate set)
- Hybrid scoring combining cosine + geodesic
- Robust handling of disconnected components
- Configurable parameters for k and retrieval depth
"""

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class ManiscopeEngine_v0:
    """
    Multi-scale retrieval engine combining cosine similarity (telescope)
    and geodesic distance on k-NN manifold (microscope).
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', k: int = 5,
                 alpha: float = 0.5, verbose: bool = False, device: str = 'cpu',
                 local_files_only: bool = True):
        """
        Initialize the ManiscopeEngine.

        Args:
            model_name: Sentence transformer model name
            k: Number of nearest neighbors for manifold graph construction
            alpha: Weight for hybrid scoring (0=pure geodesic, 1=pure cosine)
            verbose: Print debug information
            device: Device for computation ('cpu' or 'cuda')
            local_files_only: If True, use only cached models without checking HuggingFace
                             (Recommended for experiments to avoid network issues)
        """
        # Load model with local_files_only to avoid checking HuggingFace on every startup
        # This prevents timeout errors when HuggingFace is slow/unavailable
        # and makes experiments more reliable with consistent cached models
        self.model = SentenceTransformer(model_name, device=device,
                                        local_files_only=local_files_only)
        self.k = k
        self.alpha = alpha
        self.verbose = verbose
        self.documents = []
        self.embeddings = None
        self.G = None

    def fit(self, docs: List[str]):
        """
        Build the manifold graph from document corpus.

        Args:
            docs: List of document texts
        """
        if self.verbose:
            print(f"Encoding {len(docs)} documents...")

        self.documents = docs
        self.embeddings = self.model.encode(docs, show_progress_bar=self.verbose)

        if self.verbose:
            print(f"Building k-NN manifold graph (k={self.k})...")

        # Build the latent k-NN manifold graph
        # Use cosine metric to respect embedding geometry
        nbrs = NearestNeighbors(n_neighbors=min(self.k + 1, len(docs)),
                               metric='cosine').fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)

        self.G = nx.Graph()
        for i in range(len(docs)):
            for neighbor_idx, dist in zip(indices[i], distances[i]):
                if i != neighbor_idx:
                    # Add edge with distance as weight
                    self.G.add_edge(i, neighbor_idx, weight=dist)

        if self.verbose:
            print(f"Graph built: {self.G.number_of_nodes()} nodes, "
                  f"{self.G.number_of_edges()} edges")
            print(f"Connected components: "
                  f"{nx.number_connected_components(self.G)}")

    def search_baseline(self, query: str, top_n: int = 5) -> List[Tuple[str, float, int]]:
        """
        Baseline: Telescope Only (Pure Cosine Similarity)

        Args:
            query: Query text
            top_n: Number of results to return

        Returns:
            List of (document, score, doc_index) tuples
        """
        q_emb = self.model.encode([query])
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        idx = np.argsort(sims)[::-1][:top_n]
        return [(self.documents[i], float(sims[i]), int(i)) for i in idx]

    def search_maniscope(self, query: str, top_n: int = 5,
                         coarse_multiplier: int = 3) -> List[Tuple[str, float, int]]:
        """
        Maniscope: Telescope + Microscope (Geodesic Reranking)

        Args:
            query: Query text
            top_n: Number of final results to return
            coarse_multiplier: Retrieve top_n * coarse_multiplier for reranking

        Returns:
            List of (document, score, doc_index) tuples
        """
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
        anchor_node = anchor_candidates[0]  # Most similar in coarse set

        if self.verbose:
            print(f"Microscope: Anchor node = {anchor_node} "
                  f"(cosine sim = {cosine_sims[anchor_node]:.3f})")

        results = []
        for i in coarse_idx:
            cosine_score = cosine_sims[i]

            # Calculate geodesic distance on manifold
            try:
                geo_dist = nx.shortest_path_length(
                    self.G, source=anchor_node, target=i, weight='weight'
                )
                # Convert distance to similarity score
                geo_score = 1.0 / (1.0 + geo_dist)
            except nx.NetworkXNoPath:
                # Disconnected component - use only cosine
                geo_score = 0.0
                if self.verbose:
                    print(f"  Node {i} disconnected from anchor")

            # Hybrid scoring: combine cosine and geodesic
            # alpha=1: pure cosine, alpha=0: pure geodesic
            final_score = self.alpha * cosine_score + (1 - self.alpha) * geo_score

            results.append((self.documents[i], final_score, i, cosine_score, geo_score))

        # Sort by final hybrid score
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top_n with simplified format
        return [(doc, score, idx) for doc, score, idx, _, _ in results[:top_n]]

    def search_maniscope_detailed(self, query: str, top_n: int = 5,
                                  coarse_multiplier: int = 3) -> List[dict]:
        """
        Maniscope search with detailed scoring information for analysis.

        Returns:
            List of dicts with document, scores, and metadata
        """
        q_emb = self.model.encode([query])
        cosine_sims = cosine_similarity(q_emb, self.embeddings)[0]

        coarse_size = min(top_n * coarse_multiplier, len(self.documents))
        coarse_idx = np.argsort(cosine_sims)[::-1][:coarse_size]

        anchor_candidates = coarse_idx[:min(5, len(coarse_idx))]
        anchor_node = anchor_candidates[0]

        results = []
        for i in coarse_idx:
            cosine_score = float(cosine_sims[i])

            try:
                geo_dist = nx.shortest_path_length(
                    self.G, source=anchor_node, target=i, weight='weight'
                )
                geo_score = float(1.0 / (1.0 + geo_dist))
                connected = True
            except nx.NetworkXNoPath:
                geo_dist = None
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

    def compare_methods(self, query: str, top_n: int = 5) -> dict:
        """
        Compare baseline and maniscope methods side-by-side.

        Returns:
            Dict with baseline_results, maniscope_results, and analysis
        """
        baseline = self.search_baseline(query, top_n)
        maniscope = self.search_maniscope_detailed(query, top_n)

        # Check if rankings differ
        baseline_ids = [doc_id for _, _, doc_id in baseline]
        maniscope_ids = [result['doc_id'] for result in maniscope]

        ranking_changed = baseline_ids != maniscope_ids

        return {
            'query': query,
            'baseline_results': [
                {'doc_id': doc_id, 'document': doc[:100], 'score': score}
                for doc, score, doc_id in baseline
            ],
            'maniscope_results': [
                {
                    'doc_id': r['doc_id'],
                    'document': r['document'][:100],
                    'final_score': r['final_score'],
                    'cosine_score': r['cosine_score'],
                    'geo_score': r['geo_score']
                }
                for r in maniscope
            ],
            'ranking_changed': ranking_changed,
            'baseline_top1': baseline_ids[0],
            'maniscope_top1': maniscope_ids[0]
        }
