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
from typing import List, Tuple, Dict, Optional


class ManiscopeEngine:
    """
    Multi-scale retrieval engine combining cosine similarity (telescope)
    and geodesic distance on k-NN manifold (microscope).

    This class provides efficient semantic search by exploiting local
    manifold structure in embedding spaces through geodesic distances
    on k-nearest neighbor graphs.

    Attributes:
        model: Sentence transformer for text encoding
        k: Number of nearest neighbors for manifold graph
        alpha: Weight for hybrid scoring (0=pure geodesic, 1=pure cosine)
        verbose: Enable debug output
        documents: Fitted document corpus
        embeddings: Document embeddings
        G: k-NN manifold graph
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        k: int = 5,
        alpha: float = 0.5,
        verbose: bool = False,
        device: str = 'cpu',
        local_files_only: bool = True
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
        """
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

    def fit(self, docs: List[str]) -> 'ManiscopeEngine':
        """
        Build the manifold graph from document corpus.

        Args:
            docs: List of document texts

        Returns:
            self for method chaining
        """
        if self.verbose:
            print(f"Encoding {len(docs)} documents...")

        self.documents = docs
        self.embeddings = self.model.encode(docs, show_progress_bar=self.verbose)

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
        Maniscope search: Telescope + Microscope (geodesic reranking).

        This is the main search method combining global cosine similarity
        with local geodesic distances on the manifold.

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
            final_score = self.alpha * cosine_score + (1 - self.alpha) * geo_score

            results.append((self.documents[i], final_score, i))

        # Sort by final hybrid score
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_n]

    def search_detailed(
        self,
        query: str,
        top_n: int = 5,
        coarse_multiplier: int = 3
    ) -> List[Dict]:
        """
        Maniscope search with detailed scoring information for analysis.

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

    def compare_methods(
        self,
        query: str,
        top_n: int = 5
    ) -> Dict:
        """
        Compare baseline cosine similarity vs maniscope geodesic reranking.

        Args:
            query: Query text
            top_n: Number of results to compare

        Returns:
            Dict with:
            - query: Query text
            - baseline_results: Results from pure cosine similarity
            - maniscope_results: Results from geodesic reranking
            - ranking_changed: Whether rankings differ
            - baseline_top1: Top-1 doc ID from baseline
            - maniscope_top1: Top-1 doc ID from maniscope
        """
        if self.embeddings is None:
            raise ValueError("Must call fit() before search")

        baseline = self.search_baseline(query, top_n)
        maniscope = self.search_detailed(query, top_n)

        baseline_ids = [doc_id for _, _, doc_id in baseline]
        maniscope_ids = [result['doc_id'] for result in maniscope]

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
            'ranking_changed': baseline_ids != maniscope_ids,
            'baseline_top1': baseline_ids[0],
            'maniscope_top1': maniscope_ids[0]
        }
