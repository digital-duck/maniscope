"""
ManifoldRankingEngine: Manifold Ranking baseline for RAG reranking.

Implements the algorithm from:
    Zhou, D., Bousquet, O., Lal, T.N., Weston, J., & Schölkopf, B. (2003).
    Learning with Local and Global Consistency.
    Advances in Neural Information Processing Systems (NIPS), 16.

Algorithm:
    Given a query q and candidate documents {d_1, ..., d_n}:
    1. Embed query + documents → vectors in R^d
    2. Build k-NN affinity matrix W over all embeddings (docs only; query is seed)
    3. Symmetric Laplacian normalization: S = D^{-1/2} W D^{-1/2}
    4. Initial label vector y_i = cosine_sim(query_emb, doc_emb_i)
    5. Closed-form solution: f* = (I - α·S)^{-1} · y
    6. Return f* as relevance scores

Positioning vs Maniscope:
    - Zhou 2003 works over the full corpus (offline, quadratic).
      Here we apply it at query time over the top-M candidates (online, O(M²)).
    - Maniscope uses Dijkstra geodesic distances; manifold ranking uses
      Laplacian smoothing — a global spectral approach vs. local path lengths.
    - Both graph sizes are small (M=10–500) making both practical at query time.
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class ManifoldRankingEngine:
    """
    Manifold Ranking reranker (Zhou et al., NIPS 2003), adapted for RAG.

    The engine builds a k-NN affinity graph over the candidate documents,
    uses the query embedding as the seed signal, and solves the global
    consistency criterion to produce smooth relevance scores.

    Args:
        model_name: Sentence-Transformers model for encoding
        k: Number of nearest neighbors for affinity graph construction
        alpha: Propagation weight (closer to 1 = more graph smoothing)
        sigma: RBF kernel bandwidth (None = auto-set to median pairwise distance)
        device: 'cuda', 'cpu', or None for auto-detect
        verbose: Print debug info
    """

    def __init__(
        self,
        model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        k: int = 5,
        alpha: float = 0.99,
        sigma: Optional[float] = None,
        device: Optional[str] = None,
        verbose: bool = False
    ):
        self.k = k
        self.alpha = alpha
        self.sigma = sigma
        self.verbose = verbose

        # Device selection
        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
        self.device = device

        self.encoder = SentenceTransformer(model_name, device=device)
        self._doc_embeddings: Optional[np.ndarray] = None
        self._docs_hash: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def _build_affinity_matrix(self, embs: np.ndarray) -> sp.csr_matrix:
        """
        Build sparse k-NN affinity matrix W using RBF kernel.

        W_ij = exp(-||e_i - e_j||² / σ²)   if j ∈ kNN(i) or i ∈ kNN(j)
             = 0                             otherwise
        """
        n = len(embs)
        k = min(self.k, n - 1)

        # Pairwise squared Euclidean distances (using normalized embeddings)
        # ||e_i - e_j||² = 2(1 - cos_sim(e_i, e_j))  for unit vectors
        cos_sim = embs @ embs.T  # (n, n)
        sq_dist = np.clip(2.0 * (1.0 - cos_sim), 0.0, None)

        # Bandwidth σ: median heuristic if not set
        if self.sigma is None:
            upper = sq_dist[np.triu_indices(n, k=1)]
            sigma2 = float(np.median(upper)) if len(upper) > 0 else 1.0
            if sigma2 == 0.0:
                sigma2 = 1e-6
        else:
            sigma2 = self.sigma ** 2

        # k-NN mask: keep only k nearest neighbors (excluding self)
        # Use argpartition for efficiency
        rows, cols, vals = [], [], []
        for i in range(n):
            dists_i = sq_dist[i].copy()
            dists_i[i] = np.inf   # exclude self
            nn_idx = np.argpartition(dists_i, k)[:k]
            for j in nn_idx:
                w = float(np.exp(-sq_dist[i, j] / sigma2))
                rows.append(i)
                cols.append(j)
                vals.append(w)

        W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        # Symmetrize
        W = (W + W.T)
        # Remove duplicates by converting to COO and back
        W = W.tocsr()
        return W

    def _symmetric_normalize(self, W: sp.csr_matrix) -> sp.csr_matrix:
        """S = D^{-1/2} W D^{-1/2}"""
        d = np.array(W.sum(axis=1)).flatten()
        d_inv_sqrt = np.zeros_like(d)
        mask = d > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        return D_inv_sqrt @ W @ D_inv_sqrt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, docs: List[str]) -> None:
        """Pre-encode documents (cached across queries for the same doc set)."""
        docs_hash = hash(tuple(docs))
        if self._docs_hash == docs_hash and self._doc_embeddings is not None:
            return
        self._doc_embeddings = self._encode(docs)
        self._docs_hash = docs_hash

    def rerank(self, query: str, docs: List[str]) -> np.ndarray:
        """
        Rerank documents using manifold ranking.

        Args:
            query: Query text
            docs: List of candidate document texts

        Returns:
            Score array of shape (len(docs),); higher = more relevant.
        """
        n = len(docs)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        # Encode
        self.fit(docs)
        assert self._doc_embeddings is not None
        doc_embs = self._doc_embeddings   # (n, d), unit-normalized
        query_emb = self._encode([query]) # (1, d), unit-normalized

        # Initial label vector: cosine similarity of query to each doc
        y = (doc_embs @ query_emb.T).flatten()   # (n,)
        y = np.clip(y, 0.0, None)                # non-negative
        if y.sum() > 0:
            y = y / y.sum()                      # normalize

        # Build affinity matrix and normalize
        W = self._build_affinity_matrix(doc_embs)
        S = self._symmetric_normalize(W)

        # Solve: f* = (I - α·S)^{-1} · y
        # Equivalently: (I - α·S) f* = y  → sparse linear system
        n_nodes = S.shape[0]
        A = sp.eye(n_nodes, format='csr') - self.alpha * S

        try:
            # Use sparse direct solver (LU factorisation via SuperLU)
            f_star = spla.spsolve(A, y)
        except Exception:
            # Fallback: iterative Jacobi (10 iterations)
            f_star = y.copy()
            for _ in range(20):
                f_star = self.alpha * (S @ f_star) + (1.0 - self.alpha) * y

        # Ensure non-negative and finite
        f_star = np.nan_to_num(f_star, nan=0.0, posinf=0.0, neginf=0.0)
        f_star = np.clip(f_star, 0.0, None)

        if self.verbose:
            top_idx = int(np.argmax(f_star))
            print(f"[ManifoldRanking] top doc={top_idx}  score={f_star[top_idx]:.4f}")

        return f_star
