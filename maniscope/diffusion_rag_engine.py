"""
DiffusionRAGEngine: Diffusion-Aided RAG reranking baseline.

Inspired by:
    Dampanaboina, S., et al. (2025).
    Diffusion-Aided Retrieval-Augmented Generation.
    CLiC-it 2025.

    This work adapts offline manifold/diffusion ranking (Zhou 2003,
    Donoser & Bischof 2013) to the online RAG reranking setting.

Algorithm (Random Walk with Restart / Diffusion):
    Given a query q and candidate documents {d_1, ..., d_n}:
    1. Embed query + documents → vectors in R^d
    2. Build row-normalized k-NN transition matrix T = D^{-1} W
       where W is the k-NN cosine similarity graph over documents
    3. Initial score vector f_0 = cosine_sim(query_emb, doc_emb_i)
    4. Iterate: f_{t+1} = α · T^T · f_t + (1 − α) · f_0
       (Random Walk with Restart from the query-similarity seed)
    5. Return converged f* as relevance scores

Key differences from ManifoldRankingEngine (Zhou 2003):
    - Transition matrix uses asymmetric row normalisation (D^{-1} W)
      rather than symmetric Laplacian (D^{-1/2} W D^{-1/2}).
    - Solution is iterative (power iteration), not closed-form.
    - Convergence is faster in practice and numerically more stable
      for sparse, unweighted k-NN graphs.
    - Conceptually maps to "PageRank-style" diffusion from the query seed,
      which matches the RAG retrieval framing of Dampanaboina et al.
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import scipy.sparse as sp


class DiffusionRAGEngine:
    """
    Diffusion-Aided RAG reranker (Dampanaboina et al., CLiC-it 2025).

    Applies iterative random-walk-with-restart over a k-NN graph of the
    candidate documents, seeded by query-document cosine similarities.

    Args:
        model_name: Sentence-Transformers model for encoding
        k: Number of nearest neighbors in the diffusion graph
        alpha: Restart probability complement (higher = more diffusion,
               less query-seed influence; typical range 0.85–0.99)
        max_iter: Maximum diffusion iterations
        tol: Convergence tolerance (stops early when ||f_t+1 - f_t|| < tol)
        device: 'cuda', 'cpu', or None for auto-detect
        verbose: Print debug info
    """

    def __init__(
        self,
        model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        k: int = 5,
        alpha: float = 0.95,
        max_iter: int = 30,
        tol: float = 1e-6,
        device: Optional[str] = None,
        verbose: bool = False
    ):
        self.k = k
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

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

    def _build_transition_matrix(self, embs: np.ndarray) -> sp.csr_matrix:
        """
        Build row-normalised k-NN transition matrix T = D^{-1} W.

        W_ij = cos_sim(e_i, e_j)  if j ∈ kNN(i) (directed, k-nearest)
             = 0                   otherwise
        T_ij = W_ij / sum_j W_ij  (row-normalised, so T is a stochastic matrix)
        """
        n = len(embs)
        k = min(self.k, n - 1)

        # Cosine similarities (embeddings are unit-normalized)
        cos_sim = embs @ embs.T   # (n, n)

        rows, cols, vals = [], [], []
        for i in range(n):
            sims_i = cos_sim[i].copy()
            sims_i[i] = -np.inf   # exclude self
            # Top-k by cosine similarity
            nn_idx = np.argpartition(sims_i, -k)[-k:]
            for j in nn_idx:
                w = float(max(sims_i[j], 0.0))
                if w > 0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(w)

        W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

        # Row-normalise → stochastic transition matrix
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums_inv = np.zeros_like(row_sums)
        mask = row_sums > 0
        row_sums_inv[mask] = 1.0 / row_sums[mask]
        T = sp.diags(row_sums_inv) @ W
        return sp.csr_matrix(T)

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
        Rerank documents using diffusion from query seed.

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

        # Seed: query-document cosine similarity
        f0 = (doc_embs @ query_emb.T).flatten()   # (n,)
        f0 = np.clip(f0, 0.0, None)
        if f0.sum() > 0:
            f0 = f0 / f0.sum()

        # Build transition matrix
        T = self._build_transition_matrix(doc_embs)
        T_T = T.T   # transpose for propagation: f_{t+1} = α T^T f_t + (1-α) f0

        # Power iteration (random walk with restart)
        f = f0.copy()
        for iteration in range(self.max_iter):
            f_new = self.alpha * T_T.dot(f) + (1.0 - self.alpha) * f0
            delta = float(np.linalg.norm(f_new - f))
            f = f_new
            if delta < self.tol:
                if self.verbose:
                    print(f"[DiffusionRAG] Converged at iteration {iteration+1}  δ={delta:.2e}")
                break

        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        f = np.clip(f, 0.0, None)

        if self.verbose:
            top_idx = int(np.argmax(f))
            print(f"[DiffusionRAG] top doc={top_idx}  score={f[top_idx]:.4f}")

        return f
