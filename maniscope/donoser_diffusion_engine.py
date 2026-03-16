"""
DonoserDiffusionEngine: Diffusion Reranking baseline (Donoser & Bischof, CVPR 2013).

Reference:
    Donoser, M. & Bischof, H. (2013).
    Diffusion Processes for Retrieval Revisited.
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Key contribution — Pairwise Support Propagation (PSP):
    Enforces *mutual* k-NN membership before propagation.
    Edge (i, j) is retained only if j ∈ kNN(i) AND i ∈ kNN(j).
    This bidirectional agreement criterion produces a sparser, more
    structurally coherent graph than directed k-NN (Zhou 2003) or
    one-directional diffusion (Dampanaboina 2025).

Algorithm:
    Given a query q and candidate documents {d_1, ..., d_n}:
    1. Embed query + documents → vectors in R^d
    2. Build directed k-NN cosine similarity graph over documents
    3. Retain only *mutual* edges: W_mut[i,j] = W[i,j] if W[j,i] > 0 else 0
    4. Symmetric normalization: S = D^{-1/2} W_mut D^{-1/2}
    5. Seed vector: f_0 = cosine_sim(query_emb, doc_emb_i), clipped & normalised
    6. Power iteration: f_{t+1} = α · S · f_t + (1 − α) · f_0
       until ||f_{t+1} − f_t|| < tol or max_iter reached
    7. Return converged f* as relevance scores

Positioning vs other baselines:
    - ManifoldRankingEngine (Zhou 2003):
        Same normalization (D^{-1/2} W D^{-1/2}), but:
        · Directed k-NN (not mutual) → denser graph
        · Closed-form solve (I − αS)^{-1} y instead of power iteration
    - DiffusionRAGEngine (Dampanaboina 2025):
        Same iterative scheme, but:
        · Directed k-NN (not mutual)
        · Row-normalized (D^{-1} W) instead of symmetric normalization
    - Maniscope:
        Graph-based, but uses Dijkstra geodesic paths — local path
        distances vs. global spectral smoothing.
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import scipy.sparse as sp


class DonoserDiffusionEngine:
    """
    Diffusion Reranking (PSP variant) — Donoser & Bischof (CVPR 2013).

    Applies symmetrically-normalised power-iteration diffusion over a
    *mutual* k-NN graph of candidate documents, seeded by query-document
    cosine similarities.

    Args:
        model_name: Sentence-Transformers model for encoding
        k: Number of nearest neighbors for directed graph (before mutual filtering)
        alpha: Diffusion weight (higher = more graph smoothing, less query-seed
               influence; typical range 0.90–0.99)
        max_iter: Maximum diffusion iterations
        tol: Convergence tolerance (stops when ||f_{t+1} − f_t||₂ < tol)
        device: 'cuda', 'cpu', or None for auto-detect
        verbose: Print debug info
    """

    def __init__(
        self,
        model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        k: int = 5,
        alpha: float = 0.99,
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

    def _build_mutual_knn_affinity(self, embs: np.ndarray) -> sp.csr_matrix:
        """
        Build mutual k-NN affinity matrix (PSP key step).

        Step 1 — Directed k-NN graph:
            W_dir[i,j] = cos_sim(e_i, e_j)   if j ∈ kNN(i, k)
                       = 0                    otherwise

        Step 2 — Mutual filtering (Donoser & Bischof PSP):
            W_mut[i,j] = W_dir[i,j]           if W_dir[j,i] > 0  (bidirectional)
                       = 0                    otherwise

        Result is symmetric by construction (only mutual edges kept).
        """
        n = len(embs)
        k = min(self.k, n - 1)

        # Full cosine similarity matrix (unit-normalised embeddings)
        cos_sim = embs @ embs.T   # (n, n)

        # --- Step 1: directed k-NN ---
        rows, cols, vals = [], [], []
        for i in range(n):
            sims_i = cos_sim[i].copy()
            sims_i[i] = -np.inf   # exclude self
            nn_idx = np.argpartition(sims_i, -k)[-k:]
            for j in nn_idx:
                w = float(max(sims_i[j], 0.0))
                if w > 0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(w)

        W_dir = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

        # --- Step 2: mutual filtering ---
        # An edge (i,j) survives only if both W_dir[i,j] > 0 and W_dir[j,i] > 0.
        # Equivalent to: W_mut = W_dir * W_dir.T  (element-wise, then clip to original values)
        mask = (W_dir > 0).multiply(W_dir.T > 0)   # boolean mask of mutual edges
        W_mut = W_dir.multiply(mask)                # retain original weights where mutual

        # Symmetrize (W_mut is already structurally symmetric, but weights may differ
        # on each directed edge; take the average to enforce exact symmetry)
        W_sym = (W_mut + W_mut.T) * 0.5
        return sp.csr_matrix(W_sym)

    def _symmetric_normalize(self, W: sp.csr_matrix) -> sp.csr_matrix:
        """S = D^{-1/2} W D^{-1/2}  (same normalization as Zhou 2003)."""
        d = np.array(W.sum(axis=1)).flatten()
        # Compute inverse sqrt only on non-zero degrees to avoid divide-by-zero
        # warnings from np.where evaluating both branches before masking.
        d_inv_sqrt = np.zeros_like(d)
        mask = d > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        return sp.csr_matrix(D_inv_sqrt @ W @ D_inv_sqrt)

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
        Rerank documents using mutual-k-NN diffusion (PSP).

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
        doc_embs = self._doc_embeddings    # (n, d), unit-normalized
        query_emb = self._encode([query])  # (1, d), unit-normalized

        # Seed vector: query-document cosine similarity
        f0 = (doc_embs @ query_emb.T).flatten()   # (n,)
        f0 = np.clip(f0, 0.0, None)
        if f0.sum() > 0:
            f0 = f0 / f0.sum()

        # Build mutual k-NN affinity and normalize
        W_mut = self._build_mutual_knn_affinity(doc_embs)

        # Fallback: if mutual filtering eliminated too many edges, use directed graph
        if W_mut.nnz == 0:
            if self.verbose:
                print("[DonoserDiffusion] Mutual k-NN graph empty — falling back to cosine scores")
            return f0

        S = self._symmetric_normalize(W_mut)

        # Power iteration: f_{t+1} = α·S·f_t + (1−α)·f_0
        f = f0.copy()
        for iteration in range(self.max_iter):
            f_new = self.alpha * S.dot(f) + (1.0 - self.alpha) * f0
            delta = float(np.linalg.norm(f_new - f))
            f = f_new
            if delta < self.tol:
                if self.verbose:
                    print(f"[DonoserDiffusion] Converged at iteration {iteration+1}  δ={delta:.2e}")
                break

        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        f = np.clip(f, 0.0, None)

        if self.verbose:
            top_idx = int(np.argmax(f))
            mutual_edges = W_mut.nnz // 2
            print(f"[DonoserDiffusion] mutual edges={mutual_edges}  top doc={top_idx}  score={f[top_idx]:.4f}")

        return f
