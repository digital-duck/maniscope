#!/usr/bin/env python3
"""Analyze grid search results and generate Figure 1 contour heatmap.

Reads JSON files from output/grid_search/ and produces:
  - Console summary tables (k sweep, alpha sweep)
  - output/figures/figure3_contour_nfcorpus.pdf  (k x alpha NDCG@10 heatmap)
  - output/figures/figure3_contour_nfcorpus.png

Run from RAG-ReRanker root:
  python3 scripts/analyze_grid_search.py
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).parent.parent
GRID_DIR = ROOT / 'output' / 'grid_search'
FIG_DIR = ROOT / 'output' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_latest(dataset: str, k_range: str, alpha_range: str) -> dict | None:
    """Return the most recent grid search JSON matching dataset + ranges."""
    pattern = str(GRID_DIR / f'grid_search_{dataset}_*.json')
    candidates = []
    for path in glob.glob(pattern):
        d = json.load(open(path))
        m = d['metadata']
        if m.get('k_range') == k_range and m.get('alpha_range') == alpha_range:
            candidates.append((m['timestamp'], d))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]  # most recent


def print_table(title: str, headers: list, rows: list):
    col_w = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2
             for i, h in enumerate(headers)]
    sep = '+' + '+'.join('-' * w for w in col_w) + '+'
    fmt = '|' + '|'.join(f' {{:<{w-2}}} ' for w in col_w) + '|'
    print(f'\n{title}')
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


# ── k sweep (alpha=0.5) ───────────────────────────────────────────────────────

print('\n' + '='*70)
print('k SWEEP  (alpha=0.5 fixed)')
print('='*70)

for dataset in ['nfcorpus', 'trec-covid', 'msmarco']:
    d = load_latest(dataset, '3:15:2', '0.5:0.5:1')
    if d is None:
        print(f'  {dataset}: no data found'); continue
    rows = [(r['k'], f"{r['NDCG@10']:.4f}", f"{r['MRR@10']:.4f}")
            for r in d['results']]
    print_table(f'  {dataset.upper()}  (n={d["metadata"]["num_queries"]} queries)',
                ['k', 'NDCG@10', 'MRR@10'], rows)

# ── alpha sweep (k=5) ────────────────────────────────────────────────────────

print('\n' + '='*70)
print('alpha SWEEP  (k=5 fixed)')
print('='*70)

for dataset in ['nfcorpus', 'trec-covid', 'msmarco']:
    d = load_latest(dataset, '5:5:1', '0.0:1.0:0.25')
    if d is None:
        print(f'  {dataset}: no data found'); continue
    rows = [(f"{r['alpha']:.2f}", f"{r['NDCG@10']:.4f}", f"{r['MRR@10']:.4f}")
            for r in d['results']]
    print_table(f'  {dataset.upper()}  (n={d["metadata"]["num_queries"]} queries)',
                ['alpha', 'NDCG@10', 'MRR@10'], rows)

# ── contour heatmap (k x alpha, NFCorpus) ────────────────────────────────────

print('\n' + '='*70)
print('CONTOUR HEATMAP  (NFCorpus, k x alpha joint grid)')
print('='*70)

d = load_latest('nfcorpus', '3:15:2', '0.0:1.0:0.25')
if d is None:
    print('  No joint grid data found — run scripts/k-alpha-sweep.sh first')
else:
    results = d['results']
    k_vals = sorted(set(r['k'] for r in results))
    a_vals = sorted(set(r['alpha'] for r in results))

    # Build NDCG@10 matrix: rows=k, cols=alpha
    grid = np.zeros((len(k_vals), len(a_vals)))
    for r in results:
        ki = k_vals.index(r['k'])
        ai = a_vals.index(r['alpha'])
        grid[ki, ai] = r['NDCG@10']

    # Print raw grid
    header = ['k \\ α'] + [f'{a:.2f}' for a in a_vals]
    rows = [[k_vals[i]] + [f'{grid[i,j]:.4f}' for j in range(len(a_vals))]
            for i in range(len(k_vals))]
    print_table('  NDCG@10 grid', header, rows)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    from matplotlib.colors import LinearSegmentedColormap
    cmap_ylgrey = LinearSegmentedColormap.from_list('ylgrey', ['#ffffcc', '#d0d0d0'])
    im = ax.imshow(grid, aspect='auto', cmap=cmap_ylgrey,
                   vmin=grid.min() - 0.002, vmax=grid.max() + 0.002)
    plt.colorbar(im, ax=ax, label='NDCG@10')

    ax.set_xticks(range(len(a_vals)))
    ax.set_xticklabels([f'{a:.2f}' for a in a_vals])
    ax.set_yticks(range(len(k_vals)))
    ax.set_yticklabels([str(k) for k in k_vals])
    ax.set_xlabel(r'Blend weight $\alpha$ (0=pure geodesic, 1=pure cosine)')
    ax.set_ylabel(r'Graph connectivity $k$')
    ax.set_title('Figure 1: NDCG@10 sensitivity to $k$ and $\\alpha$ -- NFCorpus (323 queries)')

    # Annotate each cell
    for i in range(len(k_vals)):
        for j in range(len(a_vals)):
            ax.text(j, i, f'{grid[i,j]:.4f}', ha='center', va='center',
                    fontsize=10, color='black')

    # Mark default operating point (k=5, alpha=0.5)
    if 5 in k_vals and 0.5 in a_vals:
        ax.add_patch(plt.Rectangle(
            (a_vals.index(0.5) - 0.5, k_vals.index(5) - 0.5),
            1, 1, fill=False, edgecolor='blue', linewidth=2.5, label='Default (k=5, α=0.5)'))
        ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    out_pdf = FIG_DIR / 'figure1_contour_nfcorpus.pdf'
    out_png = FIG_DIR / 'figure1_contour_nfcorpus.png'
    plt.savefig(out_pdf, dpi=150, bbox_inches='tight')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'\n  Saved: {out_pdf}')
    print(f'  Saved: {out_png}')
