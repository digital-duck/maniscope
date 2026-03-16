#!/usr/bin/env python3
"""
Figure 4: UMAP projection of top-M candidate document embeddings.

Panel A (success): cosine top-1 is an outlier; geodesic top-1 is a cluster member.
Panel B (failure): correct answer is isolated; cluster density misleads geodesic.

Output:
  output/figure2_umap.pdf
  output/figure2_umap.png

Run from RAG-ReRanker root: python3 scripts/generate_figure2_umap.py

If no panel case is found with the default max_queries=100, increase it to 323
(full NFCorpus) via --max-queries 323.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src/app/utils')

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics.pairwise import cosine_similarity
from maniscope_engine import ManiscopeEngine_v2o
import umap as umap_module

parser = argparse.ArgumentParser()
parser.add_argument('--max-queries', type=int, default=323,
                    help='Max queries to scan when searching for panel cases (default: 323 = full NFCorpus)')
args = parser.parse_args()

with open('data/dataset-nfcorpus.json') as f:
    nfcorpus = json.load(f)
with open('data/dataset-trec-covid.json') as f:
    treccovid = json.load(f)

mani = ManiscopeEngine_v2o(
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    k=5, alpha=0.5, verbose=False, use_cache=False, use_faiss=False)

reducer = umap_module.UMAP(n_components=2, random_state=42, min_dist=0.3, n_neighbors=5)


def find_case(dataset, want_success, max_queries):
    for item in dataset[:max_queries]:
        docs, rel = item['docs'], item['relevance_map']
        mani.fit(docs)
        q_emb = mani.model.encode([item['query']], normalize_embeddings=True)
        d_embs = mani.embeddings

        cos_scores = cosine_similarity(q_emb, d_embs).flatten()
        cos_top1 = int(np.argmax(cos_scores))

        results = mani.search_maniscope_detailed(item['query'], top_n=len(docs), coarse_multiplier=3)
        geo_top1 = results[0]['doc_id']

        cos_ok = rel.get(str(cos_top1), 0) > 0
        geo_ok = rel.get(str(int(geo_top1)), 0) > 0

        if want_success and not cos_ok and geo_ok:
            return item, cos_top1, geo_top1, d_embs, q_emb.flatten()
        if not want_success and cos_ok and not geo_ok:
            return item, cos_top1, geo_top1, d_embs, q_emb.flatten()
    return None


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cases = [
    (nfcorpus,  'Panel A: Success Case (NFCorpus)',   True),
    (treccovid, 'Panel B: Failure Case (TREC-COVID)', False),
]

for ax, (dataset, title, want_success) in zip(axes, cases):
    result = find_case(dataset, want_success, args.max_queries)
    if result is None:
        ax.set_title(f'{title}\n(no case found — try --max-queries 323)')
        ax.text(0.5, 0.5, 'Not found', ha='center', va='center', transform=ax.transAxes)
        continue

    item, cos_top1, geo_top1, d_embs, q_emb = result
    rel = item['relevance_map']

    all_embs = np.vstack([d_embs, q_emb.reshape(1, -1)])
    proj = reducer.fit_transform(all_embs)
    doc_proj, q_proj = proj[:-1], proj[-1]

    for i, (x, y) in enumerate(doc_proj):
        is_rel = rel.get(str(i), 0) > 0
        ax.scatter(x, y, c='#27ae60' if is_rel else '#bdc3c7',
                   s=90, alpha=0.85, zorder=2, edgecolors='white', linewidths=0.5)

    ax.scatter(*doc_proj[cos_top1], c='#e74c3c', s=600, marker='x', zorder=5, linewidths=2.5)
    ax.scatter(*doc_proj[geo_top1], c='#2980b9', s=600, marker='+', zorder=5, linewidths=2.5)
    ax.scatter(*q_proj, c='#2c3e50', s=160, marker='^', zorder=4)

    ax.set_title(f'{title}\nQuery: {item["query"][:55]}...', fontsize=10)
    ax.set_xlabel('UMAP dim 1')
    ax.set_ylabel('UMAP dim 2')

    handles = [
        mlines.Line2D([0],[0], marker='o', color='w', markerfacecolor='#27ae60',
                      markersize=10, label='Relevant doc'),
        mlines.Line2D([0],[0], marker='o', color='w', markerfacecolor='#5d6d7e',
                      markersize=10, label='Irrelevant doc'),
        mlines.Line2D([0],[0], marker='x', color='#e74c3c', markersize=12, markeredgewidth=2.5,
                      label='Cosine top-1', linestyle='None'),
        mlines.Line2D([0],[0], marker='+', color='#2980b9', markersize=12, markeredgewidth=2.5,
                      label='Geodesic top-1', linestyle='None'),
        mlines.Line2D([0],[0], marker='^', color='w', markerfacecolor='#2c3e50',
                      markersize=10, label='Query'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=8)

plt.suptitle('Figure 2: UMAP Projections of Top-M Candidate Embeddings\n'
             'Green = relevant  |  Slate = irrelevant  |  ✕ cosine top-1  |  + geodesic top-1  |  ▲ query',
             fontsize=11, y=1.03)
plt.tight_layout()
out_dir = Path('output/figures')
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / 'figure2_umap.pdf', dpi=150, bbox_inches='tight')
plt.savefig(out_dir / 'figure2_umap.png', dpi=150, bbox_inches='tight')
print(f'Saved: {out_dir}/figure2_umap.pdf')
print(f'Saved: {out_dir}/figure2_umap.png')
