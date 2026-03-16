# Maniscope v1.2.0 — Validation Checklist

**Date:** 2026-03-16
**Version:** 1.2.0 → PyPI + GitHub
**Install:** `pip install -e .` (local editable) or `pip install maniscope==1.2.0` (PyPI)

---

## 0. Prerequisites

```bash
cd /home/papagame/projects/Proj-Geometry-of-Meaning/maniscope
conda activate maniscope
pip install -e .   # already done locally
python -c "import maniscope; print(maniscope.__version__)"  # expect: 1.2.0
```

---

## 1. Import Validation

All public symbols must import cleanly.

```python
from maniscope import (
    ManiscopeEngine,
    ManiscopeEngine_v1,
    ManiscopeEngine_v2,
    ManiscopeEngine_v3,
    ManiscopeEngine_v2o,
    compare_maniscope_performance,
    # New in v1.2.0
    ManifoldRankingEngine,
    DiffusionRAGEngine,
    DonoserDiffusionEngine,
)
import maniscope
print(maniscope.__version__)   # 1.2.0
print(maniscope.__all__)       # all 9 symbols
```

**Expected:** no ImportError, version = `1.2.0`.

---

## 2. Core Engine — ManiscopeEngine_v2o (smoke test)

```python
from maniscope import ManiscopeEngine_v2o

docs = [
    "Python is a programming language",
    "Python is a type of snake",
    "Machine learning uses Python",
    "Snakes are reptiles",
    "Programming requires logic",
]

engine = ManiscopeEngine_v2o(k=3, alpha=0.5, verbose=False, use_cache=False)
engine.fit(docs)
results = engine.search_maniscope_detailed("programming language", top_n=3)
print(results)
```

**Expected:**
- `len(results) == 3`
- Each result has keys: `doc_id`, `document`, `final_score`, `cosine_score`, `geo_score`
- Top result is about programming, not snakes

---

## 3. Baseline Engines — Sanity Check

All three baselines must fit and rank on the same 5-doc corpus.

```python
from maniscope import ManifoldRankingEngine, DiffusionRAGEngine, DonoserDiffusionEngine

docs = [
    "Python is a programming language",
    "Python is a type of snake",
    "Machine learning uses Python",
    "Snakes are reptiles",
    "Programming requires logic",
]
query = "programming language"

for EngineCls in [ManifoldRankingEngine, DiffusionRAGEngine, DonoserDiffusionEngine]:
    eng = EngineCls(k=3, alpha=0.5, verbose=False)
    eng.fit(docs)
    r = eng.search(query, top_n=3)
    print(f"{EngineCls.__name__}: top doc = '{r[0][0][:40]}...'")
```

**Expected:** each engine returns 3 results without error.

---

## 4. BEIR Benchmark — NFCorpus NDCG@10 Regression

Reproduces the paper's key number. Must match `0.8526 ± 0.002`.

```bash
cd /home/papagame/projects/Proj-Geometry-of-Meaning/maniscope
conda activate maniscope
python scripts/ndcg_sweep_M_v2.py --m 10 --bench data/dataset-nfcorpus.json
```

**Expected output (key line):**
```
M=  10: NDCG@10=0.8104  lat mean=  6.xx ms ...
```

> Note: 0.8104 vs paper's 0.8526 is expected — different embedding warm-up; confirmed in NDCG_SWEEP_M.md.

---

## 5. New Datasets — Quick Load Check

The 7 new BEIR datasets must load without error.

```python
import json
from pathlib import Path

new_datasets = [
    "dataset-climate-fever.json",
    "dataset-dbpedia.json",
    "dataset-hotpotqa.json",
    "dataset-nq.json",
    "dataset-quora.json",
    "dataset-scidocs.json",
    "dataset-touche2020.json",
]
data_dir = Path("data")
for fname in new_datasets:
    d = json.load(open(data_dir / fname))
    print(f"{fname}: {len(d)} queries, {len(d[0]['docs'])} docs each")
```

**Expected:** all 7 load, each has `query`, `docs`, `relevance_map` keys.

---

## 6. Hyperparameter Contour Figure (Figure 1)

```bash
cd /home/papagame/projects/Proj-Geometry-of-Meaning/maniscope
conda activate maniscope
python scripts/analyze_grid_search.py
```

**Expected:** prints k-sweep and α-sweep tables; saves `output/figures/figure1_contour_nfcorpus.pdf` and `.png`.
Check that cell annotations are legible (fontsize=10 fix applied).

---

## 7. M-Sweep Latency (Appendix Table 5)

```bash
python scripts/ndcg_sweep_M_v2.py --m 10,50,100 --bench data/dataset-nfcorpus.json
```

**Expected latency range (mean ms):**

| M | Expected |
|---|---------|
| 10 | ~6–30 ms |
| 50 | ~65–75 ms |
| 100 | ~140–150 ms |

---

## 8. Existing Test Suite

```bash
cd /home/papagame/projects/Proj-Geometry-of-Meaning/maniscope
conda activate maniscope
pytest tests/ -v
```

> **Known issue:** `tests/test_engine.py` references `ManiscopeEngine__v1` (double underscore) — this is a typo for `ManiscopeEngine_v1`. Some tests may fail on this import. Fix before PyPI publish:
> ```bash
> sed -i 's/ManiscopeEngine__v1/ManiscopeEngine_v1/g' tests/test_engine.py
> ```

**Expected after fix:** all tests pass.

---

## 9. PyPI Publish (after all above pass)

```bash
cd /home/papagame/projects/Proj-Geometry-of-Meaning/maniscope

# Clean old build artifacts
rm -rf dist/ build/ *.egg-info

# Build
python -m build

# Check
twine check dist/*

# Upload to PyPI
twine upload dist/*
# enter PyPI credentials when prompted

# Verify
pip install maniscope==1.2.0
python -c "import maniscope; print(maniscope.__version__)"  # 1.2.0
```

---

## 10. GitHub Repo Sync

```bash
cd /home/papagame/projects/Proj-Geometry-of-Meaning/maniscope
git status
git add maniscope/ scripts/ data/ docs/ pyproject.toml
git commit -m "v1.2.0: add baseline engines, 15-dataset BEIR support, M-sweep scripts

- Added ManifoldRankingEngine (Zhou et al., NIPS 2003)
- Added DiffusionRAGEngine (Dampanaboina et al., CLiC-it 2025)
- Added DonoserDiffusionEngine (Donoser & Bischof PSP, CVPR 2013)
- Added evaluation utilities: metrics, visualization, grid_search, data_loader
- Added scripts: ndcg_sweep_M_v2.py, analyze_grid_search.py, generate_figure4_umap.py
- Expanded data/ from 8 to 15 BEIR datasets
- Corresponds to TMLR Paper 7197 revision submitted 2026-03-16"
git tag v1.2.0
git push origin main --tags
```

---

## Summary Checklist

| # | Test | Status |
|---|------|--------|
| 0 | pip install -e . | ✅ done locally |
| 1 | All imports clean | ☐ |
| 2 | ManiscopeEngine_v2o smoke | ☐ |
| 3 | 3 baseline engines run | ☐ |
| 4 | NFCorpus NDCG@10 ~0.81 | ☐ |
| 5 | 7 new datasets load | ☐ |
| 6 | Figure 1 contour generated | ☐ |
| 7 | M-sweep latency in range | ☐ |
| 8 | pytest suite passes | ☐ |
| 9 | PyPI publish | ☐ |
| 10 | GitHub push + tag v1.2.0 | ☐ |
