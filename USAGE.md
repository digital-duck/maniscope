# Maniscope Usage Guide for Reviewers

This guide helps reviewers and researchers reproduce the results from the arXiv paper.

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/digital-duck/maniscope.git
cd maniscope

# Install in development mode
pip install -e .
```

### 2. Launch Evaluation App

```bash
python run_app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Run Quick Test

1. Navigate to **"⚡ Optimization"** page (sidebar)
2. Enable **"🧪 Test Mode"** (sidebar)
3. Click **"🚀 Run Benchmark"**
4. View results in ~10 seconds

This runs a mock benchmark to verify installation.

## Reproducing Paper Results

### Full Benchmark Suite (30-60 minutes)

#### Option 1: Via Streamlit App (Recommended)

1. **Launch app:**
   ```bash
   python run_app.py
   ```

2. **Navigate to "🚀 Batch Benchmark" page**

3. **Configure:**
   - Select all 6 datasets (AorB, SciFact, MS MARCO, TREC-COVID, ArguAna, FiQA)
   - Select Maniscope v2o
   - Configure parameters:
     - k = 5
     - α = 0.3 (or test range: 0.1, 0.3, 0.5, 0.7)
   - Disable "Test Mode"

4. **Run benchmark:**
   - Click "Start Batch Benchmark"
   - Monitor progress (real-time metrics displayed)
   - Results auto-saved to `output/` directory

5. **View results:**
   - Navigate to "📈 Analytics" page
   - Compare MRR, NDCG@K, MAP metrics
   - Export to Excel/CSV for analysis

#### Option 2: Via Python API

```python
import json
from maniscope import ManiscopeEngine_v2o

# Load dataset
with open('data/dataset-scifact.json', 'r') as f:
    data = json.load(f)

# Initialize engine
engine = ManiscopeEngine_v2o(
    k=5,
    alpha=0.3,
    verbose=True,
    use_cache=True
)

# Fit on corpus
corpus_docs = [doc['text'] for doc in data['corpus'].values()]
engine.fit(corpus_docs)

# Evaluate queries
results = []
for qid, query in data['queries'].items():
    search_results = engine.search(query, top_n=10)
    results.append({
        'query_id': qid,
        'query': query,
        'results': search_results
    })

# Calculate metrics
from ui.utils.metrics import calculate_metrics
metrics = calculate_metrics(results, data['qrels'])
print(f"MRR: {metrics['MRR']:.4f}")
print(f"NDCG@10: {metrics['NDCG@10']:.4f}")
```

## Testing Optimization Versions

Compare v0 (baseline) vs v2o (optimized):

1. **Via App:**
   - Go to "⚡ Optimization" page
   - Select versions: v0, v1, v2, v2o, v3
   - Click "Run Benchmark"
   - View speedup comparison

2. **Via Python:**
   ```python
   from maniscope import ManiscopeEngine, ManiscopeEngine_v2o, compare_maniscope_performance

   # Load and fit both engines
   engine_v0 = ManiscopeEngine(k=5, alpha=0.3)
   engine_v2o = ManiscopeEngine_v2o(k=5, alpha=0.3)

   engine_v0.fit(documents)
   engine_v2o.fit(documents)

   # Compare performance
   perf = compare_maniscope_performance(
       engine_v0, engine_v2o,
       query="test query",
       num_runs=10
   )
   print(f"Speedup: {perf['speedup']:.1f}x")
   ```

## Expected Results

### Performance Benchmarks (v2o)

| Dataset | MRR | NDCG@10 | Latency | Speedup |
|---------|-----|---------|---------|---------|
| AorB | 1.0000 | 1.0000 | ~0.5ms | 200-230× |
| SciFact | 0.9821 | 0.9850 | ~0.4ms | 230-235× |
| MS MARCO | 1.0000 | 1.0000 | ~0.6ms | 220-229× |
| TREC-COVID | 1.0000 | 1.0000 | ~0.4ms | 220-226× |
| ArguAna | 0.9912 | 0.9900 | ~0.5ms | 200-220× |
| FiQA | 0.9707 | 0.9750 | ~0.5ms | 200-220× |

**Note:** First run will be slower (cold cache). Subsequent runs leverage persistent cache.

### Optimization Comparison

| Version | Avg Latency | Speedup |
|---------|-------------|---------|
| v0 | 115ms | 1.0× |
| v1 | 40ms | 3.0× |
| v2 | 22ms | 5.0× |
| v3 | 10-50ms | 2-10× (cache-dependent) |
| v2o | 0.4-0.6ms | **20-235×** |

## Datasets

All datasets are included in `data/` directory:

- **Full datasets:** `dataset-{name}.json`
- **Quick tests:** `dataset-{name}-10.json` (10 queries)

| Dataset | Queries | Domain |
|---------|---------|--------|
| AorB | 50 | Disambiguation |
| SciFact | 100 | Scientific |
| MS MARCO | 200 | Web search |
| TREC-COVID | 50 | Medical |
| ArguAna | 100 | Argumentation |
| FiQA | 100 | Finance |

See `data/README.md` for detailed dataset documentation.

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'maniscope'`:

```bash
# Reinstall in development mode
pip install -e .
```

### GPU Not Detected

v2o auto-detects GPU. To verify:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

If GPU is not available, v2o will fall back to CPU (still fast with caching).

### Streamlit Port Already in Use

If port 8501 is busy:

```bash
streamlit run ui/Maniscope.py --server.port 8502
```

### Cache Issues

Clear cache if experiencing issues:

```bash
rm -rf ~/projects/embedding_cache/maniscope
```

## Advanced Usage

### Custom Datasets

Prepare your own dataset in BEIR format:

```json
{
  "corpus": {
    "doc1": {"text": "...", "title": "..."}
  },
  "queries": {
    "q1": "query text"
  },
  "qrels": {
    "q1": {"doc1": 1}
  }
}
```

Save as `data/dataset-custom.json` and use in app.

### Parameter Tuning

Test different k and α values:

1. Go to "⚙️ Configuration" page
2. Set parameter ranges:
   - k: 3-15 (default: 5)
   - α: 0.0-1.0 (default: 0.3)
3. Run grid search
4. Analyze results in "📈 Analytics"

### Comparing with Other Rerankers

The app supports comparison with:
- **BGE-M3** (cross-encoder)
- **Jina Reranker v2**
- **LLM Reranker** (via Ollama/OpenRouter)

1. Go to "🔬 Eval ReRanker" page
2. Select multiple rerankers
3. Run benchmark
4. Compare metrics side-by-side

## Support

For questions or issues:
- GitHub Issues: [github.com/digital-duck/maniscope/issues]
- Email: wen.gong.research@gmail.com

## Citation

```bibtex
@article{gong2026maniscope,
  title={Optimizing RAG Reranker via Geodesic Distances on k-NN Manifolds},
  author={Gong, Wen G.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
