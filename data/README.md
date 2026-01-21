# Maniscope Benchmark Datasets

This directory contains 6 BEIR benchmark datasets prepared for Maniscope evaluation.

## Dataset Overview

| Dataset | Full | Quick Test | Domain | Task |
|---------|------|------------|--------|------|
| **AorB** | 50 queries | 10 queries | Disambiguation | Semantic word sense (Python/Apple/Java/Mercury/Jaguar/Flow) |
| **SciFact** | 100 queries | 10 queries | Scientific | Scientific claim verification |
| **MS MARCO** | 200 queries | 10 queries | Web Search | General web search queries |
| **TREC-COVID** | 50 queries | 10 queries | Medical | COVID-19 research retrieval |
| **ArguAna** | 100 queries | 10 queries | Argumentation | Counter-argument retrieval |
| **FiQA** | 100 queries | 10 queries | Finance | Financial question answering |

**Total: 600 queries (full) + 60 queries (quick test)**

## File Format

Each dataset is stored as a JSON file with the following structure:

```json
{
  "corpus": {
    "doc_id_1": {
      "text": "Document text content",
      "title": "Document title (optional)"
    },
    "doc_id_2": { ... }
  },
  "queries": {
    "query_id_1": "Query text",
    "query_id_2": "Query text"
  },
  "qrels": {
    "query_id_1": {
      "doc_id_3": 1,
      "doc_id_7": 1
    }
  }
}
```

**Fields:**
- `corpus`: Dictionary of documents (doc_id → document metadata)
- `queries`: Dictionary of queries (query_id → query text)
- `qrels`: Relevance judgments (query_id → {doc_id: relevance_score})

## Dataset Descriptions

### AorB (Semantic Disambiguation)
- **50 queries** across 6 ambiguous words
- Words: Python, Apple, Java, Mercury, Jaguar, Flow
- Task: Distinguish between different meanings (e.g., Python language vs snake)
- Corpus: 24 documents (4 per word)
- Use case: Testing semantic understanding in disambiguation scenarios

### SciFact
- **100 queries** from scientific literature
- Task: Verify scientific claims using evidence from papers
- Domain: Biomedical and health sciences
- Source: BEIR benchmark, originally from SciFact dataset
- Use case: Scientific fact-checking and evidence retrieval

### MS MARCO
- **200 queries** from Bing search logs
- Task: General web search
- Domain: Mixed (questions, facts, how-to, etc.)
- Source: Microsoft MARCO dataset (BEIR subset)
- Use case: General-purpose retrieval evaluation

### TREC-COVID
- **50 queries** for COVID-19 research
- Task: Find relevant scientific papers about COVID-19
- Domain: Biomedical research
- Source: TREC-COVID track (BEIR version)
- Use case: Domain-specific scientific retrieval

### ArguAna
- **100 queries** from debate forums
- Task: Find counter-arguments to given arguments
- Domain: Argumentation and debate
- Source: ArguAna dataset (BEIR benchmark)
- Use case: Adversarial retrieval and argumentation

### FiQA
- **100 queries** from financial forums
- Task: Financial question answering
- Domain: Finance and investment
- Source: FiQA dataset (BEIR benchmark)
- Use case: Domain-specific question answering

## Quick Test Datasets

Each `-10.json` file contains:
- First 10 queries from the full dataset
- Corresponding corpus documents
- Ground truth relevance judgments

**Purpose:** Fast testing and development without running full benchmarks.

## Usage

### With Python API

```python
import json
from maniscope import ManiscopeEngine_v2o

# Load dataset
with open('data/dataset-scifact.json', 'r') as f:
    data = json.load(f)

# Extract corpus and queries
corpus = [doc['text'] for doc in data['corpus'].values()]
queries = list(data['queries'].values())

# Initialize and fit engine
engine = ManiscopeEngine_v2o(k=5, alpha=0.3)
engine.fit(corpus)

# Run benchmark
for query in queries[:5]:
    results = engine.search(query, top_n=10)
    print(f"\nQuery: {query}")
    for doc, score, idx in results[:3]:
        print(f"  [{score:.3f}] {doc[:100]}...")
```

### With Streamlit App

```bash
# Launch the evaluation app
python run_app.py

# Navigate to "Data Manager" page
# Upload any dataset-*.json file
# View statistics and preview
# Switch to "Benchmark" page to evaluate
```

## Dataset Preparation

These datasets were prepared from BEIR using custom scripts. The preparation process:

1. Download from BEIR/MTEB
2. Extract corpus, queries, and qrels
3. Format as unified JSON structure
4. Create 10-query quick test subsets
5. Validate data integrity

For details on dataset preparation, see the source repository's `data/prep_mteb_dataset.py` script.

## Citation

If you use these datasets, please cite the original sources:

**BEIR Benchmark:**
```bibtex
@inproceedings{thakur2021beir,
  title={BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
  author={Thakur, Nandan and Reimers, Nils and R{\"u}ckl{\'e}, Andreas and Srivastava, Abhishek and Gurevych, Iryna},
  booktitle={NeurIPS},
  year={2021}
}
```

**Individual datasets:** See BEIR documentation for specific citations.

## License

These datasets are used for research purposes under their original licenses. See BEIR documentation for details.
