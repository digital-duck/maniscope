# Custom PDF Dataset Import for RAG Evaluation

## Overview

The Custom PDF Dataset feature enables researchers to import PDF documents (e.g., arXiv papers) directly into Maniscope for interactive RAG evaluation. This allows you to study research papers, test retrieval quality, and generate contextual answers using state-of-the-art language models.

**Key Benefits:**
- Study arXiv papers interactively by asking custom questions
- Test retrieval and reranking on domain-specific documents
- Evaluate RAG pipelines without manual dataset creation
- No ground truth labels needed (exploratory mode)

## Motivation

Traditional RAG benchmarks (MTEB, BEIR) provide standardized evaluation, but researchers often need to:
1. **Understand new research papers** by asking specific questions
2. **Test retrieval quality** on domain-specific documents
3. **Profile RAG performance** on real-world content (not synthetic)
4. **Explore document structure** (sections, figures, tables) through search

Custom PDF datasets enable these use cases without requiring manual annotation or synthetic query generation.

## Architecture

```
PDF Upload (arXiv paper)
    ↓
Docling Conversion → Markdown + Metadata
    ↓
Section-based Chunking (with 50-100 word overlap)
    ↓
Figure/Table Caption Extraction (Tier 1)
    ↓
Dataset JSON Creation (single item, empty query/relevance_map)
    ↓
Save to data/custom/{name}.json
    ↓
Load into Session State
    ↓
Use in Eval ReRanker (Custom Query mode only)
```

## Usage Workflow

### 1. Navigate to Data Manager

Go to **📁 Data Manager** page → **📄 Custom Dataset** tab

### 2. Upload PDF

- Click "Select PDF File" and upload your document
- Academic papers (arXiv, conference PDFs) work best
- File size limit: ~10MB recommended

### 3. Configure Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Dataset Name** | `custom_paper` | Name for the dataset (will be sanitized to valid filename) |
| **Extract Figure/Table Captions** | ✓ Enabled | Include figure/table captions as searchable documents (Tier 1) |
| **Max Chunk Size** | 1000 words | Maximum words per chunk (200-2000) |
| **Overlap Between Chunks** | 50 words | Word overlap for better retrieval across boundaries (0-200) |

**Recommended Settings for Academic Papers:**
- Max Chunk Size: 1000 words (captures full paragraphs/subsections)
- Overlap: 50-100 words (prevents information loss at boundaries)
- Extract Captions: Enabled (makes figures/tables discoverable)

### 4. Process PDF

1. Click **🔄 Process PDF**
2. Wait for 4-step pipeline:
   - **Step 1:** Converting PDF to markdown (Docling)
   - **Step 2:** Chunking by sections (headings-based)
   - **Step 3:** Extracting figures and tables (captions only)
   - **Step 4:** Creating dataset (Maniscope format)

3. Review statistics:
   - Total Docs (chunks + figures/tables)
   - Text Chunks
   - Figures/Tables
   - Pages

4. Preview:
   - **Text Chunks:** First 3 chunks with headings
   - **Figures/Tables:** First 5 captions

### 5. Load Dataset

Click **✅ Load This Dataset** to activate in session state

The dataset is now available in:
- **🔬 Eval ReRanker** (Custom Query mode only)
- **ℹ️ Current Dataset** tab (statistics and preview)

### 6. Evaluate in Eval ReRanker

1. Navigate to **🔬 Eval ReRanker**
2. Notice: **"Custom datasets use Custom Query mode (no ground truth available)"**
3. Enter your custom query (e.g., "What is the main contribution of this paper?")
4. Select document source:
   - **`__ALL_DOCS__`:** Use all chunks + figures (recommended for paper exploration)
   - **Specific query:** Use docs from a specific chunk
5. Run reranker → Generate RAG answer with top-tier LLM

## Figure/Table Handling (Tier 1)

### Current Implementation (Tier 1)
**Caption extraction only** - no vision models involved.

**Format:**
```
[Figure 1]: Architecture diagram showing the pipeline from PDF upload to RAG evaluation...
[Table 2]: Performance comparison across MTEB datasets (MRR, NDCG@10, MAP)
```

**Stored Metadata:**
```json
{
  "type": "figure",
  "caption": "Architecture diagram showing...",
  "page_number": 3,
  "figure_id": "fig_1"
}
```

**Benefits:**
- Makes figures/tables **discoverable** through text search
- Enables **referencing** figures in RAG answers (e.g., "As shown in Figure 1...")
- Works without GPU/vision models (fast, cost-free)

**Limitations:**
- Cannot search figure **visual content** (only captions)
- Cannot answer questions about diagram details (unless in caption)

### Future Enhancement (Tier 2+)
Planned features:
- Extract figure images as separate files
- Index with vision models (CLIP, multimodal embeddings)
- Enable visual similarity search
- Support "What does Figure 3 show?" queries with image analysis

## Eval ReRanker Integration

### Custom Query Mode (Forced for Custom Datasets)

When you load a custom dataset, Eval ReRanker **automatically hides** "From Dataset" mode because:
- Custom datasets have **empty queries** (`query: ""`)
- Custom datasets have **empty relevance_map** (`relevance_map: {}`)
- No ground truth means **no metrics** (MRR, NDCG, MAP)

**What You CAN Do:**
- ✅ Enter custom queries (your research questions)
- ✅ Test baseline and Maniscope reranking
- ✅ Generate RAG answers with LLM
- ✅ Use `__ALL_DOCS__` mode for full paper context
- ✅ View latency breakdown (retrieval, reranking, LLM)
- ✅ Download reranked results

**What You CANNOT Do:**
- ❌ Compute metrics (no ground truth labels)
- ❌ Use "From Dataset" query mode (no queries in dataset)
- ❌ Benchmark quantitatively (use MTEB datasets for that)

### Document Source: `__ALL_DOCS__` Mode

**Recommended for Custom Datasets**

When you select `__ALL_DOCS__`, the system:
1. Concatenates all chunks + figures/tables into a single document pool
2. Runs retrieval on the full paper
3. Returns top-K most relevant sections
4. Uses reranking to refine order

**Perfect for:**
- "What is the main contribution?" → searches entire paper
- "How does the method work?" → finds relevant sections across introduction, methods, results
- "What datasets were used?" → locates experimental setup sections

**Example Query Flow:**
```
User Query: "What evaluation metrics were used?"
    ↓
Retrieval (Top-10 from all chunks + figures)
    ↓
Reranking (Maniscope refines order)
    ↓
RAG Answer Generation (Claude 4.5 with top-3 context)
    ↓
Answer: "The paper evaluated using MRR, NDCG@10, and MAP on BEIR datasets..."
```

## Top-Tier LLM Models for Research

### Available Models (Added to Config)

**Top-Tier Models (for studying arXiv papers):**
- `anthropic/claude-opus-4-5` - Latest Claude Opus 4.5
- `anthropic/claude-sonnet-4-5` - Latest Claude Sonnet 4.5 (recommended)
- `anthropic/claude-opus-4` - Claude Opus 4
- `google/gemini-2.5-pro-latest` - Gemini 2.5 Pro
- `google/gemini-3-pro-latest` - Gemini 3 Pro (future)
- `openai/gpt-5` - GPT-5 (future)
- `qwen/qwen-3.5-turbo` - Qwen 3.5 Turbo

**Free Models (for testing):**
- `meta-llama/llama-3.2-3b-instruct:free` - Llama 3.2 3B
- `qwen/qwen-2-7b-instruct:free` - Qwen 2 7B
- `microsoft/phi-3-mini-128k-instruct:free` - Phi-3 Mini (128K context)

### When to Use Top-Tier Models

**Recommended for:**
- Complex research questions requiring deep understanding
- Multi-hop reasoning across paper sections
- Technical explanations (methods, math, algorithms)
- Long-context tasks (entire paper analysis)

**Example Use Case:**
```
Query: "Compare the proposed method to prior work and explain why it's better"
Model: anthropic/claude-sonnet-4-5
Context: Top-3 chunks (Introduction, Related Work, Conclusion)
Result: Detailed comparison with citations to specific sections
```

### When to Use Free Models

**Recommended for:**
- Quick tests during development
- Simple factual queries ("What year was this published?")
- Cost-free experimentation
- Prototyping workflows before production

## Dataset Format Example

### Saved File: `data/custom/attention_is_all_you_need.json`

```json
[
  {
    "query": "",
    "docs": [
      "## Introduction\n\nThe dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
      "## Background\n\nThe goal of reducing sequential computation also forms the foundation of the Extended Neural GPU...",
      "[Figure 1]: The Transformer model architecture showing multi-head attention layers and feed-forward networks",
      "[Table 1]: Performance comparison on WMT 2014 English-German and English-French translation tasks",
      "## Model Architecture\n\nMost competitive neural sequence transduction models have an encoder-decoder structure..."
    ],
    "relevance_map": {},
    "query_id": "custom_0",
    "num_docs": 5,
    "metadata": {
      "source": "pdf_import",
      "dataset_name": "attention_is_all_you_need",
      "pdf_filename": "1706.03762.pdf",
      "has_figures": true,
      "num_pages": 15,
      "num_chunks": 3,
      "num_figures": 1,
      "num_tables": 1,
      "chunking_strategy": "section_based",
      "max_chunk_size": 1000,
      "overlap_words": 50,
      "created_at": "2026-01-25T12:00:00"
    }
  }
]
```

### Schema Validation

All custom datasets are validated against Maniscope schema:
- ✅ `query`: string (empty for custom datasets)
- ✅ `docs`: list of strings (non-empty)
- ✅ `relevance_map`: dict (empty for custom datasets)
- ✅ `query_id`: string (defaults to "custom_0")
- ✅ `metadata`: dict (optional, recommended)

## Verification Checklist

### PDF Processing
- [ ] Upload PDF file successfully
- [ ] Docling converts to markdown without errors
- [ ] Sections are correctly identified by headings (#, ##, ###)
- [ ] Chunks respect max size limit
- [ ] 50-100 word overlap is applied between chunks
- [ ] Figure captions extracted with `[Figure X]:` prefix
- [ ] Table captions extracted with `[Table X]:` prefix
- [ ] Metadata includes page numbers and IDs

### Dataset Creation
- [ ] Dataset has single item with empty query (`query: ""`)
- [ ] All chunks and figures in `docs` array
- [ ] `relevance_map` is empty dict (`{}`)
- [ ] Validates with `validate_dataset_schema()`
- [ ] Saves to `data/custom/{name}.json`
- [ ] File is valid JSON (can be reloaded)

### UI Integration
- [ ] Tab 4 "Custom Dataset" appears in Data Manager
- [ ] File uploader accepts PDF only
- [ ] Settings are configurable (name, figures, chunk size, overlap)
- [ ] Progress bar shows during processing (4 steps)
- [ ] Preview shows chunks and figures correctly
- [ ] "Load This Dataset" updates session state
- [ ] Page reruns and shows dataset in "Current Dataset" tab

### Eval ReRanker Compatibility
- [ ] Custom dataset loads in Eval ReRanker
- [ ] **UI auto-hides "From Dataset" mode for custom datasets**
- [ ] **Only "Custom Query" mode shown with info message**
- [ ] Custom query text input works
- [ ] Baseline method works with custom docs
- [ ] Maniscope reranker works with custom docs
- [ ] `__ALL_DOCS__` mode concatenates all chunks + figures
- [ ] RAG answer generation uses correct context
- [ ] Figure captions (Tier 1) appear in retrieved docs when relevant
- [ ] Latency breakdown shows retrieval/reranking/LLM times

### Free LLM Models
- [ ] Free models appear in LLM model dropdown
- [ ] Free models work for RAG answer generation
- [ ] No API errors with `:free` suffix models

## Future Enhancements

### Tier 2: Vision Model Integration
- Extract figure images as PNG/JPG files
- Index with CLIP or multimodal embeddings
- Enable visual similarity search
- Support "Show me similar figures" queries
- Answer "What does Figure 3 show?" with image analysis

### Tier 3: Multi-PDF Datasets
- Batch import multiple PDFs
- Create cross-document retrieval datasets
- Support "Compare these two papers" queries
- Build literature review datasets

### Tier 4: Synthetic Query Generation
- Auto-generate queries from section headings
- Create weak supervision labels (section → query → relevant chunks)
- Enable quantitative evaluation (MRR, NDCG)
- Transform exploratory datasets into benchmarks

### Tier 5: Custom Relevance Annotation
- Manual labeling UI for ground truth
- Mark relevant/irrelevant chunks per query
- Compute metrics after annotation
- Export to MTEB format

### Tier 6: Advanced Chunking Strategies
- Semantic chunking (sentence embeddings + clustering)
- Sliding window with dynamic overlap
- Entity-aware chunking (preserve named entities)
- Citation-aware chunking (keep references intact)

### Tier 7: OCR Support
- Detect scanned PDFs (low text extraction)
- Auto-trigger OCR preprocessing
- Support image-only PDFs
- Integrate with Tesseract/PaddleOCR

## Example Use Case: Studying an arXiv Paper

### Scenario
You want to deeply understand the "Attention Is All You Need" paper (Transformer architecture).

### Workflow

**1. Upload PDF**
- Download from arXiv: `1706.03762.pdf`
- Upload to Data Manager → Custom Dataset tab

**2. Process with Settings**
- Dataset Name: `attention_is_all_you_need`
- Extract Figures: ✓ Enabled
- Max Chunk Size: 1000 words
- Overlap: 100 words (high overlap for technical paper)

**3. Load Dataset**
- Preview shows sections: Introduction, Background, Model Architecture, etc.
- Figures extracted: Architecture diagram, attention visualization
- Load into session

**4. Ask Questions in Eval ReRanker**

**Query 1:** "What is the main contribution of this paper?"
- Document Source: `__ALL_DOCS__`
- Reranker: Maniscope_v2o
- LLM: `anthropic/claude-sonnet-4-5`
- Result: "The paper introduces the Transformer, a novel architecture based solely on attention mechanisms..."

**Query 2:** "How does multi-head attention work?"
- Document Source: `__ALL_DOCS__`
- Retrieved chunks: Model Architecture section
- Result: Detailed explanation with mathematical formulation

**Query 3:** "What datasets were used for evaluation?"
- Document Source: `__ALL_DOCS__`
- Retrieved chunks: Experiments section, Table 1 caption
- Result: "WMT 2014 English-German and English-French translation tasks..."

**5. Profile Performance**
- Latency Breakdown:
  - Retrieval: 15ms (embedding query)
  - Reranking: 2ms (Maniscope v2o)
  - LLM: 1200ms (Claude Sonnet 4.5)
  - Total: 1217ms

**6. Iterate**
- Try different rerankers (baseline vs Maniscope)
- Test free models vs top-tier models
- Experiment with `__ALL_DOCS__` vs specific sections

### Expected Outcomes
- ✅ Deep understanding of Transformer architecture
- ✅ Validated retrieval quality on technical content
- ✅ Profiled RAG latency for production planning
- ✅ Discovered optimal reranker for research papers

## Troubleshooting

### Error: "Docling not installed"

**Solution:**
```bash
pip install docling>=1.0.0
```

### Error: "PDF conversion failed"

**Possible Causes:**
1. **Password-protected PDF**
   - Solution: Unlock PDF first using PDF tools
   - Command: `qpdf --decrypt input.pdf output.pdf`

2. **Corrupted PDF**
   - Solution: Re-download from source (arXiv, conference site)
   - Check file integrity with `file input.pdf`

3. **Scanned PDF (image-only)**
   - Solution: Use OCR preprocessing (future Tier 7 feature)
   - Current workaround: Skip these PDFs or use text-based versions

4. **Memory error (large file)**
   - Solution: Reduce file size or split into parts
   - Recommended limit: < 10MB per PDF

### Error: "No sections found, using fixed-size chunking"

**Cause:** PDF has no markdown headings (flat structure)

**Solution:**
- This is expected for some PDFs
- Fallback chunking still works (fixed-size with overlap)
- Result quality may be lower (no semantic section boundaries)

### Warning: "Large file (>10MB) may take longer to process"

**Recommendation:**
- Processing time scales with file size
- For 50+ page PDFs, expect 30-60 seconds
- Consider reducing max chunk size to speed up

### Error: "Dataset validation failed: docs list is empty"

**Cause:** PDF had no extractable text

**Solution:**
- Check if PDF is scanned (image-only)
- Verify PDF opens correctly in reader
- Try re-downloading from source

### No Metrics Shown in Eval ReRanker

**Expected Behavior:** Custom datasets have no ground truth, so metrics are not computed.

**Solution:**
- This is by design (exploratory mode)
- Use MTEB datasets for quantitative evaluation
- Future: Add manual annotation UI (Tier 5)

## Related Documentation

- **MTEB Dataset Loading:** See Tab 1 in Data Manager
- **Synthetic Dataset Generation:** See Tab 2 in Data Manager
- **Benchmark Mode:** For quantitative evaluation with metrics
- **Grid Search:** For hyperparameter optimization on MTEB datasets

## Summary

The Custom PDF Dataset feature transforms Maniscope into an **interactive research assistant** for studying academic papers:

✅ **Upload PDF** → **Process** → **Ask Questions** → **Get Answers**

Perfect for:
- Researchers studying arXiv papers
- Students learning from textbooks/papers
- Engineers testing RAG on domain documents
- Data scientists profiling retrieval quality

**Key Advantages:**
- No manual annotation required
- Works with any PDF (papers, reports, books)
- Leverages state-of-the-art LLMs (Claude 4.5, Gemini 2.5)
- Free models available for testing
- Fast processing (section-based chunking)
- Preserves document structure (headings, figures, tables)

**Next Steps:**
1. Upload your first arXiv paper
2. Try `__ALL_DOCS__` mode with custom queries
3. Compare baseline vs Maniscope reranking
4. Test top-tier LLMs vs free models
5. Profile latency for your use case

Happy exploring! 🚀
