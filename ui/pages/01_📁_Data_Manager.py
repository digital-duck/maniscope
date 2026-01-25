"""
Data Manager Page

Supports:
1. Loading MTEB datasets from prep_*.py tools
2. Generating synthetic datasets via OpenRouter
3. Viewing current dataset statistics
"""

import os
import streamlit as st
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import (
    load_mteb_dataset,
    load_synthetic_dataset,
    get_dataset_statistics,
    preview_dataset,
    DatasetValidationError
)

st.set_page_config(page_title="Data Manager", layout="wide", page_icon="📁")
st.header("📁 Dataset Management")

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None
if 'dataset_name' not in st.session_state:
    st.session_state['dataset_name'] = None
if 'dataset_source' not in st.session_state:
    st.session_state['dataset_source'] = None

# Auto-load default dataset for testing (SciFact)
DEFAULT_DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "dataset-scifact.json"

if st.session_state['dataset'] is None and DEFAULT_DATASET_PATH.exists():
    try:
        with open(DEFAULT_DATASET_PATH, 'r') as f:
            data = load_mteb_dataset(f, validate=True)
            st.session_state['dataset'] = data
            st.session_state['dataset_name'] = DEFAULT_DATASET_PATH.name
            st.session_state['dataset_path'] = str(DEFAULT_DATASET_PATH.resolve())
            st.session_state['dataset_source'] = 'mteb'
            st.info(f"ℹ️ Auto-loaded default dataset: **{DEFAULT_DATASET_PATH.resolve()}** ({len(data)} queries)")
    except Exception as e:
        st.warning(f"⚠️ Could not auto-load default dataset: {str(e)}")

# Create tabs
tab1, tab2, tab4, tab3 = st.tabs([
    "📁 Load MTEB Dataset",
    "🤖 Generate Synthetic",
    "📄 Custom Dataset",
    "ℹ️ Current Dataset"
])

# ============================================================================
# TAB 1: Load MTEB Dataset
# ============================================================================
with tab1:

    c_exist, _, c_new = st.columns([8,1,8])

    with c_exist:
        # Existing datasets dropdown
        st.markdown("#### 📂 Select Existing Dataset")

        data_dir = Path(__file__).parent.parent.parent / "data"
        dataset_files = sorted(data_dir.glob("dataset-*.json"))

        if dataset_files:
            # Create options list with full paths
            dataset_options = {f.name: str(f.resolve()) for f in dataset_files}

            # Set default to dataset-scifact-10.json if it exists, else first file
            default_file = "dataset-scifact-10.json"
            default_index = 0
            if default_file in dataset_options:
                default_index = list(dataset_options.keys()).index(default_file)

            selected_dataset_name = st.selectbox(
                "Available Datasets",
                options=sorted(list(dataset_options.keys())),
                index=default_index,
                help="Select a dataset from the data directory"
            )

            selected_dataset_path = dataset_options[selected_dataset_name]

            # Show file info
            try:
                file_size = Path(selected_dataset_path).stat().st_size / 1024  # KB
                st.caption(f"📁 Path: `{selected_dataset_path}`, 💾 Size: {file_size:.1f} KB")
            except:
                pass

            col1, _ = st.columns([3, 3])
            with col1:
                if st.button("📥 Load Data", type="primary"):
                    try:
                        with st.spinner(f"Loading {selected_dataset_name}..."):
                            with open(selected_dataset_path, 'r') as f:
                                data = load_mteb_dataset(f, validate=True)

                            stats = get_dataset_statistics(data)

                            # Load into session
                            st.session_state['dataset'] = data
                            st.session_state['dataset_name'] = selected_dataset_name
                            st.session_state['dataset_path'] = selected_dataset_path
                            st.session_state['dataset_source'] = 'mteb'

                            st.success(f"✅ Loaded {stats['num_queries']} queries from {selected_dataset_name}")
                            st.rerun()

                    except DatasetValidationError as e:
                        st.error(f"❌ Dataset validation failed:\n\n{str(e)}")
                    except Exception as e:
                        st.error(f"❌ Error loading dataset: {str(e)}")

        else:
            st.warning(f"⚠️ No datasets found in `{data_dir}`")
            st.info("Create datasets using `download_*.py` scripts in the data directory")

    with c_new:
        st.markdown("#### 📤 Upload New Dataset")

        uploaded_file = st.file_uploader(
            "Upload Dataset (JSON)",
            type=['json'],
            help="Upload a JSON file from prep_scifact.py or prep_mteb_dataset.py"
        )

        if uploaded_file is not None:
            try:
                with st.spinner("Loading and validating dataset..."):
                    # Load dataset
                    data = load_mteb_dataset(uploaded_file, validate=True)

                    # Get statistics
                    stats = get_dataset_statistics(data)

                    st.success(f"✅ Loaded {stats['num_queries']} queries successfully!")

                    # Display statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Queries", stats['num_queries'])
                    with col2:
                        st.metric("Docs/Query (avg)", f"{stats['num_docs_per_query']:.1f}")
                    with col3:
                        st.metric("Gold/Query (avg)", f"{stats['num_gold_per_query']:.1f}")
                    with col4:
                        st.metric("Total Docs", stats['total_docs'])

                    # Show preview
                    st.markdown("### Preview")
                    st.text(preview_dataset(data, num_samples=3))

                    # Confirm button
                    if st.button("✅ Use This Dataset", type="primary"):
                        st.session_state['dataset'] = data
                        st.session_state['dataset_name'] = uploaded_file.name
                        st.session_state['dataset_path'] = f"uploaded: {uploaded_file.name}"
                        st.session_state['dataset_source'] = 'mteb'
                        st.success(f"Dataset '{uploaded_file.name}' loaded into session!")
                        st.rerun()

            except DatasetValidationError as e:
                st.error(f"❌ Dataset validation failed:\n\n{str(e)}")
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")

# ============================================================================
# TAB 2: Generate Synthetic Dataset
# ============================================================================
with tab2:
    st.markdown("""
    ### Generate Synthetic Dataset
    Create test datasets using LLMs via OpenRouter API.

    **Use for:**
    - Quick testing
    - Prototyping
    - Small-scale experiments

    **For production benchmarks, use MTEB datasets (Tab 1).**
    """)

    col2, col1 = st.columns([1, 1])

    with col1:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY","")
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=openrouter_api_key,
            help="Get your key at https://openrouter.ai"
        )

    with col2:
        model_choice = st.selectbox(
            "Generator Model",
            [
                "google/gemini-2.0-flash-lite-001",
                "anthropic/claude-3.5-haiku",
                "meta-llama/llama-3.1-8b-instruct"
            ],
            help="Model to use for generating synthetic data"
        )

    seeds = st.text_area(
        "Seed Texts (one per line)",
        value="The 2026 Winter Olympics will be in Italy.\nMetformin reduces glucose production in liver cells.\nPython 3.12 introduces improved error messages.",
        height=150,
        help="Each line will generate one query with multiple candidate documents"
    )

    num_negatives = st.slider(
        "Number of negative docs per query",
        min_value=2,
        max_value=19,
        value=9,
        help="Total docs = 1 gold + N negatives"
    )

    if st.button("🚀 Generate Dataset", type="primary"):
        if not api_key:
            st.error("❌ Please provide an OpenRouter API key")
        else:
            try:
                from openai import OpenAI

                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key
                )

                dataset = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                seed_list = [s.strip() for s in seeds.split("\n") if s.strip()]

                for i, seed in enumerate(seed_list):
                    status_text.text(f"Generating query {i+1}/{len(seed_list)}...")

                    prompt = f"""Generate a RAG reranking dataset item for this seed text: "{seed}"

Create:
1. A query (question or claim related to the seed)
2. A gold document (highly relevant answer/evidence)
3. {num_negatives} negative documents (plausible but not relevant)

Return ONLY valid JSON in this exact format:
{{
  "query": "the query text",
  "docs": ["gold document", "negative doc 1", "negative doc 2", ...],
  "relevance_map": {{"0": 1, "1": 0, "2": 0, ...}}
}}

The gold document must be at index 0.
Negative documents should be plausible but not directly relevant."""

                    response = client.chat.completions.create(
                        model=model_choice,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.7
                    )

                    content = response.choices[0].message.content
                    item = json.loads(content)

                    # Add query_id
                    item['query_id'] = f"synthetic_{i}"

                    # Validate generated item has correct structure
                    if len(item.get('docs', [])) != num_negatives + 1:
                        st.warning(f"Query {i+1}: Expected {num_negatives+1} docs, got {len(item.get('docs', []))}")

                    dataset.append(item)
                    progress_bar.progress((i + 1) / len(seed_list))

                status_text.empty()
                progress_bar.empty()

                st.success(f"✅ Generated {len(dataset)} queries!")

                # Show preview
                st.markdown("### Preview")
                st.text(preview_dataset(dataset, num_samples=2))

                # Save to file
                output_file = "synthetic_dataset.json"
                with open(output_file, "w") as f:
                    json.dump(dataset, f, indent=2)

                st.info(f"💾 Saved to `{output_file}`")

                # Load into session
                if st.button("✅ Use This Dataset", key="use_synthetic"):
                    st.session_state['dataset'] = load_synthetic_dataset(output_file)
                    st.session_state['dataset_name'] = output_file
                    st.session_state['dataset_path'] = str(Path(output_file).resolve())
                    st.session_state['dataset_source'] = 'synthetic'
                    st.success("Synthetic dataset loaded into session!")
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")
                st.exception(e)

# ============================================================================
# TAB 4: Custom Dataset (PDF Import)
# ============================================================================
with tab4:
    st.markdown("""
    ### 📄 Import PDF for RAG Evaluation

    Upload PDF documents (e.g., arXiv papers) to create custom datasets for RAG testing.

    **Use for:**
    - Studying research papers interactively
    - Testing retrieval on domain-specific documents
    - Exploring new content with custom queries

    **Features:**
    - Section-based chunking with overlap
    - Figure/table caption extraction
    - No ground truth needed (Custom Query mode only)
    """)

    col_upload, col_settings = st.columns([1, 1])

    with col_upload:
        st.markdown("#### 📤 Upload PDF")

        pdf_file = st.file_uploader(
            "Select PDF File",
            type=['pdf'],
            help="Upload a PDF document (academic papers work best)"
        )

        if pdf_file is not None:
            file_size_mb = len(pdf_file.getvalue()) / (1024 * 1024)
            st.caption(f"📊 File: {pdf_file.name} ({file_size_mb:.2f} MB)")

            if file_size_mb > 10:
                st.warning("⚠️ Large file (>10MB) may take longer to process")

    with col_settings:
        st.markdown("#### ⚙️ Settings")

        dataset_name = st.text_input(
            "Dataset Name",
            value="custom_paper",
            help="Name for the dataset (will be sanitized)"
        )

        include_figures = st.checkbox(
            "Extract Figure/Table Captions",
            value=True,
            help="Include figure and table captions as searchable documents (Tier 1)"
        )

        max_chunk_size = st.slider(
            "Max Chunk Size (words)",
            min_value=200,
            max_value=2000,
            value=1000,
            step=100,
            help="Maximum words per chunk"
        )

        overlap_words = st.slider(
            "Overlap Between Chunks (words)",
            min_value=0,
            max_value=200,
            value=50,
            step=10,
            help="Word overlap for better retrieval across boundaries"
        )

    st.markdown("---")

    # Process button
    if pdf_file is not None:
        col_process, col_info = st.columns([1, 2])

        with col_process:
            process_button = st.button("🔄 Process PDF", type="primary")

        with col_info:
            st.caption("Processing will convert PDF → markdown → chunks → dataset")

        if process_button:
            try:
                from utils.pdf_processor import (
                    convert_pdf_to_markdown,
                    chunk_markdown_by_sections,
                    extract_figures_and_tables,
                    create_custom_dataset,
                    save_custom_dataset,
                    validate_dataset_schema,
                    PDFProcessingError
                )

                # Step 1: Convert PDF to markdown
                with st.spinner("Step 1/4: Converting PDF to markdown..."):
                    pdf_data = convert_pdf_to_markdown(pdf_file)
                    markdown = pdf_data['markdown']
                    figures = pdf_data['figures']
                    tables = pdf_data['tables']
                    metadata = pdf_data['metadata']

                st.success(f"✅ Converted {metadata['num_pages']} pages")

                # Step 2: Chunk markdown
                with st.spinner("Step 2/4: Chunking by sections..."):
                    chunks = chunk_markdown_by_sections(
                        markdown,
                        max_chunk_size=max_chunk_size,
                        overlap_words=overlap_words
                    )

                st.success(f"✅ Created {len(chunks)} chunks")

                # Step 3: Extract figures/tables
                figure_table_docs = []
                if include_figures:
                    with st.spinner("Step 3/4: Extracting figures and tables..."):
                        figure_table_docs = extract_figures_and_tables(figures, tables)

                    st.success(f"✅ Extracted {len(figure_table_docs)} figures/tables")
                else:
                    st.info("ℹ️ Skipping figure/table extraction")

                # Step 4: Create dataset
                with st.spinner("Step 4/4: Creating dataset..."):
                    dataset = create_custom_dataset(
                        chunks=chunks,
                        figure_table_docs=figure_table_docs,
                        dataset_name=dataset_name,
                        pdf_metadata=metadata,
                        chunking_config={
                            'max_chunk_size': max_chunk_size,
                            'overlap_words': overlap_words
                        }
                    )

                    # Validate schema
                    is_valid, error_msg = validate_dataset_schema(dataset)
                    if not is_valid:
                        st.error(f"❌ Dataset validation failed: {error_msg}")
                        st.stop()

                    # Save to disk
                    saved_path = save_custom_dataset(dataset, dataset_name)

                st.success(f"✅ Dataset created successfully!")
                st.info(f"💾 Saved to: `{saved_path}`")

                # Display statistics
                st.markdown("### 📊 Dataset Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Docs", len(dataset[0]['docs']))
                with col2:
                    st.metric("Text Chunks", len(chunks))
                with col3:
                    st.metric("Figures/Tables", len(figure_table_docs))
                with col4:
                    st.metric("Pages", metadata['num_pages'])

                # Preview chunks
                st.markdown("### 📝 Preview: Text Chunks (First 3)")
                with st.expander("Show chunks", expanded=True):
                    for i, chunk in enumerate(chunks[:3]):
                        st.markdown(f"**Chunk {i+1}: {chunk['heading']}**")
                        preview_text = chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text']
                        st.text(preview_text)
                        st.markdown("---")

                # Preview figures
                if figure_table_docs:
                    st.markdown("### 🖼️ Preview: Figures/Tables (First 5)")
                    with st.expander("Show figures/tables", expanded=False):
                        for i, fig_table in enumerate(figure_table_docs[:5]):
                            st.text(fig_table['text'])
                            if i < len(figure_table_docs) - 1:
                                st.markdown("---")

                # Load button
                st.markdown("---")
                if st.button("✅ Load This Dataset", type="primary", key="load_custom"):
                    st.session_state['dataset'] = dataset
                    st.session_state['dataset_name'] = f"{dataset_name}.json"
                    st.session_state['dataset_path'] = str(saved_path)
                    st.session_state['dataset_source'] = 'custom'
                    st.success(f"📥 Custom dataset loaded into session!")
                    st.rerun()

            except PDFProcessingError as e:
                st.error(f"❌ PDF Processing Error:\n\n{str(e)}")
            except ImportError as e:
                st.error(f"❌ Docling not installed:\n\n```\npip install docling>=1.0.0\n```")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.exception(e)

    else:
        st.info("👆 Upload a PDF file to get started")

# ============================================================================
# TAB 3: Current Dataset Info
# ============================================================================
with tab3:
    # First show existing custom datasets
    st.markdown("### 📁 Existing Custom Datasets")

    # Check data/custom/ directory for custom datasets
    custom_dir = Path(__file__).parent.parent.parent / "data" / "custom"

    if custom_dir.exists() and list(custom_dir.glob("*.json")):
        st.caption("Custom datasets saved in data/custom/ directory:")

        custom_files = sorted(custom_dir.glob("*.json"))
        for json_file in custom_files:
            try:
                # Get file info
                file_size = json_file.stat().st_size / 1024  # KB

                # Try to get dataset info
                with open(json_file, 'r') as f:
                    data = json.load(f)

                if isinstance(data, list) and data:
                    num_items = len(data)
                    total_docs = sum(len(item.get('docs', [])) for item in data)

                    # Get dataset type and metadata
                    first_item = data[0]
                    metadata = first_item.get('metadata', {})
                    source_type = "PDF Import" if metadata.get('source') == 'pdf_import' else "Custom"

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        if num_items == 1 and total_docs > 1:
                            # PDF-style dataset
                            st.write(f"📄 **{json_file.stem}** - {source_type} dataset")
                            st.caption(f"   └ 1 dataset item, {total_docs} documents, {file_size:.1f} KB")
                        else:
                            # Multi-query dataset
                            st.write(f"📊 **{json_file.stem}** - {source_type} dataset")
                            st.caption(f"   └ {num_items} queries, {total_docs} docs, {file_size:.1f} KB")

                        # Show PDF filename if available
                        if 'pdf_filename' in metadata:
                            st.caption(f"   📎 Source: {metadata['pdf_filename']}")

                    with col2:
                        # Load button
                        if st.button(f"Load", key=f"load_{json_file.stem}"):
                            try:
                                st.session_state['dataset'] = data
                                st.session_state['dataset_name'] = json_file.name
                                st.session_state['dataset_path'] = str(json_file.resolve())
                                st.session_state['dataset_source'] = 'custom'
                                st.success(f"✅ Loaded {json_file.stem}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to load: {str(e)}")

            except (json.JSONDecodeError, Exception) as e:
                # Show invalid files
                st.write(f"⚠️ **{json_file.stem}** - Invalid JSON file")
                st.caption(f"   └ {file_size:.1f} KB - {str(e)[:50]}...")
    else:
        st.info("📁 No custom datasets found. Upload PDFs in the 'Custom Dataset' tab to create custom datasets.")

    # Show MTEB datasets for reference
    st.markdown("### 📊 Available MTEB Datasets")

    data_dir = Path(__file__).parent.parent.parent / "data"
    mteb_files = sorted(data_dir.glob("dataset-*.json"))

    if mteb_files:
        st.caption(f"Found {len(mteb_files)} MTEB datasets in data/ directory:")

        # Group files for more compact display
        for i, json_file in enumerate(mteb_files):
            if i % 2 == 0:
                col1, col2 = st.columns(2)

            target_col = col1 if i % 2 == 0 else col2

            with target_col:
                try:
                    file_size = json_file.stat().st_size / 1024  # KB
                    st.write(f"📊 **{json_file.stem}**")
                    st.caption(f"   └ {file_size:.1f} KB")

                    # Quick load button
                    if st.button(f"Load", key=f"load_mteb_{json_file.stem}"):
                        try:
                            with open(json_file, 'r') as f:
                                data = load_mteb_dataset(f, validate=True)
                            st.session_state['dataset'] = data
                            st.session_state['dataset_name'] = json_file.name
                            st.session_state['dataset_path'] = str(json_file.resolve())
                            st.session_state['dataset_source'] = 'mteb'
                            st.success(f"✅ Loaded {json_file.stem}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to load: {str(e)}")

                except Exception as e:
                    st.write(f"⚠️ **{json_file.stem}** - Error reading file")
    else:
        st.info("📊 No MTEB datasets found in data/ directory.")

    st.markdown("---")
    st.markdown("### 🔄 Currently Loaded Dataset")

    if st.session_state['dataset'] is None:
        st.info("ℹ️ No dataset loaded. Use tabs above to load a dataset.")
    else:
        data = st.session_state['dataset']
        stats = get_dataset_statistics(data)

        dataset_path = st.session_state.get('dataset_path', st.session_state['dataset_name'])
        st.success(f"✅ Active Dataset: **{dataset_path}**")
        source = st.session_state.get('dataset_source')
        if source:
            st.caption(f"Source: {source.upper()}")

        # Custom dataset metadata
        if source == 'custom' and len(data) > 0:
            metadata = data[0].get('metadata', {})
            if 'pdf_filename' in metadata:
                st.caption(f"📄 Original PDF: {metadata['pdf_filename']}")
            if metadata.get('has_figures'):
                num_figures = metadata.get('num_figures', 0)
                num_tables = metadata.get('num_tables', 0)
                st.caption(f"🖼️ Includes {num_figures} figures and {num_tables} tables")

        # Statistics
        st.markdown("### Dataset Statistics")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Queries", stats['num_queries'])
        with col2:
            st.metric("Total Docs", stats['total_docs'])
        with col3:
            st.metric("Total Gold", stats['total_gold'])
        with col4:
            st.metric("Docs/Query", f"{stats['num_docs_per_query']:.1f}")
        with col5:
            st.metric("Gold/Query", f"{stats['num_gold_per_query']:.1f}")

        # Length statistics
        st.markdown("### Text Length Statistics")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Queries (words)**")
            st.write(f"- Min: {stats['query_length_min']}")
            st.write(f"- Max: {stats['query_length_max']}")
            st.write(f"- Avg: {stats['query_length_avg']:.1f}")

        with col2:
            st.markdown("**Documents (words)**")
            st.write(f"- Min: {stats['doc_length_min']}")
            st.write(f"- Max: {stats['doc_length_max']}")
            st.write(f"- Avg: {stats['doc_length_avg']:.1f}")

        # Preview
        st.markdown("### Sample Queries")
        with st.expander("Show full preview"):
            st.text(preview_dataset(data, num_samples=5))

        # Export current dataset
        st.markdown("### Export Dataset")
        col1, col2 = st.columns([1, 3])

        with col1:
            export_filename = st.text_input(
                "Filename",
                value="exported_dataset.json",
                help="Filename for exported dataset"
            )

        with col2:
            if st.button("💾 Export to File"):
                try:
                    with open(export_filename, 'w') as f:
                        json.dump(data, f, indent=2)
                    st.success(f"✅ Exported to {export_filename}")
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")

        # Clear dataset
        st.markdown("---")
        if st.button("🗑️ Clear Current Dataset", type="secondary"):
            st.session_state['dataset'] = None
            st.session_state['dataset_name'] = None
            st.session_state['dataset_source'] = None
            st.success("Dataset cleared!")
            st.rerun()

with st.sidebar:
    st.markdown("""
    ### MTEB Dataset Format:
    ```json
    [
      {
        "query": "text",
        "docs": ["doc1", "doc2", ...],
        "relevance_map": {"0": 1, "1": 0, ...},
        "query_id": "id"
      }
    ]
    ```
    """)
