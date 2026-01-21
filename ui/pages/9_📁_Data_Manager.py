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
tab1, tab2, tab3 = st.tabs(["📁 Load MTEB Dataset", "🤖 Generate Synthetic", "ℹ️ Current Dataset"])

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
# TAB 3: Current Dataset Info
# ============================================================================
with tab3:
    if st.session_state['dataset'] is None:
        st.info("ℹ️ No dataset loaded. Use Tab 1 or Tab 2 to load a dataset.")
    else:
        data = st.session_state['dataset']
        stats = get_dataset_statistics(data)

        dataset_path = st.session_state.get('dataset_path', st.session_state['dataset_name'])
        st.success(f"✅ Active Dataset: **{dataset_path}**")
        st.caption(f"Source: {st.session_state['dataset_source'].upper()}")

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
