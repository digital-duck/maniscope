"""
PDF Processing for Custom Dataset Import

Converts PDF documents (e.g., arXiv papers) to Maniscope-compatible datasets
for RAG evaluation using Docling.

Features:
- Section-based chunking with configurable overlap
- Figure/table caption extraction (Tier 1 - no vision models)
- Preserves document structure for academic papers
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass


def convert_pdf_to_markdown(pdf_file) -> Dict[str, Any]:
    """
    Convert PDF to markdown using Docling.

    Args:
        pdf_file: Streamlit UploadedFile or file path

    Returns:
        dict with keys:
            - markdown: str (full markdown text)
            - figures: List[dict] (figure metadata)
            - tables: List[dict] (table metadata)
            - metadata: dict (document metadata)

    Raises:
        PDFProcessingError: If conversion fails
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise PDFProcessingError(
            "Docling not installed. Run: pip install docling>=1.0.0"
        )

    try:
        # Convert PDF to Docling document
        converter = DocumentConverter()

        # Handle both file path and UploadedFile
        if hasattr(pdf_file, 'read'):
            # Streamlit UploadedFile - save temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name

            result = converter.convert(tmp_path)

            # Clean up temp file
            Path(tmp_path).unlink()
            filename = pdf_file.name
        else:
            result = converter.convert(str(pdf_file))
            filename = Path(pdf_file).name

        # Extract markdown
        markdown = result.document.export_to_markdown()

        # Extract figures and tables metadata
        figures = []
        tables = []

        # Parse document elements for figures and tables
        for element in result.document.body:
            if hasattr(element, 'label') and hasattr(element, 'caption'):
                element_data = {
                    'type': element.label.lower() if hasattr(element, 'label') else 'unknown',
                    'caption': element.caption if hasattr(element, 'caption') else '',
                    'page': getattr(element, 'page', 0),
                    'id': getattr(element, 'id', '')
                }

                if 'figure' in element_data['type']:
                    figures.append(element_data)
                elif 'table' in element_data['type']:
                    tables.append(element_data)

        # Get document metadata
        metadata = {
            'num_pages': getattr(result.document, 'num_pages', 0),
            'title': getattr(result.document, 'title', filename),
            'filename': filename
        }

        return {
            'markdown': markdown,
            'figures': figures,
            'tables': tables,
            'metadata': metadata
        }

    except Exception as e:
        if "password" in str(e).lower():
            raise PDFProcessingError(
                "This PDF is password-protected. Please unlock it first."
            )
        elif "memory" in str(e).lower():
            raise PDFProcessingError(
                "PDF too large - memory error. Try a smaller file (< 10MB)."
            )
        else:
            raise PDFProcessingError(f"PDF conversion failed: {str(e)}")


def chunk_markdown_by_sections(
    markdown: str,
    max_chunk_size: int = 1000,
    overlap_words: int = 50
) -> List[Dict[str, Any]]:
    """
    Chunk markdown by sections (headings) with word overlap.

    Args:
        markdown: Markdown text from Docling
        max_chunk_size: Maximum chunk size in words
        overlap_words: Number of words to overlap between chunks (50-100)

    Returns:
        List of chunks with metadata:
            [{'text': str, 'heading': str, 'level': int}, ...]
    """
    chunks = []

    # Split by markdown headings (##, ###, etc.)
    # Pattern: lines starting with # (heading markers)
    lines = markdown.split('\n')

    current_section = []
    current_heading = "Introduction"
    current_level = 1

    for line in lines:
        # Check if line is a heading
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)

        if heading_match:
            # Save previous section if not empty
            if current_section:
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    # Split large sections with overlap
                    section_chunks = _split_with_overlap(
                        section_text,
                        current_heading,
                        current_level,
                        max_chunk_size,
                        overlap_words
                    )
                    chunks.extend(section_chunks)

            # Start new section
            current_level = len(heading_match.group(1))
            current_heading = heading_match.group(2).strip()
            current_section = []
        else:
            current_section.append(line)

    # Add final section
    if current_section:
        section_text = '\n'.join(current_section).strip()
        if section_text:
            section_chunks = _split_with_overlap(
                section_text,
                current_heading,
                current_level,
                max_chunk_size,
                overlap_words
            )
            chunks.extend(section_chunks)

    # Fallback: if no sections found, split by fixed size with overlap
    if not chunks and markdown.strip():
        chunks = _split_with_overlap(
            markdown,
            "Document",
            1,
            max_chunk_size,
            overlap_words
        )

    return chunks


def _split_with_overlap(
    text: str,
    heading: str,
    level: int,
    max_chunk_size: int,
    overlap_words: int
) -> List[Dict[str, Any]]:
    """
    Split text into chunks with word overlap.

    Args:
        text: Text to split
        heading: Section heading
        level: Heading level (1-6)
        max_chunk_size: Maximum chunk size in words
        overlap_words: Number of words to overlap

    Returns:
        List of chunk dicts
    """
    words = text.split()

    # If section fits in one chunk, return as is
    if len(words) <= max_chunk_size:
        return [{
            'text': text,
            'heading': heading,
            'level': level
        }]

    # Split with overlap
    chunks = []
    start = 0
    chunk_num = 1

    while start < len(words):
        # Get chunk words
        end = min(start + max_chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)

        # Add heading suffix if multiple chunks
        chunk_heading = f"{heading} (Part {chunk_num})" if len(words) > max_chunk_size else heading

        chunks.append({
            'text': chunk_text,
            'heading': chunk_heading,
            'level': level
        })

        # Move start with overlap
        if end >= len(words):
            break
        start = end - overlap_words
        chunk_num += 1

    return chunks


def extract_figures_and_tables(
    figures: List[Dict[str, Any]],
    tables: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract figure/table captions as referenceable documents (Tier 1).

    Args:
        figures: List of figure metadata from Docling
        tables: List of table metadata from Docling

    Returns:
        List of formatted caption documents with metadata
    """
    caption_docs = []

    # Process figures
    for i, fig in enumerate(figures, 1):
        caption = fig.get('caption', '').strip()
        if caption:
            caption_docs.append({
                'text': f"[Figure {i}]: {caption}",
                'metadata': {
                    'type': 'figure',
                    'caption': caption,
                    'page_number': fig.get('page', 0),
                    'figure_id': fig.get('id', f'fig_{i}')
                }
            })

    # Process tables
    for i, table in enumerate(tables, 1):
        caption = table.get('caption', '').strip()
        if caption:
            caption_docs.append({
                'text': f"[Table {i}]: {caption}",
                'metadata': {
                    'type': 'table',
                    'caption': caption,
                    'page_number': table.get('page', 0),
                    'table_id': table.get('id', f'table_{i}')
                }
            })

    return caption_docs


def create_custom_dataset(
    chunks: List[Dict[str, Any]],
    figure_table_docs: List[Dict[str, Any]],
    dataset_name: str,
    pdf_metadata: Dict[str, Any],
    chunking_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Create Maniscope-compatible dataset from chunks and figures.

    Format: Single item with empty query/relevance_map for Custom Query mode.

    Args:
        chunks: List of text chunks from chunking
        figure_table_docs: List of figure/table caption docs
        dataset_name: User-provided dataset name
        pdf_metadata: Metadata from PDF conversion
        chunking_config: Chunking parameters (max_size, overlap)

    Returns:
        Maniscope dataset format (list with single item)
    """
    # Combine chunks and figures into docs list
    docs = []

    # Add text chunks
    for chunk in chunks:
        # Format with heading for context
        heading = chunk.get('heading', '')
        text = chunk.get('text', '')

        if heading:
            formatted_text = f"## {heading}\n\n{text}"
        else:
            formatted_text = text

        docs.append(formatted_text)

    # Add figure/table captions
    for fig_table in figure_table_docs:
        docs.append(fig_table['text'])

    # Create single dataset item
    dataset_item = {
        "query": "",  # Empty - use Custom Query mode
        "docs": docs,
        "relevance_map": {},  # Empty - no ground truth
        "query_id": "custom_0",
        "num_docs": len(docs),
        "metadata": {
            "source": "pdf_import",
            "dataset_name": dataset_name,
            "pdf_filename": pdf_metadata.get('filename', 'unknown.pdf'),
            "has_figures": len(figure_table_docs) > 0,
            "num_pages": pdf_metadata.get('num_pages', 0),
            "num_chunks": len(chunks),
            "num_figures": len([d for d in figure_table_docs if d.get('metadata', {}).get('type') == 'figure']),
            "num_tables": len([d for d in figure_table_docs if d.get('metadata', {}).get('type') == 'table']),
            "chunking_strategy": "section_based",
            "max_chunk_size": chunking_config.get('max_chunk_size', 1000),
            "overlap_words": chunking_config.get('overlap_words', 50),
            "created_at": datetime.now().isoformat()
        }
    }

    return [dataset_item]


def save_custom_dataset(dataset: List[Dict[str, Any]], dataset_name: str) -> Path:
    """
    Save custom dataset to data/custom/{name}.json

    Args:
        dataset: Maniscope dataset format
        dataset_name: User-provided name (will be sanitized)

    Returns:
        Path to saved file

    Raises:
        PDFProcessingError: If save fails
    """
    try:
        # Sanitize filename
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', dataset_name)
        safe_name = safe_name.lower().strip('_')

        if not safe_name:
            safe_name = "custom_dataset"

        # Ensure .json extension
        if not safe_name.endswith('.json'):
            safe_name += '.json'

        # Create custom directory if not exists
        custom_dir = Path(__file__).parent.parent.parent / "data" / "custom"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Save path
        save_path = custom_dir / safe_name

        # Check if file exists (offer overwrite in UI, not here)
        # For now, just save/overwrite

        # Write JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        return save_path

    except Exception as e:
        raise PDFProcessingError(f"Failed to save dataset: {str(e)}")


def validate_dataset_schema(dataset: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validate that dataset conforms to Maniscope schema.

    Args:
        dataset: Dataset to validate

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(dataset, list):
        return False, "Dataset must be a list"

    if len(dataset) == 0:
        return False, "Dataset is empty"

    for i, item in enumerate(dataset):
        # Check required fields
        if 'query' not in item:
            return False, f"Item {i}: missing 'query' field"
        if 'docs' not in item:
            return False, f"Item {i}: missing 'docs' field"
        if 'relevance_map' not in item:
            return False, f"Item {i}: missing 'relevance_map' field"

        # Check types
        if not isinstance(item['docs'], list):
            return False, f"Item {i}: 'docs' must be a list"
        if not isinstance(item['relevance_map'], dict):
            return False, f"Item {i}: 'relevance_map' must be a dict"

        # Check docs not empty (for custom datasets)
        if len(item['docs']) == 0:
            return False, f"Item {i}: 'docs' list is empty"

    return True, ""
