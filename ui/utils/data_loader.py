"""
Dataset loading and validation utilities.

Supports loading datasets from:
1. MTEB preparation tools (prep_scifact.py, prep_mteb_dataset.py)
2. Synthetic data generation
3. Custom user datasets

All datasets are normalized to a standard format for consistent benchmarking.
"""

import json
from typing import Dict, List, Tuple, Any, Union
from pathlib import Path
import streamlit as st


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


def load_json_file(file_path: Union[str, Path, Any]) -> List[Dict]:
    """
    Load JSON file from various sources.

    Args:
        file_path: Can be a file path (str/Path) or Streamlit UploadedFile object

    Returns:
        Parsed JSON data as list of dicts

    Raises:
        DatasetValidationError: If file cannot be loaded or parsed
    """
    try:
        # Handle Streamlit UploadedFile
        if hasattr(file_path, 'read'):
            content = file_path.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return json.loads(content)

        # Handle file path
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        raise DatasetValidationError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise DatasetValidationError(f"Failed to load file: {e}")


def validate_dataset_schema(data: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate dataset schema for required fields.

    Required fields for each item:
    - query (str): The query text
    - docs (list): List of document texts
    - relevance_map (dict): Mapping of doc index (str) to relevance score (int)

    Optional fields:
    - query_id: Query identifier
    - num_docs: Number of documents
    - num_gold: Number of gold documents

    Args:
        data: List of dataset items

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if not isinstance(data, list):
        errors.append("Dataset must be a list of items")
        return False, errors

    if len(data) == 0:
        errors.append("Dataset is empty")
        return False, errors

    for i, item in enumerate(data):
        # Check required fields
        if not isinstance(item, dict):
            errors.append(f"Item {i}: Must be a dictionary")
            continue

        # Validate query field
        if 'query' not in item:
            errors.append(f"Item {i}: Missing 'query' field")
        elif not isinstance(item['query'], str):
            errors.append(f"Item {i}: 'query' must be a string")

        # Validate docs field
        if 'docs' not in item:
            errors.append(f"Item {i}: Missing 'docs' field")
        elif not isinstance(item['docs'], list):
            errors.append(f"Item {i}: 'docs' must be a list")
        elif len(item['docs']) == 0:
            errors.append(f"Item {i}: 'docs' list is empty")
        else:
            # Check all docs are strings
            for j, doc in enumerate(item['docs']):
                if not isinstance(doc, str):
                    errors.append(f"Item {i}, doc {j}: Document must be a string")

        # Validate relevance_map field
        if 'relevance_map' not in item:
            errors.append(f"Item {i}: Missing 'relevance_map' field")
        elif not isinstance(item['relevance_map'], dict):
            errors.append(f"Item {i}: 'relevance_map' must be a dictionary")
        else:
            # Check relevance_map keys are strings and values are integers
            for key, value in item['relevance_map'].items():
                if not isinstance(key, str):
                    errors.append(f"Item {i}: relevance_map keys must be strings, got {type(key)}")
                if not isinstance(value, (int, float)):
                    errors.append(f"Item {i}: relevance_map values must be numeric, got {type(value)}")

    is_valid = len(errors) == 0
    return is_valid, errors


def normalize_dataset(data: List[Dict]) -> List[Dict]:
    """
    Normalize dataset to standard format.

    Ensures:
    - All required fields are present
    - relevance_map keys are strings
    - Optional fields are populated if missing

    Args:
        data: Raw dataset

    Returns:
        Normalized dataset
    """
    normalized = []

    for i, item in enumerate(data):
        normalized_item = {
            'query': item['query'],
            'docs': item['docs'],
            'relevance_map': {str(k): int(v) for k, v in item['relevance_map'].items()},
            'query_id': item.get('query_id', str(i)),
            'num_docs': len(item['docs']),
        }

        # Add optional fields if present
        if 'num_gold' in item:
            normalized_item['num_gold'] = item['num_gold']

        normalized.append(normalized_item)

    return normalized


def load_mteb_dataset(file_path: Union[str, Path, Any], validate: bool = True) -> List[Dict]:
    """
    Load MTEB dataset from prep_*.py output.

    Expected format:
    [
      {
        "query": "text",
        "docs": ["doc1", "doc2", ...],
        "relevance_map": {"0": 1, "1": 0, ...},
        "query_id": "id",
        "num_docs": 10
      }
    ]

    Args:
        file_path: Path to JSON file or Streamlit UploadedFile
        validate: Whether to validate schema (default: True)

    Returns:
        Loaded and normalized dataset

    Raises:
        DatasetValidationError: If validation fails
    """
    data = load_json_file(file_path)

    if validate:
        is_valid, errors = validate_dataset_schema(data)
        if not is_valid:
            error_msg = "Dataset validation failed:\n" + "\n".join(f"  - {e}" for e in errors[:10])
            if len(errors) > 10:
                error_msg += f"\n  ... and {len(errors) - 10} more errors"
            raise DatasetValidationError(error_msg)

    return normalize_dataset(data)


def load_synthetic_dataset(file_path: Union[str, Path, Any], validate: bool = True) -> List[Dict]:
    """
    Load synthetically generated dataset.

    Handles legacy formats that may use 'gold_idx' instead of 'relevance_map'.

    Args:
        file_path: Path to JSON file or Streamlit UploadedFile
        validate: Whether to validate schema (default: True)

    Returns:
        Loaded and normalized dataset

    Raises:
        DatasetValidationError: If validation fails
    """
    data = load_json_file(file_path)

    # Convert legacy 'gold_idx' format to 'relevance_map'
    for item in data:
        if 'gold_idx' in item and 'relevance_map' not in item:
            gold_idx = item['gold_idx']
            num_docs = len(item.get('docs', []))

            item['relevance_map'] = {
                str(i): 1 if i == gold_idx else 0
                for i in range(num_docs)
            }

    if validate:
        is_valid, errors = validate_dataset_schema(data)
        if not is_valid:
            error_msg = "Dataset validation failed:\n" + "\n".join(f"  - {e}" for e in errors[:10])
            if len(errors) > 10:
                error_msg += f"\n  ... and {len(errors) - 10} more errors"
            raise DatasetValidationError(error_msg)

    return normalize_dataset(data)


def get_dataset_statistics(data: List[Dict]) -> Dict[str, Any]:
    """
    Calculate dataset statistics.

    Args:
        data: Dataset

    Returns:
        Dict with statistics:
        - num_queries: Total number of queries
        - num_docs_per_query: Average docs per query
        - num_gold_per_query: Average gold docs per query
        - total_docs: Total documents across all queries
        - total_gold: Total gold documents
        - query_length_avg: Average query length in words
        - doc_length_avg: Average document length in words
    """
    if not data:
        return {}

    num_queries = len(data)
    total_docs = sum(item['num_docs'] for item in data)
    total_gold = sum(sum(1 for v in item['relevance_map'].values() if v > 0) for item in data)

    query_lengths = [len(item['query'].split()) for item in data]
    doc_lengths = [
        len(doc.split())
        for item in data
        for doc in item['docs']
    ]

    return {
        'num_queries': num_queries,
        'num_docs_per_query': total_docs / num_queries if num_queries > 0 else 0,
        'num_gold_per_query': total_gold / num_queries if num_queries > 0 else 0,
        'total_docs': total_docs,
        'total_gold': total_gold,
        'query_length_avg': sum(query_lengths) / len(query_lengths) if query_lengths else 0,
        'query_length_min': min(query_lengths) if query_lengths else 0,
        'query_length_max': max(query_lengths) if query_lengths else 0,
        'doc_length_avg': sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0,
        'doc_length_min': min(doc_lengths) if doc_lengths else 0,
        'doc_length_max': max(doc_lengths) if doc_lengths else 0,
    }


def preview_dataset(data: List[Dict], num_samples: int = 3) -> str:
    """
    Generate a text preview of the dataset.

    Args:
        data: Dataset
        num_samples: Number of samples to show (default: 3)

    Returns:
        Formatted string preview
    """
    if not data:
        return "Dataset is empty"

    lines = []
    for i, item in enumerate(data[:num_samples]):
        lines.append(f"\n**Query {i+1}** (ID: {item.get('query_id', 'N/A')})")
        lines.append(f"  Text: {item['query'][:100]}{'...' if len(item['query']) > 100 else ''}")
        lines.append(f"  Docs: {item['num_docs']}")

        # Show gold doc indices
        gold_indices = [idx for idx, rel in item['relevance_map'].items() if rel > 0]
        lines.append(f"  Gold docs: {', '.join(gold_indices)}")

        # Show first doc preview
        if item['docs']:
            first_doc = item['docs'][0]
            lines.append(f"  First doc: {first_doc[:80]}{'...' if len(first_doc) > 80 else ''}")

    if len(data) > num_samples:
        lines.append(f"\n... and {len(data) - num_samples} more queries")

    return "\n".join(lines)


# Test if run directly
if __name__ == "__main__":
    # Test with sample data
    sample_data = [
        {
            "query": "What is the capital of France?",
            "docs": [
                "Paris is the capital and most populous city of France.",
                "London is the capital of the United Kingdom.",
                "Berlin is the capital of Germany."
            ],
            "relevance_map": {"0": 1, "1": 0, "2": 0},
            "query_id": "test_1"
        }
    ]

    print("Testing data validation...")
    is_valid, errors = validate_dataset_schema(sample_data)
    print(f"Valid: {is_valid}")
    if errors:
        print("Errors:", errors)

    print("\nTesting normalization...")
    normalized = normalize_dataset(sample_data)
    print(json.dumps(normalized[0], indent=2))

    print("\nTesting statistics...")
    stats = get_dataset_statistics(normalized)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nTesting preview...")
    print(preview_dataset(normalized))
