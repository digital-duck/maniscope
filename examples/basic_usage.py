"""
Basic usage example for Maniscope.

This script demonstrates:
1. Initializing ManiscopeEngine
2. Fitting on a document corpus
3. Performing semantic search
4. Comparing baseline vs Maniscope
"""

from maniscope import ManiscopeEngine

# Sample document corpus
documents = [
    "Python is a high-level programming language known for its simplicity.",
    "Python is a non-venomous snake found in tropical regions.",
    "Machine learning frameworks like TensorFlow use Python.",
    "The python snake hunts small mammals and birds.",
    "NumPy and pandas are popular Python libraries for data science.",
    "Ball pythons are popular pets due to their docile nature.",
    "Django is a web framework written in Python.",
    "Pythons are constrictors that squeeze their prey.",
    "Python's syntax emphasizes code readability.",
    "Reticulated pythons are among the longest snakes in the world.",
]

def main():
    print("=" * 70)
    print("Maniscope: Efficient Neural Reranking Demo")
    print("=" * 70)
    
    # Initialize Maniscope engine
    print("\n1. Initializing Maniscope Engine...")
    engine = ManiscopeEngine(
        model_name='all-MiniLM-L6-v2',
        k=5,              # 5 nearest neighbors
        alpha=0.3,        # 30% cosine, 70% geodesic
        verbose=True
    )
    
    # Fit on documents
    print("\n2. Fitting on document corpus...")
    engine.fit(documents)
    
    # Test query
    query = "What does a python eat?"
    print(f"\n3. Query: '{query}'")
    print("-" * 70)
    
    # Baseline search (cosine only)
    print("\n📡 Baseline (Cosine Similarity Only):")
    baseline = engine.search_baseline(query, top_n=3)
    for i, (doc, score, idx) in enumerate(baseline, 1):
        print(f"  [{i}] Score: {score:.3f} | {doc}")
    
    # Maniscope search (cosine + geodesic)
    print("\n🔬 Maniscope (Geodesic Reranking):")
    maniscope = engine.search(query, top_n=3)
    for i, (doc, score, idx) in enumerate(maniscope, 1):
        print(f"  [{i}] Score: {score:.3f} | {doc}")
    
    # Detailed comparison
    print("\n4. Detailed Comparison:")
    print("-" * 70)
    comparison = engine.compare_methods(query, top_n=3)
    print(f"Ranking changed: {comparison['ranking_changed']}")
    print(f"Baseline top-1: Doc #{comparison['baseline_top1']}")
    print(f"Maniscope top-1: Doc #{comparison['maniscope_top1']}")
    
    # Detailed scores
    print("\n5. Detailed Scoring Breakdown:")
    print("-" * 70)
    detailed = engine.search_detailed(query, top_n=3)
    for i, result in enumerate(detailed, 1):
        print(f"\n[{i}] Doc #{result['doc_id']}")
        print(f"    Cosine:  {result['cosine_score']:.3f}")
        print(f"    Geodesic: {result['geo_score']:.3f}")
        print(f"    Final:    {result['final_score']:.3f}")
        print(f"    Connected: {result['connected']}")
        print(f"    Text: {result['document'][:60]}...")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
