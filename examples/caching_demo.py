"""
Demonstration of embedding cache benefits.

This script shows how caching saves time when testing different
k and alpha parameters on the same document corpus.
"""

import time
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
] * 10  # 100 documents total

query = "programming language features"

print("=" * 70)
print("Embedding Cache Benefits Demo")
print("=" * 70)
print(f"\nDocument corpus size: {len(documents)} documents")
print(f"Query: '{query}'\n")

# Test different parameter combinations
parameter_sets = [
    {"k": 3, "alpha": 0.3},
    {"k": 5, "alpha": 0.5},
    {"k": 7, "alpha": 0.7},
]

print("Testing different k/alpha parameters...")
print("=" * 70)

total_time_with_cache = 0
for i, params in enumerate(parameter_sets, 1):
    print(f"\n[{i}] k={params['k']}, alpha={params['alpha']}")

    start = time.time()
    engine = ManiscopeEngine(
        model_name='all-MiniLM-L6-v2',
        k=params['k'],
        alpha=params['alpha'],
        verbose=True,  # Show cache messages
        use_cache=True
    )
    engine.fit(documents)
    results = engine.search(query, top_n=3)
    elapsed = time.time() - start
    total_time_with_cache += elapsed

    print(f"   Time: {elapsed:.2f}s")
    print(f"   Top result: {results[0][0][:60]}...")

print("\n" + "=" * 70)
print(f"Total time with caching: {total_time_with_cache:.2f}s")
print("=" * 70)

print("\n💡 Key Insight:")
print("   - First run: Encodes documents and saves to cache")
print("   - Subsequent runs: Load embeddings from cache instantly")
print("   - This allows rapid experimentation with k/alpha parameters")
print("   - Especially valuable for large document corpora\n")
