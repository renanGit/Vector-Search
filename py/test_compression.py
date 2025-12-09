"""
Test compression functionality in HNSW
"""

import random

from hnsw import HNSWIndex
from vector_compression import PQCompression


def test_pq_with_pretrained_compression():
    """Test PQ compression with pre-trained quantizer"""
    random.seed(42)

    # Generate training data
    D = 128
    num_train = 1000
    train_data = []
    for _ in range(num_train):
        vec = [random.random() for _ in range(D)]
        train_data.append(vec)

    # Pre-train PQ (use smaller K for test)
    pq_compression = PQCompression(M=8, K=16, D=D)
    pq_compression.Train(train_data)

    # Create HNSW index with pre-trained compression
    index = HNSWIndex(M=16, ef_construction=200, compression=pq_compression)

    # Insert vectors (should be compressed automatically)
    num_vectors = 500
    for _ in range(num_vectors):
        vec = [random.random() for _ in range(D)]
        index.Insert(vec)

    # Verify compression during insertion
    assert index.use_compression
    assert len(index.encoded_vectors) == num_vectors
    # When using compression, we don't store original vectors
    assert len(index.vectors) == 0

    # Search should work
    query = [random.random() for _ in range(D)]
    results = index.KNNSearch(query, topK=10)
    assert len(results) == 10

    print("PASS: PQ with pre-trained compression works correctly")


def test_no_compression():
    """Test that HNSW still works without compression"""
    random.seed(42)

    # Create HNSW index without compression
    D = 128
    index = HNSWIndex(M=16, ef_construction=200)

    # Insert vectors
    num_vectors = 50
    for _ in range(num_vectors):
        vec = [random.random() for _ in range(D)]
        index.Insert(vec)

    # Verify no compression
    assert not index.use_compression
    assert len(index.encoded_vectors) == 0

    # Search should work
    query = [random.random() for _ in range(D)]
    results = index.KNNSearch(query, topK=10)
    assert len(results) == 10

    print("PASS: HNSW without compression works correctly")


if __name__ == "__main__":
    test_no_compression()
    test_pq_with_pretrained_compression()
    print("\nAll compression tests passed!")
