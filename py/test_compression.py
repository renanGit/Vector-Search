import random
import unittest

from hnsw import HNSWIndex
from vector_compression import PQCompression


class TestHNSWWithCompression(unittest.TestCase):
    """Test cases for HNSW with PQ compression"""

    def setUp(self):
        """Set up test fixtures"""
        random.seed(42)
        self.D = 128
        self.M_pq = 8  # PQ subspaces
        self.K_pq = 16  # PQ centroids per subspace
        self.M_hnsw = 16  # HNSW M parameter
        self.ef_construction = 200

    def test_pq_with_pretrained_compression(self):
        """Test PQ compression with pre-trained quantizer"""
        # Generate training data
        num_train = 1000
        train_data = []
        for _ in range(num_train):
            vec = [random.random() for _ in range(self.D)]
            train_data.append(vec)

        # Pre-train PQ
        pq_compression = PQCompression(M=self.M_pq, K=self.K_pq, D=self.D)
        pq_compression.Train(train_data)

        # Create HNSW index with pre-trained compression
        index = HNSWIndex(M=self.M_hnsw, ef_construction=self.ef_construction, compression=pq_compression)

        # Insert vectors (should be compressed automatically)
        num_vectors = 500
        for _ in range(num_vectors):
            vec = [random.random() for _ in range(self.D)]
            index.Insert(vec)

        # Verify compression during insertion
        self.assertTrue(index.use_compression)
        self.assertEqual(len(index.encoded_vectors), num_vectors)
        # When using compression, we don't store original vectors
        self.assertEqual(len(index.vectors), 0)

        # Search should work
        query = [random.random() for _ in range(self.D)]
        results = index.KNNSearch(query, topK=10)
        self.assertEqual(len(results), 10)

        # Verify results are sorted by distance
        for i in range(len(results) - 1):
            self.assertLess(results[i][0], results[i + 1][0])

    def test_no_compression(self):
        """Test that HNSW still works without compression"""
        # Create HNSW index without compression
        index = HNSWIndex(M=self.M_hnsw, ef_construction=self.ef_construction)

        # Insert vectors
        num_vectors = 50
        for _ in range(num_vectors):
            vec = [random.random() for _ in range(self.D)]
            index.Insert(vec)

        # Verify no compression
        self.assertFalse(index.use_compression)
        self.assertEqual(len(index.encoded_vectors), 0)
        self.assertEqual(len(index.vectors), num_vectors)

        # Search should work
        query = [random.random() for _ in range(self.D)]
        results = index.KNNSearch(query, topK=10)
        self.assertEqual(len(results), 10)

    def test_compression_search_accuracy(self):
        """Test that compressed search maintains reasonable accuracy"""
        # Generate training and index data
        num_train = 500
        train_data = [[random.random() for _ in range(self.D)] for _ in range(num_train)]

        # Train PQ
        pq_compression = PQCompression(M=self.M_pq, K=self.K_pq, D=self.D)
        pq_compression.Train(train_data)

        # Create index with compression
        index = HNSWIndex(M=self.M_hnsw, ef_construction=self.ef_construction, compression=pq_compression)

        # Insert vectors
        num_vectors = 100
        for _ in range(num_vectors):
            vec = [random.random() for _ in range(self.D)]
            index.Insert(vec)

        # Search should return valid results
        query = [random.random() for _ in range(self.D)]
        results = index.KNNSearch(query, topK=5)

        self.assertEqual(len(results), 5)
        # All distances should be non-negative
        for dist, idx in results:
            self.assertGreaterEqual(dist, 0.0)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, num_vectors)

    def test_compression_with_small_dataset(self):
        """Test compression with a small dataset"""
        # Generate minimal training data
        num_train = 100
        train_data = [[random.random() for _ in range(self.D)] for _ in range(num_train)]

        # Train PQ with smaller K
        pq_compression = PQCompression(M=4, K=8, D=self.D)
        pq_compression.Train(train_data)

        # Create index
        index = HNSWIndex(M=8, ef_construction=50, compression=pq_compression)

        # Insert a few vectors
        for _ in range(10):
            vec = [random.random() for _ in range(self.D)]
            index.Insert(vec)

        # Verify compression is active
        self.assertTrue(index.use_compression)
        self.assertEqual(len(index.encoded_vectors), 10)

        # Search should work
        query = [random.random() for _ in range(self.D)]
        results = index.KNNSearch(query, topK=5)
        self.assertEqual(len(results), 5)


class TestHNSWCompressionEdgeCases(unittest.TestCase):
    """Test edge cases for HNSW with compression"""

    def setUp(self):
        """Set up test fixtures"""
        random.seed(42)

    def test_single_vector_with_compression(self):
        """Test compression with only one vector"""
        D = 64
        train_data = [[random.random() for _ in range(D)] for _ in range(100)]

        pq_compression = PQCompression(M=4, K=8, D=D)
        pq_compression.Train(train_data)

        index = HNSWIndex(M=16, ef_construction=200, compression=pq_compression)
        vec = [random.random() for _ in range(D)]
        index.Insert(vec)

        # Should have one compressed vector
        self.assertEqual(len(index.encoded_vectors), 1)

        # Search should return the single vector
        query = [random.random() for _ in range(D)]
        results = index.KNNSearch(query, topK=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], 0)  # Should be vector at index 0

    def test_identical_vectors_with_compression(self):
        """Test compression with identical vectors"""
        D = 32
        train_data = [[random.random() for _ in range(D)] for _ in range(50)]

        pq_compression = PQCompression(M=4, K=8, D=D)
        pq_compression.Train(train_data)

        index = HNSWIndex(M=8, ef_construction=100, compression=pq_compression)

        # Insert same vector multiple times
        same_vec = [0.5] * D
        for _ in range(5):
            index.Insert(same_vec)

        self.assertEqual(len(index.encoded_vectors), 5)

        # Search for the same vector
        results = index.KNNSearch(same_vec, topK=3)
        self.assertEqual(len(results), 3)

        # All distances should be small (quantization error only)
        for dist, _ in results:
            self.assertLess(dist, 1.0)

    def test_high_dimensional_with_compression(self):
        """Test compression with high-dimensional vectors"""
        D = 512
        num_train = 200
        train_data = [[random.random() for _ in range(D)] for _ in range(num_train)]

        # Use more subspaces for high dimensions
        pq_compression = PQCompression(M=16, K=16, D=D)
        pq_compression.Train(train_data)

        index = HNSWIndex(M=16, ef_construction=200, compression=pq_compression)

        # Insert vectors
        num_vectors = 20
        for _ in range(num_vectors):
            vec = [random.random() for _ in range(D)]
            index.Insert(vec)

        self.assertEqual(len(index.encoded_vectors), num_vectors)

        # Search should work
        query = [random.random() for _ in range(D)]
        results = index.KNNSearch(query, topK=5)
        self.assertEqual(len(results), 5)


class TestCompressionIntegration(unittest.TestCase):
    """Integration tests for compression functionality"""

    def setUp(self):
        """Set up test fixtures"""
        random.seed(42)

    def test_compression_vs_no_compression_consistency(self):
        """Test that compression and no-compression give consistent structure"""
        D = 64
        num_vectors = 30

        # Create dataset
        random.seed(42)
        vectors = [[random.random() for _ in range(D)] for _ in range(num_vectors)]

        # Index without compression
        index_no_comp = HNSWIndex(M=16, ef_construction=200)
        for vec in vectors:
            index_no_comp.Insert(vec)

        # Index with compression
        random.seed(42)
        train_data = [[random.random() for _ in range(D)] for _ in range(100)]
        pq_compression = PQCompression(M=8, K=16, D=D)
        pq_compression.Train(train_data)

        index_with_comp = HNSWIndex(M=16, ef_construction=200, compression=pq_compression)
        for vec in vectors:
            index_with_comp.Insert(vec)

        # Both should have same number of items (stored differently)
        self.assertEqual(len(index_no_comp.vectors), num_vectors)
        self.assertEqual(len(index_with_comp.encoded_vectors), num_vectors)

        # Both should return results
        query = [random.random() for _ in range(D)]
        results_no_comp = index_no_comp.KNNSearch(query, topK=5)
        results_with_comp = index_with_comp.KNNSearch(query, topK=5)

        self.assertEqual(len(results_no_comp), 5)
        self.assertEqual(len(results_with_comp), 5)

    def test_compression_memory_efficiency(self):
        """Test that compression uses encoded vectors instead of full vectors"""
        D = 128
        num_train = 500
        train_data = [[random.random() for _ in range(D)] for _ in range(num_train)]

        pq_compression = PQCompression(M=8, K=16, D=D)
        pq_compression.Train(train_data)

        index = HNSWIndex(M=16, ef_construction=200, compression=pq_compression)

        # Insert vectors
        num_vectors = 100
        for _ in range(num_vectors):
            vec = [random.random() for _ in range(D)]
            index.Insert(vec)

        # Verify memory efficiency: no full vectors stored
        self.assertEqual(len(index.vectors), 0)
        self.assertEqual(len(index.encoded_vectors), num_vectors)

        # Each encoded vector should be much smaller than original
        # M subspaces, each encoded as 1 byte (0-255 for K<=256)
        for code in index.encoded_vectors:
            self.assertEqual(len(code), pq_compression.pq.M)
            for subcode in code:
                self.assertGreaterEqual(subcode, 0)
                self.assertLess(subcode, pq_compression.pq.K)


if __name__ == "__main__":
    unittest.main()
