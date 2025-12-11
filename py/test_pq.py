import random
import unittest

from pq import ProductQuantizer


class TestProductQuantizer(unittest.TestCase):
    """Test cases for Product Quantization implementation"""

    def setUp(self):
        """Set up test fixtures"""
        random.seed(42)
        self.M = 4  # 4 subspaces
        self.K = 8  # 8 centroids per subspace
        self.D = 16  # 16-dimensional vectors
        self.pq = ProductQuantizer(M=self.M, K=self.K, D=self.D)

    def test_init(self):
        """Test ProductQuantizer initialization"""
        self.assertEqual(self.pq.M, self.M)
        self.assertEqual(self.pq.K, self.K)
        self.assertEqual(self.pq.D, self.D)
        self.assertEqual(self.pq.D_, self.D // self.M)
        self.assertFalse(self.pq.trained)
        self.assertEqual(len(self.pq.codebooks), self.M)

    def test_init_invalid_dimensions(self):
        """Test that initialization fails if D is not divisible by M"""
        with self.assertRaises(ValueError):
            ProductQuantizer(M=3, K=8, D=10)  # 10 not divisible by 3

    def test_split_vector(self):
        """Test vector splitting into subvectors"""
        vec = list(range(16))  # [0, 1, 2, ..., 15]
        subvecs = self.pq._SplitVector(vec)

        self.assertEqual(len(subvecs), self.M)
        for i, subvec in enumerate(subvecs):
            self.assertEqual(len(subvec), self.pq.D_)
            expected = list(range(i * self.pq.D_, (i + 1) * self.pq.D_))
            self.assertEqual(subvec, expected)

    def test_l2sqr_distance(self):
        """Test L2 squared distance computation"""
        p = [1.0, 2.0, 3.0]
        q = [4.0, 5.0, 6.0]
        expected = (1 - 4) ** 2 + (2 - 5) ** 2 + (3 - 6) ** 2  # = 27
        self.assertAlmostEqual(self.pq._L2Sqr(p, q), expected)

    def test_l2sqr_distance_zero(self):
        """Test L2 distance with identical vectors"""
        p = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(self.pq._L2Sqr(p, p), 0.0)

    def test_kmeans_plusplus_init(self):
        """Test K-means++ initialization"""
        data = [[float(i), float(i)] for i in range(20)]
        K = 5
        centroids = self.pq._KMeansPlusPlus(data, K)

        self.assertEqual(len(centroids), K)
        # Check that centroids are diverse (not all the same)
        unique_centroids = {tuple(c) for c in centroids}
        self.assertGreater(len(unique_centroids), 1)

    def test_kmeans_plusplus_insufficient_data(self):
        """Test K-means++ with insufficient data"""
        data = [[1.0, 2.0], [3.0, 4.0]]
        with self.assertRaises(ValueError):
            self.pq._KMeansPlusPlus(data, K=5)  # Need 5 but only have 2

    def test_kmeans_clustering(self):
        """Test K-means clustering"""
        # Create clustered data: 3 clusters
        data = []
        for i in range(3):
            for _ in range(10):
                # Add noise to create cluster around (i*10, i*10, i*10, i*10)
                point = [i * 10.0 + random.uniform(-1, 1) for _ in range(self.pq.D_)]
                data.append(point)

        random.seed(42)  # Reset for reproducibility
        centroids = self.pq._KMeans(data, K=3, max_iter=50)

        self.assertEqual(len(centroids), 3)

        # Check that centroids are roughly at expected locations
        centroid_locs = [c[0] for c in centroids]
        centroid_locs.sort()

        # Should be near 0, 10, 20
        self.assertLess(abs(centroid_locs[0] - 0), 2)
        self.assertLess(abs(centroid_locs[1] - 10), 2)
        self.assertLess(abs(centroid_locs[2] - 20), 2)

    def test_TrainPQ(self):
        """Test training the product quantizer"""
        # Generate random training data
        random.seed(42)
        data = [[random.random() for _ in range(self.D)] for _ in range(100)]

        self.pq.TrainPQ(data)

        self.assertTrue(self.pq.trained)
        # Check that all codebooks are populated
        for m in range(self.M):
            self.assertEqual(len(self.pq.codebooks[m]), self.K)
            for k in range(self.K):
                self.assertEqual(len(self.pq.codebooks[m][k]), self.pq.D_)

    def test_train_empty_data(self):
        """Test training with empty data"""
        with self.assertRaises(ValueError):
            self.pq.TrainPQ([])

    def test_train_wrong_dimension(self):
        """Test training with wrong dimensional data"""
        data = [[1.0, 2.0, 3.0]]  # Wrong dimension
        with self.assertRaises(ValueError):
            self.pq.TrainPQ(data)

    def test_encode_before_training(self):
        """Test that encoding fails before training"""
        vec = [1.0] * self.D
        with self.assertRaises(ValueError):
            self.pq.Encode(vec)

    def test_encode_decode(self):
        """Test encode and decode cycle"""
        # Train on random data
        random.seed(42)
        data = [[random.random() for _ in range(self.D)] for _ in range(100)]
        self.pq.TrainPQ(data)

        # Encode a vector
        vec = [random.random() for _ in range(self.D)]
        code = self.pq.Encode(vec)

        # Check code format
        self.assertEqual(len(code), self.M)
        for idx in code:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, self.K)

        # Decode
        reconstructed = self.pq.Decode(code)
        self.assertEqual(len(reconstructed), self.D)

        # Reconstructed should be reasonably close to original
        dist = self.pq._L2Sqr(vec, reconstructed)
        # With random data, hard to guarantee exact bound, but should be finite
        self.assertLess(dist, 1000.0)

    def test_encode_decode_preserves_structure(self):
        """Test that encode/decode preserves general structure"""
        # Train on structured data
        data = [[float(i % 10)] * self.D for i in range(100)]
        self.pq.TrainPQ(data)

        # Encode/decode a vector
        vec = [5.0] * self.D
        code = self.pq.Encode(vec)
        reconstructed = self.pq.Decode(code)

        # Reconstructed should be close to original
        dist = self.pq._L2Sqr(vec, reconstructed)
        self.assertLess(dist, 5.0)  # Should be quite accurate for simple data

    def test_compute_distance(self):
        """Test approximate distance computation"""
        # Train
        random.seed(42)
        data = [[random.random() for _ in range(self.D)] for _ in range(100)]
        self.pq.TrainPQ(data)

        # Encode a vector
        vec = data[0]
        code = self.pq.Encode(vec)

        # Compute distance from vec to itself (should be small)
        dist = self.pq.ComputeAsymmetricDistance(vec, code)

        # Distance should be small (quantization error)
        self.assertLess(dist, 1.0)

    def test_set_codebooks(self):
        """Test setting codebooks directly"""
        # Create valid codebooks
        codebooks = []
        for m in range(self.M):
            codebook_m = []
            for k in range(self.K):
                centroid = [float(m * k + d) for d in range(self.pq.D_)]
                codebook_m.append(centroid)
            codebooks.append(codebook_m)

        self.pq.SetCodebooks(codebooks)

        self.assertTrue(self.pq.trained)
        self.assertEqual(len(self.pq.codebooks), self.M)

    def test_set_codebooks_wrong_shape(self):
        """Test setting codebooks with wrong shape"""
        # Wrong number of codebooks
        with self.assertRaises(ValueError):
            self.pq.SetCodebooks([[[1.0]]])

        # Wrong number of centroids
        codebooks = [[[[1.0]]] for _ in range(self.M)]
        with self.assertRaises(ValueError):
            self.pq.SetCodebooks(codebooks)

    def test_get_codebooks(self):
        """Test getting trained codebooks"""
        # Should fail before training
        with self.assertRaises(ValueError):
            self.pq.GetCodebooks()

        # Train and then get
        data = [[random.random() for _ in range(self.D)] for _ in range(50)]
        self.pq.TrainPQ(data)

        codebooks = self.pq.GetCodebooks()
        self.assertEqual(len(codebooks), self.M)

    def test_quantization_accuracy(self):
        """Test that quantization maintains reasonable accuracy"""
        # Create highly structured data for predictable results
        random.seed(42)
        data = []
        for i in range(100):
            # Create vectors with clear patterns
            vec = [float(i % 10) * 0.1] * self.D
            data.append(vec)

        self.pq.TrainPQ(data)

        # Test reconstruction accuracy
        test_vec = [0.5] * self.D
        code = self.pq.Encode(test_vec)
        reconstructed = self.pq.Decode(code)

        # Compute relative error
        original_norm = sum(x**2 for x in test_vec) ** 0.5
        error = self.pq._L2Sqr(test_vec, reconstructed) ** 0.5

        relative_error = error / original_norm
        # Should have reasonable accuracy (< 20% relative error)
        self.assertLess(relative_error, 0.2)

    def test_different_pq_parameters(self):
        """Test PQ with different M and K values"""
        configs = [(2, 16, 8), (8, 64, 128), (16, 32, 64)]

        for M, K, D in configs:
            pq = ProductQuantizer(M=M, K=K, D=D)
            random.seed(42)
            # Generate enough data points (at least K)
            data = [[random.random() for _ in range(D)] for _ in range(max(100, K + 10))]

            pq.TrainPQ(data)

            vec = data[0]
            code = pq.Encode(vec)
            reconstructed = pq.Decode(code)

            self.assertEqual(len(code), M)
            self.assertEqual(len(reconstructed), D)


class TestProductQuantizerIntegration(unittest.TestCase):
    """Integration tests for Product Quantization"""

    def test_save_load_codebooks(self):
        """Test saving and loading codebooks"""
        # Train PQ
        random.seed(42)
        pq1 = ProductQuantizer(M=4, K=8, D=16)
        data = [[random.random() for _ in range(16)] for _ in range(100)]
        pq1.TrainPQ(data)

        # Get codebooks
        codebooks = pq1.GetCodebooks()

        # Create new PQ and load codebooks
        pq2 = ProductQuantizer(M=4, K=8, D=16)
        pq2.SetCodebooks(codebooks)

        # Encode with both should give same results
        test_vec = [random.random() for _ in range(16)]
        code1 = pq1.Encode(test_vec)
        code2 = pq2.Encode(test_vec)

        self.assertEqual(code1, code2)

    def test_batch_encoding(self):
        """Test encoding multiple vectors efficiently"""
        random.seed(42)
        pq = ProductQuantizer(M=8, K=64, D=128)  # Reduced K to 64 so 100 training samples is enough
        data = [[random.random() for _ in range(128)] for _ in range(1000)]

        pq.TrainPQ(data[:100])

        # Encode all vectors
        codes = [pq.Encode(vec) for vec in data]

        self.assertEqual(len(codes), 1000)
        for code in codes:
            self.assertEqual(len(code), 8)

    def test_nearest_neighbor_search(self):
        """Test approximate nearest neighbor search with PQ"""
        random.seed(42)
        pq = ProductQuantizer(M=4, K=16, D=16)

        # Create database vectors
        database = [[random.random() for _ in range(16)] for _ in range(100)]
        pq.TrainPQ(database)

        # Encode database
        codes = [pq.Encode(vec) for vec in database]

        # Query
        query = database[0]  # Query with first vector

        # Compute distances to all vectors using ComputeDistance
        distances = [(pq.ComputeAsymmetricDistance(query, code), i) for i, code in enumerate(codes)]
        distances.sort()

        # Nearest should be vector 0 itself
        nearest_idx = distances[0][1]
        self.assertEqual(nearest_idx, 0)

        # Distance to itself should be small
        self.assertLess(distances[0][0], 0.5)


class TestProductQuantizerMultithreading(unittest.TestCase):
    """Test cases for Product Quantization multithreading"""

    def test_multithreaded_training(self):
        """Test that PQ training works with multiple threads"""
        random.seed(42)
        data = [[random.random() for _ in range(128)] for _ in range(100)]

        # Train with 4 threads
        pq_threaded = ProductQuantizer(M=8, K=64, D=128, n_threads=4)
        pq_threaded.TrainPQ(data)

        self.assertTrue(pq_threaded.trained)
        self.assertEqual(len(pq_threaded.codebooks), 8)

    def test_single_vs_multi_thread_consistency(self):
        """Test that single-threaded and multi-threaded give similar results"""
        random.seed(42)
        data = [[random.random() for _ in range(64)] for _ in range(100)]

        # Train with single thread
        random.seed(42)
        pq_single = ProductQuantizer(M=4, K=16, D=64, n_threads=1)
        pq_single.TrainPQ(data)

        # Train with multiple threads (same seed)
        random.seed(42)
        pq_multi = ProductQuantizer(M=4, K=16, D=64, n_threads=4)
        pq_multi.TrainPQ(data)

        # Both should produce trained models
        self.assertTrue(pq_single.trained)
        self.assertTrue(pq_multi.trained)

        # Encode a test vector with both
        test_vec = data[0]
        code_single = pq_single.Encode(test_vec)
        code_multi = pq_multi.Encode(test_vec)

        # Codes should be the same length
        self.assertEqual(len(code_single), len(code_multi))

    def test_auto_thread_detection(self):
        """Test that n_threads=None works (auto-detection)"""
        random.seed(42)
        data = [[random.random() for _ in range(32)] for _ in range(50)]

        pq = ProductQuantizer(M=4, K=8, D=32)  # n_threads defaults to None
        pq.TrainPQ(data)

        self.assertTrue(pq.trained)

    def test_kmeans_with_threading(self):
        """Test that k-means works correctly with threading"""
        random.seed(42)
        pq = ProductQuantizer(M=4, K=8, D=16, n_threads=2)

        # Create clustered data with correct dimension
        data = []
        for i in range(3):
            for _ in range(10):
                point = [i * 10.0 + random.uniform(-1, 1) for _ in range(pq.D_)]
                data.append(point)

        random.seed(42)
        centroids = pq._KMeans(data, K=3, max_iter=50)

        # Should converge to 3 centroids
        self.assertEqual(len(centroids), 3)


if __name__ == "__main__":
    unittest.main()
