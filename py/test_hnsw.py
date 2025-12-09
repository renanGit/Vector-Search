import random
import unittest

from hnsw import Graph, HNSWIndex, Item


class TestGraph(unittest.TestCase):
    """Test cases for the Graph class"""

    def setUp(self):
        self.graph = Graph()

    def test_init_graph(self):
        """Test graph initialization"""
        self.assertEqual(self.graph.GetHeight(), 0)
        self.assertEqual(len(self.graph.layers), 0)

    def test_get_height(self):
        """Test GetHeight returns correct number of layers"""
        self.graph.InitLevels(2)
        self.assertEqual(self.graph.GetHeight(), 3)

    def test_is_layer_empty(self):
        """Test IsLayerEmpty for various scenarios"""
        # Empty graph
        self.assertTrue(self.graph.IsLayerEmpty(0))

        # After initializing layers
        self.graph.InitLevels(1)
        self.assertTrue(self.graph.IsLayerEmpty(0))
        self.assertTrue(self.graph.IsLayerEmpty(1))

        # After adding edges
        self.graph.AddEdge(0, 1, 2)
        self.assertFalse(self.graph.IsLayerEmpty(0))

        # Layer out of bounds
        self.assertTrue(self.graph.IsLayerEmpty(10))

    def test_layer_node_cnt(self):
        """Test LayerNodeCnt returns correct count"""
        self.graph.InitLevels(0)
        self.assertEqual(self.graph.LayerNodeCnt(0), 0)

        self.graph.AddEdge(0, 1, 2)
        self.assertEqual(self.graph.LayerNodeCnt(0), 1)

        self.graph.AddEdge(0, 3, 4)
        self.assertEqual(self.graph.LayerNodeCnt(0), 2)

    def test_layer_node_adj_cnt(self):
        """Test LayerNodeAdjCnt returns correct adjacency count"""
        self.graph.InitLevels(0)

        # Non-existent node
        self.assertEqual(self.graph.LayerNodeAdjCnt(0, 1), 0)

        # Node with neighbors
        self.graph.AddEdge(0, 1, 2)
        self.graph.AddEdge(0, 1, 3)
        self.graph.AddEdge(0, 1, 4)
        self.assertEqual(self.graph.LayerNodeAdjCnt(0, 1), 3)

    def test_get_neighbors(self):
        """Test GetNeighbors returns correct neighbor set"""
        self.graph.InitLevels(0)

        # Non-existent node
        self.assertEqual(self.graph.GetNeighbors(0, 1), set())

        # Node with neighbors
        self.graph.AddEdge(0, 1, 2)
        self.graph.AddEdge(0, 1, 3)
        self.assertEqual(self.graph.GetNeighbors(0, 1), {2, 3})

    def test_get_layer_nodes(self):
        """Test GetLayerNodes returns all nodes in layer"""
        self.graph.InitLevels(0)
        self.graph.AddEdge(0, 1, 2)
        self.graph.AddEdge(0, 3, 4)

        nodes = list(self.graph.GetLayerNodes(0))
        self.assertEqual(set(nodes), {1, 3})

    def test_init_levels(self):
        """Test InitLevels creates correct number of layers"""
        self.graph.InitLevels(4)
        self.assertEqual(self.graph.GetHeight(), 5)

        # Verify all layers are dicts
        for layer in self.graph.layers:
            self.assertIsInstance(layer, dict)

    def test_add_edge(self):
        """Test AddEdge adds edges correctly"""
        self.graph.InitLevels(0)

        # Add first edge
        self.graph.AddEdge(0, 1, 2)
        self.assertIn(2, self.graph.GetNeighbors(0, 1))

        # Add another edge from same node
        self.graph.AddEdge(0, 1, 3)
        self.assertEqual(self.graph.GetNeighbors(0, 1), {2, 3})

        # Verify no duplicates
        self.graph.AddEdge(0, 1, 2)
        self.assertEqual(len(self.graph.GetNeighbors(0, 1)), 2)

    def test_remove_edge(self):
        """Test RemoveEdge removes edges correctly"""
        self.graph.InitLevels(0)
        self.graph.AddEdge(0, 1, 2)
        self.graph.AddEdge(0, 1, 3)

        self.graph.RemoveEdge(0, 1, 2)
        self.assertEqual(self.graph.GetNeighbors(0, 1), {3})

        # Remove non-existent edge (should not error)
        self.graph.RemoveEdge(0, 1, 99)
        self.assertEqual(self.graph.GetNeighbors(0, 1), {3})


class TestItem(unittest.TestCase):
    """Test cases for the Item class"""

    def setUp(self):
        self.vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        self.dist_fn = lambda q, node: sum([(x - y) ** 2 for x, y in zip(q, self.vectors[node])])
        self.dist_fn_cache = lambda idx_v, idx_w: sum(
            [(x - y) ** 2 for x, y in zip(self.vectors[idx_v], self.vectors[idx_w])]
        )

    def test_item_with_query_vector(self):
        """Test Item with query vector (no cache)"""
        q = [2.0, 3.0]
        item = Item(self.dist_fn, q)

        # Distance to node 0
        expected_dist = (2.0 - 1.0) ** 2 + (3.0 - 2.0) ** 2  # = 2.0
        self.assertAlmostEqual(item.DistToNode(0), expected_dist)

    def test_item_with_index(self):
        """Test Item with index (cache version)"""
        item = Item(self.dist_fn_cache, None, 0)

        # Distance from vector 0 to vector 1
        expected_dist = (1.0 - 3.0) ** 2 + (2.0 - 4.0) ** 2  # = 8.0
        self.assertAlmostEqual(item.DistToNode(1), expected_dist)


class TestHNSWIndex(unittest.TestCase):
    """Test cases for the HNSWIndex class"""

    def setUp(self):
        self.M = 4
        self.ef_construction = 10
        self.index = HNSWIndex(self.M, self.ef_construction)

    def test_init_hnsw_index(self):
        """Test HNSWIndex initialization"""
        self.assertEqual(self.index.M, self.M)
        self.assertEqual(self.index.M_max, self.M)
        self.assertEqual(self.index.M_max0, self.M * 2)
        self.assertEqual(self.index.ef_construction, self.ef_construction)
        self.assertEqual(self.index.ef_search, 200)
        self.assertEqual(self.index.ep, 0)
        self.assertEqual(self.index.L, 0)
        self.assertEqual(len(self.index.vectors), 0)
        self.assertIsInstance(self.index.graph, Graph)

    def test_l2_sqr_distance(self):
        """Test L2Sqr distance calculation"""
        p = (1.0, 2.0, 3.0)
        q = (4.0, 5.0, 6.0)

        expected = (1 - 4) ** 2 + (2 - 5) ** 2 + (3 - 6) ** 2  # = 27
        self.assertEqual(self.index.L2Sqr(p, q), expected)

    def test_l2_sqr_distance_zero(self):
        """Test L2Sqr distance with identical vectors"""
        p = (1.0, 2.0, 3.0)
        self.assertEqual(self.index.L2Sqr(p, p), 0.0)

    def test_insert_first_element(self):
        """Test inserting the first element"""
        q = [1.0, 2.0, 3.0]
        self.index.Insert(q)

        self.assertEqual(len(self.index.vectors), 1)
        self.assertEqual(self.index.vectors[0], q)

    def test_insert_multiple_elements(self):
        """Test inserting multiple elements"""
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [2.0, 3.0, 4.0]]

        for vec in vectors:
            self.index.Insert(vec)

        self.assertEqual(len(self.index.vectors), len(vectors))
        for i, vec in enumerate(vectors):
            self.assertEqual(self.index.vectors[i], vec)

    def test_search_layer_simple(self):
        """Test SearchLayer with a simple graph"""
        # Insert some vectors
        vectors = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

        for vec in vectors:
            self.index.Insert(vec)

        # Search for nearest to query
        q_item = Item(self.index.dist_to_node, [0.5, 0.5])
        results = self.index.SearchLayer(q_item, 1, ef=4, l_c=0)

        # Should return results
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 2)  # (distance, index)

    def test_knn_search_simple(self):
        """Test KNN search with simple dataset"""
        # Create a simple 2D dataset
        vectors = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]

        for vec in vectors:
            self.index.Insert(vec)

        # Search for points near [0.1, 0.1]
        query = [0.1, 0.1]
        results = self.index.KNNSearch(query, topK=2)

        self.assertEqual(len(results), 2)

        # First result should be closest (vector 0: [0.0, 0.0])
        dist, idx = results[0]
        self.assertEqual(idx, 0)
        self.assertAlmostEqual(dist, 0.02, places=5)  # (0.1)^2 + (0.1)^2

    def test_knn_search_with_custom_ef(self):
        """Test KNN search with custom ef_search parameter"""
        vectors = [[i, i] for i in range(10)]

        for vec in vectors:
            self.index.Insert(vec)

        query = [5.0, 5.0]
        results = self.index.KNNSearch(query, topK=3, ef_search=50)

        self.assertEqual(len(results), 3)

    def test_select_neighbors_simple(self):
        """Test SelectNeighbors with simple selection"""
        # Add some vectors to test with
        vectors = [[i, i] for i in range(5)]
        for vec in vectors:
            self.index.Insert(vec)

        # Create candidate list with distances
        C = [(1.0, 1), (2.0, 2), (0.5, 3), (3.0, 4)]

        result = self.index.SelectNeighbors(C, M=2, use_simple=True)

        self.assertEqual(len(result), 2)
        # Should select the two with smallest distances
        self.assertEqual(result[0][1], 3)  # distance 0.5
        self.assertEqual(result[1][1], 1)  # distance 1.0

    def test_select_neighbors_heuristic(self):
        """Test SelectNeighbors with heuristic selection"""
        vectors = [[i, i] for i in range(10)]
        for vec in vectors:
            self.index.Insert(vec)

        C = [(float(i), i) for i in range(1, 6)]

        result = self.index.SelectNeighbors(C, M=3, use_simple=False)

        # Should return at most M neighbors
        self.assertLessEqual(len(result), 3)
        # Results should be sorted by distance
        for i in range(len(result) - 1):
            self.assertLessEqual(result[i][0], result[i + 1][0])

    def test_update_connection(self):
        """Test UpdateConnection updates edges correctly"""
        vectors = [[i, i] for i in range(5)]
        for vec in vectors:
            self.index.Insert(vec)

        # Manually add some edges
        self.index.graph.InitLevels(0)
        self.index.graph.AddEdge(0, 0, 1)
        self.index.graph.AddEdge(0, 0, 2)

        # Update connections
        new_neighbors = [(1.0, 3), (2.0, 4)]
        self.index.UpdateConnection(0, 0, new_neighbors)

        # Check that old connections are removed and new ones added
        neighbors = self.index.graph.GetNeighbors(0, 0)
        self.assertIn(3, neighbors)
        self.assertIn(4, neighbors)

    def test_empty_search(self):
        """Test that searching empty index handles gracefully"""
        # Insert at least one element (required)
        self.index.Insert([0.0, 0.0])

        query = [1.0, 1.0]
        results = self.index.KNNSearch(query, topK=5)

        # Should get at most 1 result since only 1 element exists
        self.assertLessEqual(len(results), 1)

    def test_knn_search_topk_larger_than_dataset(self):
        """Test KNN search when topK is larger than dataset size"""
        vectors = [[i, i] for i in range(5)]
        for vec in vectors:
            self.index.Insert(vec)

        query = [2.5, 2.5]
        results = self.index.KNNSearch(query, topK=100)

        # Should return at most the number of vectors in dataset
        self.assertLessEqual(len(results), len(vectors))

    def test_graph_connectivity(self):
        """Test that graph maintains connectivity after insertions"""
        vectors = [[i, i] for i in range(10)]
        for vec in vectors:
            self.index.Insert(vec)

        # Check that layer 0 has nodes
        self.assertFalse(self.index.graph.IsLayerEmpty(0))

        # Check that some nodes have neighbors
        has_neighbors = False
        for node in range(1, len(vectors)):
            if self.index.graph.LayerNodeAdjCnt(0, node) > 0:
                has_neighbors = True
                break

        self.assertTrue(has_neighbors)

    def test_dimensional_consistency(self):
        """Test that vectors of same dimension work correctly"""
        dim = 128
        num_vectors = 20

        random.seed(42)
        vectors = [[random.random() for _ in range(dim)] for _ in range(num_vectors)]

        for vec in vectors:
            self.index.Insert(vec)

        query = [random.random() for _ in range(dim)]
        results = self.index.KNNSearch(query, topK=5)

        self.assertEqual(len(results), 5)

        # Verify results are sorted by distance
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i][0], results[i + 1][0])

    def test_recall_accuracy(self):
        """Test recall accuracy of HNSW search"""
        # Create a simple dataset where we know the nearest neighbors
        random.seed(42)
        vectors = [[i * 1.0, i * 1.0] for i in range(50)]

        for vec in vectors:
            self.index.Insert(vec)

        # Query should find nearest points
        query = [25.0, 25.0]
        results = self.index.KNNSearch(query, topK=5)

        # The closest should be index 25
        indices = [idx for _, idx in results]
        self.assertIn(25, indices)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_single_element_search(self):
        """Test search with only one element"""
        index = HNSWIndex(M=4, ef_construction=10)
        index.Insert([1.0, 2.0])

        results = index.KNNSearch([1.0, 2.0], topK=1)
        self.assertEqual(len(results), 1)

    def test_identical_vectors(self):
        """Test with identical vectors"""
        index = HNSWIndex(M=4, ef_construction=10)

        same_vec = [1.0, 2.0, 3.0]
        for _ in range(5):
            index.Insert(same_vec)

        results = index.KNNSearch(same_vec, topK=3)

        # All distances should be 0
        for dist, _ in results:
            self.assertAlmostEqual(dist, 0.0)

    def test_high_dimensional_vectors(self):
        """Test with high-dimensional vectors"""
        index = HNSWIndex(M=4, ef_construction=10)

        dim = 512
        random.seed(42)
        vectors = [[random.random() for _ in range(dim)] for _ in range(10)]

        for vec in vectors:
            index.Insert(vec)

        query = [random.random() for _ in range(dim)]
        results = index.KNNSearch(query, topK=5)

        self.assertEqual(len(results), 5)

    def test_small_m_parameter(self):
        """Test with small M parameter"""
        index = HNSWIndex(M=2, ef_construction=5)

        # Set random seed for reproducibility
        random.seed(42)
        vectors = [[float(i), float(i)] for i in range(10)]
        for vec in vectors:
            index.Insert(vec)

        results = index.KNNSearch([5.0, 5.0], topK=3)
        self.assertEqual(len(results), 3)

    def test_large_m_parameter(self):
        """Test with large M parameter"""
        index = HNSWIndex(M=20, ef_construction=50)

        vectors = [[i, i] for i in range(10)]
        for vec in vectors:
            index.Insert(vec)

        results = index.KNNSearch([5.0, 5.0], topK=3)
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
