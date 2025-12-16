import math
import random
from functools import lru_cache
from heapq import heapify, heappop, heappush, nlargest, nsmallest

from vector_compression import VectorCompression


class Graph:
    """A layered graph for HNSW algo manipulation"""

    def __init__(self):
        self.layers = list()

    def GetHeight(self) -> int:
        """Returns number of layers present in the hierarchy

        Returns:
            int: Number of layers in the hierarchy"""
        return len(self.layers)

    def IsLayerEmpty(self, l_c: int) -> bool:
        """Returns if a layer is empty, ie. no items in layer l_c

        Parameters:
            l_c (int): Layer index

        Returns:
            bool: True if layer is empty, False otherwise"""
        if l_c > self.GetHeight() - 1 or len(self.layers[l_c]) == 0:
            return True
        return False

    def LayerNodeCnt(self, l_c: int) -> int:
        """Returns number of nodes in layer l_c

        Parameters:
            l_c (int): Layer index

        Returns:
            int: Number of nodes in the layer"""
        if self.IsLayerEmpty(l_c):
            return 0
        return len(self.layers[l_c])

    def LayerNodeAdjCnt(self, l_c: int, node: int) -> int:
        """Returns number of adjacent nodes from parent node

        Parameters:
            l_c (int): Layer index
            node (int): Node index

        Returns:
            int: Number of adjacent nodes"""
        if self.IsLayerEmpty(l_c) or node not in self.layers[l_c]:
            return 0
        return len(self.layers[l_c][node])

    def GetNeighbors(self, l_c: int, node: int) -> set[int]:
        """Returns adjacent nodes, ie. neighbors to node

        Parameters:
            l_c (int): Layer index
            node (int): Node index

        Returns:
            set[int]: Set of neighbor node indices"""
        if node not in self.layers[l_c]:
            return set()
        return self.layers[l_c][node]

    def GetLayerNodes(self, l_c: int) -> list[int]:
        """Returns all parent nodes at layer l_c

        Parameters:
            l_c (int): Layer index

        Returns:
            list[int]: List of all node indices in the layer"""
        return self.layers[l_c].keys()

    def InitLevels(self, l_c: int) -> None:
        """Fills layers with empty dict(), up till l_c

        Parameters:
            l_c (int): Layer index to initialize up to"""
        while l_c > self.GetHeight() - 1:
            self.layers.append(dict())

    def AddEdge(self, l_c: int, p: int, q: int) -> None:
        """Inits p if p doesnt exist, then add edge p to q

        Parameters:
            l_c (int): Layer index
            p (int): Source node index
            q (int): Target node index"""
        if p not in self.layers[l_c]:
            self.layers[l_c][p] = set()
        self.layers[l_c][p].add(q)

    def PrintLayer(self, l_c: int) -> None:
        """Print dict at level l_c

        Parameters:
            l_c (int): Layer index"""
        print(self.layers[l_c])

    def RemoveEdge(self, l_c: int, node: int, nei: int) -> None:
        """Remove nei from parent node

        Parameters:
            l_c (int): Layer index
            node (int): Parent node index
            nei (int): Neighbor node index to remove"""
        if node not in self.layers[l_c] or nei not in self.layers[l_c][node]:
            return
        self.layers[l_c][node].remove(nei)


class Item:
    """An Item container class for representing inserted items in HNSW.

    Parameters:
        dist_fn: a distance function used during insert and search
        q (list): only populated in search
        idx_q (int, default -1): only populated in insert"""

    def __init__(self, dist_fn, q: list, idx_q: int = -1):
        self.q = q
        self.idx_q = idx_q
        # Use cache version if both idx_q and previous add node in vectors are present
        self.dist_fn = dist_fn

    def DistToNode(self, node: int) -> float:
        """Call dist_fn, depending on cache version it will use idx_q instead of q

        Parameters:
            node (int): Node index

        Returns:
            float: Distance to the node"""
        if self.idx_q < 0:
            return self.dist_fn(self.q, node)
        return self.dist_fn(self.idx_q, node)


class HNSWIndex:
    """Implementation for Hierarchical Navigable Small World graphs (HNSW).

    Parameters:
        M (int): number of established connections
        ef_construction (int): size of the dynamic candidate list for graph build
        compression: abstracted compression class for vector compression and search computation"""

    def __init__(self, M: int, ef_construction: int, compression: VectorCompression = None):
        # M: number of established connections
        self.M = M
        # M_max: maximum number of connections for each element per layer
        self.M_max = M
        # M_max0: maximum number of connections for layer 0
        self.M_max0 = M * 2
        # ef_construction: size of the dynamic candidate list
        self.ef_construction = ef_construction
        # ef_search: default ef for search
        self.ef_search = 200
        # m_l: normalization factor level generation
        self.m_l = 1 / math.log(1.0 * M)
        # ep: current entry point
        self.ep = 0
        # L: max layer height
        self.L = 0
        # if the added elements in R < M, then keep the some of the connections
        self.keep_pruned_connections = True

        self.graph = Graph()

        # Storage for vectors (original or compressed)
        self.vectors = list()

        # Compression configuration
        self.compression = compression
        self.use_compression = compression is not None and compression.IsTrained()

        if self.use_compression:
            self.dist_to_node = lambda q, idx: self.PQDistance(q, idx)
            self.dist_to_node_cache = lambda idx_v, idx_w: self.PQDistanceCache(idx_v, idx_w)
        else:
            self.dist_to_node = lambda q, idx: self.L2Sqr(q, self.vectors[idx])
            self.dist_to_node_cache = lambda idx_v, idx_w: self.L2SqrCache(idx_v, idx_w)

    def L2Sqr(self, p: tuple, q: tuple) -> float:
        """Squared Euclidean distance.

        Parameters:
            p (tuple): First vector
            q (tuple): Second vector

        Returns:
            float: Squared L2 distance between p and q"""
        total = 0.0
        for xy in range(len(p)):
            total += (p[xy] - q[xy]) ** 2
        return total

    @lru_cache(maxsize=32768)
    def L2SqrCache(self, idx_v: int, idx_w: int) -> float:
        """Squared Euclidean distance. Reuse computed distance from cache if called before.

        Parameters:
            idx_v (int): index to vector v
            idx_w (int): index to vector w

        Returns:
            float: Squared L2 distance between vectors v an w"""
        return self.L2Sqr(self.vectors[idx_v], self.vectors[idx_w])

    def PQDistance(self, q: list, idx: int) -> float:
        """Asymmetric distance compare. q is a new query, which needs to be encoded for distance compare.

        Parameters:
            q (list): Query vector
            idx (int): Index of encoded vector

        Returns:
            float: Distance between query and encoded vector"""
        return self.compression.ComputeAsymmetricDistance(q, self.vectors[idx])

    @lru_cache(maxsize=16384)
    def PQDistanceCache(self, idx_v: int, idx_w: int) -> float:
        """Symmetric distance compare. Reuse computed distance from cache if called before.

        Parameters:
            idx_v (int): index to vector v
            idx_w (int): index to vector w

        Returns:
            float: Distance between encoded vectors v and w"""
        return self.compression.ComputeSymmetricDistance(self.vectors[idx_v], self.vectors[idx_w])

    def SearchLayer(self, q: Item, ep: int, ef: int = 1, l_c: int = 0) -> list[tuple[float, int]]:
        """Searches nearest neighbor to query

        Parameters:
            q (Item): query item
            ep (int): entry point
            ef (int): number of nearest to q elements to return
            l_c (int): layer number

        Returns:
            list[tuple[float, int]]: W, nearest neighbors to q"""
        ep_dist = q.DistToNode(ep)
        v = {ep}  # set of visited elements
        C = [(ep_dist, ep)]  # set of candidates
        W = [(ep_dist, ep)]  # dynamic list of found nearest neighbors

        while len(C) > 0:
            d_c, c = heappop(C)  # nearest
            d_f, _ = nlargest(1, W, key=lambda w: w[0])[0]  # furthest

            if d_c > d_f:
                break

            for nei in self.graph.GetNeighbors(l_c, c):
                if nei in v:
                    continue

                v.add(nei)
                d_f, _ = nlargest(1, W, key=lambda w: w[0])[0]
                d_e = q.DistToNode(nei)

                if d_e < d_f or len(W) < ef:
                    heappush(C, (d_e, nei))
                    heappush(W, (d_e, nei))

                    if len(W) > ef:
                        W = nsmallest(ef, W, key=lambda w: w[0])
        return W

    def SelectNeighbors(self, C: list[tuple[float, int]], M: int, use_simple: bool = False) -> list[tuple[float, int]]:
        """Select neighbors for updating the edges Clustered neighbors

        Parameters:
            C (list[tuple[float, int]]): candidates with precomputed distances to q
            M (int): max connections to consider
            use_simple (bool): Whether to use simple selection

        Returns:
            list[tuple[float, int]]: R, clustered candidates"""
        dist_to_node = self.dist_to_node_cache

        # SELECT-NEIGHBORS-SIMPLE
        if use_simple:
            return nsmallest(M, C, key=lambda c: c[0])

        # SELECT-NEIGHBORS-HEURISTIC
        # R: how compact the neighbor points are to q, a.k.a clustered points
        R = []
        # W_d: discarded candidates
        W_d = []
        heap = C
        heapify(heap)

        while len(heap) and len(R) < M:
            d_c, c = heappop(heap)
            is_clustered = True

            for _, r in R:
                cur_dist = dist_to_node(r, c)
                if cur_dist < d_c:
                    is_clustered = False
                    break
            if is_clustered:
                R.append((d_c, c))
            else:
                W_d.append((d_c, c))

        if self.keep_pruned_connections and len(W_d) > 0 and len(R) < M:
            R.extend(nsmallest(M - len(R), W_d, key=lambda wd: wd[0]))

        return nsmallest(M, R, key=lambda r: r[0])

    def UpdateConnection(self, l_c: int, nei: int, new_neighbors: list[tuple[float, int]]) -> None:
        """Updates connections for nei to new_neighbors

        Parameters:
            l_c (int): level to update
            nei (int): node index for add/remove edges
            new_neighbors (list[tuple[float, int]]): new edges for nei"""
        # remove all connections from nei to nodeX, where nodeX is previous edge
        # there are better connections from nei to new_neighbors
        for node in self.graph.GetLayerNodes(l_c):
            self.graph.RemoveEdge(l_c, nei, node)
        for _, new_nei in new_neighbors:
            self.graph.AddEdge(l_c, nei, new_nei)
        return

    def Insert(self, q: list) -> None:
        """Insert a vector into the graph

        Parameters:
            q (list): new element"""
        # idx_q represents index of q
        # If using compression, encode and store code
        idx_q = len(self.vectors)
        if self.use_compression:
            q = self.compression.Encode(q)

        self.vectors.append(q)

        W = list()  # list for the currently found nearest elements
        ep = self.ep  # get entry point for hnsw
        L = self.graph.GetHeight() - 1  # top layer for hnsw
        l = math.floor(-math.log(random.uniform(0.0, 1.0) * self.m_l))  # new element's level

        # fill the layers with empty dict
        if self.graph.IsLayerEmpty(l):
            self.graph.InitLevels(l)
        if idx_q == 0:
            return

        dist_to_node = self.dist_to_node_cache
        q_item = Item(dist_to_node, None, idx_q)

        # looking for entry point for q
        for l_c in range(L, l, -1):
            # nearest element from p to q
            ep = self.SearchLayer(q_item, ep, 1, l_c)[0][1]

        for l_c in range(min(L, l), -1, -1):
            # if l_c = 0 then M_max0 -> M
            M = self.M_max0 if l == 0 else self.M_max

            W = self.SearchLayer(q_item, ep, self.ef_construction, l_c)
            neighbors = self.SelectNeighbors(list(W), M)  # , use_simple=True)

            # add bidirectional connections from neighbors to q at layer l_c
            for _, nei in neighbors:
                self.graph.AddEdge(l_c, nei, idx_q)
                self.graph.AddEdge(l_c, idx_q, nei)

            for _, nei in neighbors:
                nei_adj_cnt = self.graph.LayerNodeAdjCnt(l_c, nei)
                # shrink connections of nei
                if nei_adj_cnt > M:
                    C = [(dist_to_node(nei, c), c) for c in self.graph.GetNeighbors(l_c, nei)]
                    e_new_conn = self.SelectNeighbors(C, M)
                    self.UpdateConnection(l_c, nei, e_new_conn)
            ep = W[0][1]  # ep is updated for the next iteration with the nearest point to q for layer l_c - 1

        if l > L:
            self.ep = idx_q  # set entry point for hnsw to q
        return

    def KNNSearch(self, q: list, topK: int, ef_search: int = 0) -> list[tuple[float, int]]:
        """Search function for a query

        Parameters:
            q (list): the vector to search for
            topK (int): amount of results to return
            ef_search (int): number of candidates to consider, defaults to 200 if 0

        Returns:
            list[tuple[float, int]]: topk closest points to q ([(distance1, index1), ...])"""
        L = self.graph.GetHeight() - 1
        ep = self.ep
        ef = self.ef_search if ef_search <= 0 else ef_search
        q_item = Item(self.dist_to_node, q)
        for l in range(L, 0, -1):
            ep = self.SearchLayer(q_item, ep, 1, l)[0][1]
        return self.SearchLayer(q_item, ep, ef, 0)[:topK]
