"""
Product Quantization Implementation

This module implements Product Quantization (PQ), a lossy compression technique for
high-dimensional vectors that enables efficient similarity search with significant
memory savings.

Key concepts:
1. Decompose vectors into M subvectors
2. Cluster each subspace independently with K-means
3. Encode vectors as M indices (one per subspace)
4. Fast distance computation using precomputed lookup tables

References:
- Jégou, H., Douze, M., & Schmid, C. (2011). "Product quantization for nearest
  neighbor search." IEEE TPAMI.
"""

import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


class ProductQuantizer:
    """
    Product Quantization for vector compression and fast approximate search.

    Splits D-dimensional vectors into M subvectors of dimension D'=D/M, then
    quantizes each subspace independently using a codebook of K centroids.

    Parameters
    ----------
    M : int
        Number of subquantizers (subspaces). D must be divisible by M.
    K : int
        Number of centroids per subspace (codebook size). Typically 256.
    D : int
        Dimensionality of input vectors.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    codebooks : list[list[list[float]]]
        M codebooks, each containing K centroids of dimension D'.
        codebooks[m][k] is the k-th centroid in subspace m.
    trained : bool
        Whether the codebooks have been trained.
    """

    def __init__(self, M: int, K: int, D: int, seed: int = 42, n_threads: Optional[int] = None):
        # M: number of subquantizers (how many pieces to split each vector into)
        self.M = M
        # K: number of centroids per subspace (codebook size)
        self.K = K
        # D: dimensionality of input vectors
        self.D = D
        # D_: dimensionality of each subvector (D' = D / M)
        self.D_ = D // M
        # Number of threads for parallel computation (None = auto-detect)
        self.n_threads = n_threads

        # Validate that D is divisible by M
        if D % M != 0:
            raise ValueError(f"D ({D}) must be divisible by M ({M})")

        # Initialize codebooks: M codebooks, each with K centroids of dimension D_
        # codebooks[m] is a list of K centroids for subspace m
        # codebooks[m][k] is a list of D_ float values representing centroid k
        self.codebooks: list[list[list[float]]] = []
        for _ in range(M):
            self.codebooks.append([])

        # Track whether codebooks have been trained
        self.trained = False

        # Random seed for reproducibility
        random.seed(seed)

    def _L2Sqr(self, p: list[float], q: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(p, q))

    # Split Vectors based on M subspaces
    def _SplitVector(self, vec: list[float]) -> list[list[float]]:
        subvectors = []
        for m in range(self.M):
            # Extract subvector for subspace m
            start = m * self.D_
            end = (m + 1) * self.D_
            subvec = vec[start:end]
            subvectors.append(subvec)
        return subvectors

    # K-means++ provides better initialization than random selection by choosing
    #   centroids that are far apart from each other.
    #   https://en.wikipedia.org/wiki/K-means%2B%2B
    # data : Training data points.
    # K : Number of centroids to initialize.
    # return : K initial centroids
    def _KMeansPlusPlus(self, data: list[list[float]], K: int) -> list[list[float]]:
        if len(data) < K:
            raise ValueError(f"Need at least K={K} data points, got {len(data)}")

        # Choose first centroid uniformly at random
        first_idx = random.randint(0, len(data) - 1)
        centroids = [data[first_idx][:]]

        # Choose remaining K-1 centroids
        # Track which indices have been selected to avoid duplicates
        selected_indices = {first_idx}
        selectable_indices = {idx for idx in range(len(data)) if idx != first_idx}

        for _ in range(K - 1):
            # Compute D(x)² for each point not yet selected
            # D(x)² = squared distance to nearest existing centroid
            distances_sq = []
            indices = []

            for idx in selectable_indices:
                point = data[idx]
                # D(x)² = min(distance²(x, centroid)) for all centroids
                # _L2Sqr already returns squared distance
                min_dist_sq = min(self._L2Sqr(point, c) for c in centroids)
                distances_sq.append(min_dist_sq)
                indices.append(idx)

            # Calculate total for normalization: Σ(D(x)²)
            total_dist_sq = sum(distances_sq)

            # Select with probability P(x) = D(x)² / Σ(D(x)²)
            # Use weighted random selection via cumulative distribution
            threshold = random.uniform(0, total_dist_sq)
            selected_idx = indices[-1]  # init
            cumulative = 0.0

            for i, dist_sq in enumerate(distances_sq):
                cumulative += dist_sq
                if cumulative >= threshold:
                    selected_idx = indices[i]
                    break

            # Add the selected point to centroids
            centroids.append(data[selected_idx][:])
            selected_indices.add(selected_idx)
            selectable_indices.remove(selected_idx)

        return centroids

    def _FindNearestCentroid(self, point: list[float], centroids: list[list[float]]) -> int:
        """Helper function to find the nearest centroid for a single point."""
        min_dist = float("inf")
        best_k = 0
        for k, centroid in enumerate(centroids):
            dist = self._L2Sqr(point, centroid)
            if dist < min_dist:
                min_dist = dist
                best_k = k
        return best_k

    def _ComputeClusterMean(self, cluster_indices: list[int], data: list[list[float]], dim: int) -> list[float]:
        """Helper function to compute the mean of a cluster."""
        if len(cluster_indices) == 0:
            return None

        cluster_points = [data[i] for i in cluster_indices]
        mean = [0.0] * dim

        for point in cluster_points:
            for d in range(dim):
                mean[d] += point[d]

        for d in range(dim):
            mean[d] /= len(cluster_points)

        return mean

    # Run K-means clustering algorithm.
    #   https://en.wikipedia.org/wiki/K-means_clustering
    # data : Training data points (subvectors).
    # K : Number of clusters (centroids).
    # max_iter : Optional (default: 100), maximum number of iterations.
    # return: K centroids after convergence
    def _KMeans(self, data: list[list[float]], K: int, max_iter: int = 100) -> list[list[float]]:
        if len(data) == 0:
            return []

        # Initialize centroids using K-means++
        centroids = self._KMeansPlusPlus(data, K)
        converged = False

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            while max_iter > 0 and not converged:
                max_iter -= 1

                # Parallelize point assignment to nearest centroid
                assignments = list(executor.map(lambda point, c=centroids: self._FindNearestCentroid(point, c), data))

                # Create clusters based on assignments
                clusters = [[] for _ in range(K)]
                for i, best_k in enumerate(assignments):
                    clusters[best_k].append(i)

                # Parallelize centroid computation
                # Submit cluster mean computations in parallel
                future_to_k = {}
                for k in range(K):
                    if len(clusters[k]) > 0:
                        future = executor.submit(self._ComputeClusterMean, clusters[k], data, self.D_)
                        future_to_k[future] = k

                # Collect results
                new_centroids = [None] * K
                for future, k in future_to_k.items():
                    new_centroids[k] = future.result()

                # Fill in empty clusters with old centroids
                for k in range(K):
                    if new_centroids[k] is None:
                        new_centroids[k] = centroids[k][:]

                # Check convergence: stop if ALL centroids have converged
                # (i.e., no centroid changed significantly)
                tmp_converged = True
                for k in range(K):
                    if self._L2Sqr(centroids[k], new_centroids[k]) > 1e-6:
                        tmp_converged = False
                        break

                if tmp_converged:
                    converged = True
                else:
                    centroids = new_centroids
        return centroids

    def _TrainSubspace(self, m: int, data_sample: list[list[float]]) -> list[list[float]]:
        """Helper function to train k-means for a single subspace."""
        # Extract subvectors for subspace m from all training vectors
        subvectors = []
        for vec in data_sample:
            subvec = vec[m * self.D_ : (m + 1) * self.D_]
            subvectors.append(subvec)

        # Run K-means to find K representative centroids for this subspace
        centroids = self._KMeans(subvectors, self.K)
        return centroids

    # Train the product quantizer by learning codebooks from data.
    #   Training time: O(iterations * M * K * N * D'), where N is number of samples.
    #   Typically use 10K-100K samples for training.
    # data_sample: Training vectors, each of dimension D.
    def TrainPQ(self, data_sample: list[list[float]]) -> None:
        if len(data_sample) == 0:
            raise ValueError("Training data cannot be empty")

        if len(data_sample[0]) != self.D:
            raise ValueError(f"Expected vectors of dimension {self.D}, got {len(data_sample[0])}")

        # Parallelize training across subspaces
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            # Submit training for each subspace in parallel
            future_to_m = {executor.submit(self._TrainSubspace, m, data_sample): m for m in range(self.M)}

            # Collect results
            for future in future_to_m:
                m = future_to_m[future]
                self.codebooks[m] = future.result()

        self.trained = True

    # Encode a vector as M indices (one per subspace).
    # Each index represents the nearest centroid in that subspace's codebook.
    # vec: Vector to encode, of dimension D.
    # returns: M indices, each in range [0, K-1].
    def Encode(self, vec: list[float]) -> list[int]:
        if not self.trained:
            raise ValueError("Product quantizer must be trained before encoding")

        # Split vector into M subvectors
        subvectors = self._SplitVector(vec)

        # For each subspace, find nearest centroid and store its index
        code = []
        for m in range(self.M):
            subvec = subvectors[m]
            codebook_m = self.codebooks[m]

            # Find nearest centroid in codebook m
            min_dist = float("inf")
            best_idx = 0
            for k, centroid in enumerate(codebook_m):
                dist = self._L2Sqr(subvec, centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = k

            code.append(best_idx)

        return code

    # Decode indices back to approximate original vector.
    # Reconstructs the vector by concatenating the centroids specified by each index.
    # code: M indices, each in range [0, K-1].
    # returns: Reconstructed D-dimensional vector.
    def Decode(self, code: list[int]) -> list[float]:
        if not self.trained:
            raise ValueError("Product quantizer must be trained before decoding")

        # Reconstruct vector by concatenating centroids
        reconstructed = []
        for m in range(self.M):
            idx = code[m]
            centroid = self.codebooks[m][idx]
            reconstructed.extend(centroid)

        return reconstructed

    # Compute approximate distance from query to encoded vector.
    # query: Query vector of dimension D.
    # code: Encoded vector (M indices) from codebook.
    # returns: Approximate squared L2 distance.
    def ComputeDistance(self, query: list[float], code: list[int]) -> float:
        if not self.trained:
            raise ValueError("Product quantizer must be trained before computing distances")

        # Split query into M subvectors
        query_subvecs = self._SplitVector(query)

        # Compute distance by summing distances in each subspace
        # Only compute the M distances we need (not the full M×K table)
        total_distance = 0.0
        for m in range(self.M):
            query_subvec = query_subvecs[m]
            centroid_idx = code[m]
            centroid = self.codebooks[m][centroid_idx]

            # Distance in subspace m to the specific centroid
            dist = self._L2Sqr(query_subvec, centroid)
            total_distance += dist

        return total_distance

    # Compute symmetric distance between two encoded vectors.
    # This is centroid-to-centroid distance
    # code_v: First encoded vector (M indices).
    # code_w: Second encoded vector (M indices).
    # returns: Approximate squared L2 distance.
    def ComputeSymmetricDistance(self, code_v: list[int], code_w: list[int]) -> float:
        if not self.trained:
            raise ValueError("Product quantizer must be trained before computing distances")

        # Compute distance by summing centroid-to-centroid distances in each subspace
        total_distance = 0.0
        for m in range(self.M):
            centroid_v = self.codebooks[m][code_v[m]]
            centroid_w = self.codebooks[m][code_w[m]]

            # Distance between centroids in subspace m
            dist = self._L2Sqr(centroid_v, centroid_w)
            total_distance += dist

        return total_distance

    # Set codebooks directly (e.g., from a saved model).
    # codebooks: Codebooks to use, shape (M, K, D').
    def SetCodebooks(self, codebooks: list[list[list[float]]]) -> None:
        if len(codebooks) != self.M:
            raise ValueError(f"Expected {self.M} codebooks, got {len(codebooks)}")

        for m, codebook_m in enumerate(codebooks):
            if len(codebook_m) != self.K:
                raise ValueError(f"Expected {self.K} centroids in codebook {m}, got {len(codebook_m)}")

            for k, centroid in enumerate(codebook_m):
                if len(centroid) != self.D_:
                    raise ValueError(
                        f"Expected centroids of dimension {self.D_} in codebook {m}, "
                        f"got {len(centroid)} for centroid {k}"
                    )

        self.codebooks = codebooks
        self.trained = True

    # Get the trained codebooks.
    # returns: Codebooks, shape (M, K, D').
    def GetCodebooks(self) -> list[list[list[float]]]:
        if not self.trained:
            raise ValueError("Product quantizer must be trained before accessing codebooks")

        return self.codebooks
