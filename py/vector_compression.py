from abc import ABC, abstractmethod


# Abstract interface for vector compression techniques
class VectorCompression(ABC):
    """
    Abstract base class for vector compression techniques.

    Subclasses implement specific compression methods (PQ, OPQ, SQ, etc.)
    and provide encode/decode/distance computation functionality.
    """

    @abstractmethod
    def Train(self, data: list[list[float]]) -> None:
        """Train the compression model on data.

        Parameters:
            data (list[list[float]]): Training data vectors"""
        pass

    @abstractmethod
    def Encode(self, vec: list[float]) -> any:
        """Encode a vector into compressed representation.

        Parameters:
            vec (list[float]): Vector to encode

        Returns:
            any: Compressed representation"""
        pass

    @abstractmethod
    def Decode(self, code: any) -> list[float]:
        """Decode compressed representation back to vector.

        Parameters:
            code (any): Compressed representation

        Returns:
            list[float]: Decoded vector"""
        pass

    @abstractmethod
    def ComputeAsymmetricDistance(self, query: list[float], code: any) -> float:
        """Compute asymmetric distance from query vector to encoded vector.

        This is the standard query-to-database distance used during search.

        Parameters:
            query (list[float]): Query vector
            code (any): Encoded vector

        Returns:
            float: Distance between query and encoded vector"""
        pass

    @abstractmethod
    def ComputeSymmetricDistance(self, code_v: any, code_w: any) -> float:
        """Compute symmetric distance between two encoded vectors.

        Used during graph construction when comparing database vectors to each other.
        For PQ, this is centroid-to-centroid distance, NOT decode-then-L2.

        Parameters:
            code_v (any): First encoded vector
            code_w (any): Second encoded vector

        Returns:
            float: Distance between the two encoded vectors"""
        pass

    @abstractmethod
    def IsTrained(self) -> bool:
        """Check if the compression model is trained.

        Returns:
            bool: True if trained, False otherwise"""
        pass

    @abstractmethod
    def SetCodebooks(self, codebooks: any) -> None:
        """Set codebooks directly (e.g., from a saved model).

        Parameters:
            codebooks (any): Codebooks to set"""
        pass

    @abstractmethod
    def GetCodebooks(self) -> any:
        """Get the trained codebooks.

        Returns:
            any: The trained codebooks"""
        pass


class PQCompression(VectorCompression):
    """Product Quantization compression adapter.

    Parameters
        M (int): Number of subquantizers (subspaces). Dimension must be divisable by M.
        K (int): Number of centroids per subspace
        D (int): Dimensionality of vectors
        seed (int, optional): Random seed for reproducibility
        n_threads (int, optional): Threads used during training"""

    def __init__(self, M: int, K: int, D: int, seed: int = 42, n_threads: int = 16):
        from pq import ProductQuantizer

        self.pq = ProductQuantizer(M=M, K=K, D=D, seed=seed, n_threads=n_threads)

    def Train(self, data: list[list[float]]) -> None:
        self.pq.TrainPQ(data)

    def Encode(self, vec: list[float]) -> list[int]:
        return self.pq.Encode(vec)

    def Decode(self, code: list[int]) -> list[float]:
        return self.pq.Decode(code)

    def ComputeAsymmetricDistance(self, query: list[float], code: list[int]) -> float:
        return self.pq.ComputeAsymmetricDistance(query, code)

    def ComputeSymmetricDistance(self, code_v: list[int], code_w: list[int]) -> float:
        return self.pq.ComputeSymmetricDistance(code_v, code_w)

    def IsTrained(self) -> bool:
        return self.pq.trained

    def SetCodebooks(self, codebooks: list[list[list[float]]]) -> None:
        self.pq.SetCodebooks(codebooks)

    def GetCodebooks(self) -> list[list[list[float]]]:
        return self.pq.GetCodebooks()
