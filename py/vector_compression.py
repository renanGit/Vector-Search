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
        """Train the compression model on data."""
        pass

    @abstractmethod
    def Encode(self, vec: list[float]) -> any:
        """Encode a vector into compressed representation."""
        pass

    @abstractmethod
    def Decode(self, code: any) -> list[float]:
        """Decode compressed representation back to vector."""
        pass

    @abstractmethod
    def ComputeDistance(self, query: list[float], code: any) -> float:
        """
        Compute asymmetric distance from query vector to encoded vector.

        This is the standard query-to-database distance used during search.
        """
        pass

    @abstractmethod
    def ComputeSymmetricDistance(self, code_v: any, code_w: any) -> float:
        """
        Compute symmetric distance between two encoded vectors.

        Used during graph construction when comparing database vectors to each other.
        For PQ, this is centroid-to-centroid distance, NOT decode-then-L2.
        """
        pass

    @abstractmethod
    def IsTrained(self) -> bool:
        """Check if the compression model is trained."""
        pass

    @abstractmethod
    def SetCodebooks(self, codebooks: any) -> None:
        """Set codebooks directly (e.g., from a saved model)."""
        pass

    @abstractmethod
    def GetCodebooks(self) -> any:
        """Get the trained codebooks."""
        pass


class PQCompression(VectorCompression):
    """
    Product Quantization compression adapter.

    Parameters
    ----------
    M : int
        Number of subquantizers (subspaces). Dimension must be divisable by M.
    K : int
        Number of centroids per subspace
    D : int
        Dimensionality of vectors
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(self, M: int, K: int, D: int, seed: int = 42):
        from pq import ProductQuantizer

        self.pq = ProductQuantizer(M=M, K=K, D=D, seed=seed)

    def Train(self, data: list[list[float]]) -> None:
        self.pq.TrainPQ(data)

    def Encode(self, vec: list[float]) -> list[int]:
        return self.pq.Encode(vec)

    def Decode(self, code: list[int]) -> list[float]:
        return self.pq.Decode(code)

    def ComputeDistance(self, query: list[float], code: list[int]) -> float:
        return self.pq.ComputeDistance(query, code)

    def ComputeSymmetricDistance(self, code_v: list[int], code_w: list[int]) -> float:
        return self.pq.ComputeSymmetricDistance(code_v, code_w)

    def IsTrained(self) -> bool:
        return self.pq.trained

    def SetCodebooks(self, codebooks: list[list[list[float]]]) -> None:
        self.pq.SetCodebooks(codebooks)

    def GetCodebooks(self) -> list[list[list[float]]]:
        return self.pq.GetCodebooks()
