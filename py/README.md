# Python Vector Search Implementation

Learning material for vector search implementations in Python (with no third-party libs, just standard libs), including HNSW (Hierarchical Navigable Small World), PQ, and brute force algorithm.

The python code here is by no means for performance, its to learn the algo without all the specialized libs that hide the implementation.

## Quick Start

### Setup

1. **Create virtual environment:**
   ```bash
   cd py
   python -m venv venv
   ```

2. **Activate virtual environment:**

   **Windows:**
   ```bash
   venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies (for development):**
   ```bash
   pip install -r requirements-dev.txt
   ```

### Usage

```python
from hnsw import HNSWIndex

# Create index
index = HNSWIndex(M=16, ef_construction=200)

# Insert vectors
vectors = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
]

for vec in vectors:
    index.Insert(vec)

# Search for nearest neighbors
query = [2.0, 3.0, 4.0]
results = index.KNNSearch(query, topK=5)

# Results are tuples of (distance, index)
for dist, idx in results:
    print(f"Vector {idx}: distance {dist}")
```

## Implementations

### HNSW (Hierarchical Navigable Small World)

Fast approximate nearest neighbor search with excellent recall.

**File:** `hnsw.py`

**Parameters:**
- `M`: Number of connections per element
- `ef_construction`: Size of dynamic candidate list during construction
- `ef_search`: Size of dynamic candidate list during search (default: 200)

### Product Quantization (PQ)

Lossy compression technique for high-dimensional vectors that enables 64-256× memory reduction with minimal accuracy loss.

**File:** `pq.py`

**Usage:**
```python
from pq import ProductQuantizer

# Create and train quantizer
pq = ProductQuantizer(M=8, K=256, D=128)
pq.TrainPQ(training_vectors)

# Compress a vector
code = pq.Encode(vector)  # 128D float vector → 8 bytes

# Decode back to approximate vector
reconstructed = pq.Decode(code)

# Distance computation
distance = pq.ComputeDistance(query, code)  # Asymmetric: query to code
sym_dist = pq.ComputeSymmetricDistance(code1, code2)  # Symmetric: code to code
```

**Parameters:**
- `M`: Number of subquantizers (must divide D evenly)
- `K`: Number of centroids per subspace (typically 256)
- `D`: Vector dimensionality

### HNSW + PQ

Combine HNSW's fast graph search with PQ's memory efficiency.

**Pre-train PQ and pass to HNSW:**
```python
from hnsw import HNSWIndex
from vector_compression import PQCompression

# 1. Sample training data from your dataset
training_data = vectors[:10000]  # Use subset for training

# 2. Train PQ on the sample
pq_compression = PQCompression(M=8, K=256, D=128)
pq_compression.Train(training_data)

# 3. Create HNSW with pre-trained compression
index = HNSWIndex(M=16, ef_construction=200, compression=pq_compression)

# 4. Insert vectors (automatically compressed during insertion)
for vec in vectors:
    index.Insert(vec)

# 5. Search (uses PQ distances automatically)
results = index.KNNSearch(query, topK=10)
```

**Save and load trained codebooks:**
```python
# Save codebooks
codebooks = pq_compression.GetCodebooks()
# ... save codebooks to disk ...

# Load codebooks later
pq_compression = PQCompression(M=8, K=256, D=128)
pq_compression.SetCodebooks(codebooks)
index = HNSWIndex(M=16, ef_construction=200, compression=pq_compression)
```

### Brute Force

Exact nearest neighbor search (slower but 100% accurate).

**File:** `bruteforce.py`

## Testing

Run all tests:
```bash
python -m unittest discover -s . -p "test_*.py" -v
```

Run with coverage:
```bash
pytest test_hnsw.py -v --cov=. --cov-report=term
```

## Requirements

- Python 3.12+
- No external dependencies for core functionality
- Development dependencies in `requirements-dev.txt`
