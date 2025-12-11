# Product Quantization (PQ)

## Overview

**What is Quantization?**

Quantization is the process of mapping a large set of continuous values to a smaller set of discrete values. Think of it as "rounding" or "bucketing" values to reduce precision and save space.

**Product Quantization** is a lossy compression technique for high-dimensional vectors that enables efficient similarity search while significantly reducing memory usage. It was introduced by Jégou et al. in 2011 and has become a fundamental technique in large-scale vector search systems.

The "Product" part means we quantize **multiple subspaces independently** and combine (take the Cartesian product of) their codebooks.

**Key Paper:** Jégou, H., Douze, M., & Schmid, C. (2011). "Product quantization for nearest neighbor search." IEEE transactions on pattern analysis and machine intelligence, 33(1), 117-128.

## The Problem

Consider storing 1 billion 128-dimensional vectors where each dimension is a 32-bit float:
- Memory required: 1B × 128 × 4 bytes = **512 GB**

This is prohibitively expensive. Product Quantization can reduce this to:
- Memory required: 1B × 8 bytes = **8 GB** (64× reduction!)

## Core Concepts

### 1. Vector Decomposition

**Why split vectors into subspaces?**

By splitting the vector, we exploit the **product structure**: the Cartesian product of M small codebooks gives us an exponentially large effective codebook (K^M combinations) while only storing M×K centroids.

Product Quantization splits each vector into **M** disjoint subvectors:

Consider quantizing a 128D vector with K=256 centroids:
- **Full space approach:** Need 256^128 possible codewords (impossibly large!)
- **Product approach (M=8):** Need 8 codebooks × 256 centroids = only 2,048 centroids total

```
Original vector (D dimensions):
[x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈]

Split into M=4 subvectors (D'=D/M=2 dimensions each):
subvector₁: [x₁, x₂]  ┐
subvector₂: [x₃, x₄]  ├─ Each quantized independently
subvector₃: [x₅, x₆]  ├─ with its own K=256 codebook
subvector₄: [x₇, x₈]  ┘

Total representations: 256^4 = 4.3 billion combinations
Total storage: 4 × 256 × 2 = 2,048 values (not 4.3 billion!)
```

### 2. Codebook Creation

For each subspace, we create a **codebook** of K representative vectors (centroids) using K-means clustering:

```
Codebook for subspace 1 (K=256 centroids):
c₁⁽¹⁾ = [0.5, 0.3]
c₂⁽¹⁾ = [0.1, 0.9]
...
c₂₅₆⁽¹⁾ = [0.8, 0.2]

Similarly for subspaces 2, 3, 4...
```

### 3. Vector Encoding

Each subvector is replaced by the **index** of its nearest centroid:

```
Original subvector: [0.52, 0.31]
Nearest centroid: c₁⁽¹⁾ = [0.5, 0.3] (index = 0)

Encoded: 0 (1 byte if K=256)
```

A full vector becomes M indices:

```
Original: [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈]  (32 bytes)
Encoded:  [0, 42, 17, 203]                   (4 bytes with M=4, K=256)
```

### 4. Distance Computation

There are two types of PQ distance computation:

#### 4a. Asymmetric Distance (Query to Database)

To compute the distance between a query vector q and an encoded database vector:

1. Split query into M subvectors: q₁, q₂, ..., qₘ
2. For each subspace m, compute distance from qₘ to the centroid specified by the code
3. Sum the M subspace distances

```python
# Asymmetric distance (query to code)
def ComputeDistance(query, code):
    query_subvecs = split_into_M_parts(query)
    total_distance = 0
    for m in range(M):
        centroid = codebooks[m][code[m]]
        dist = L2(query_subvecs[m], centroid)
        total_distance += dist
    return total_distance
```

#### 4b. Symmetric Distance (Database to Database)

To compute the distance between two encoded database vectors:

1. For each subspace m, compute distance between the two centroids specified by the codes
2. Sum the M centroid-to-centroid distances

```python
# Symmetric distance (code to code)
def ComputeSymmetricDistance(code_v, code_w):
    total_distance = 0
    for m in range(M):
        centroid_v = codebooks[m][code_v[m]]
        centroid_w = codebooks[m][code_w[m]]
        dist = L2(centroid_v, centroid_w)
        total_distance += dist
    return total_distance
```

## Detailed Example

### Example 1: Simple 4D Vector

```
Original vector: [0.8, 0.3, 0.1, 0.9]

Step 1: Split into M=2 subvectors (D'=2 each)
  subvector₁: [0.8, 0.3]
  subvector₂: [0.1, 0.9]

Step 2: Find nearest centroids (assume K=4 centroids per subspace)
  Codebook 1:
    c₀: [0.1, 0.1]  distance: 0.73
    c₁: [0.8, 0.3]  distance: 0.00  ← nearest!
    c₂: [0.5, 0.5]  distance: 0.13
    c₃: [0.2, 0.8]  distance: 0.61

  Codebook 2:
    c₀: [0.9, 0.1]  distance: 0.89
    c₁: [0.1, 0.9]  distance: 0.00  ← nearest!
    c₂: [0.5, 0.5]  distance: 0.32
    c₃: [0.2, 0.3]  distance: 0.37

Step 3: Encode as indices
  Encoded vector: [1, 1]  (2 bytes instead of 16 bytes!)

Step 4: Query distance computation
  Query: [0.7, 0.4, 0.2, 0.8]
  Split: q₁=[0.7, 0.4], q₂=[0.2, 0.8]

  Precompute distance table 1:
    d(q₁, c₀) = 0.73
    d(q₁, c₁) = 0.11  ← we'll use this
    d(q₁, c₂) = 0.22
    d(q₁, c₃) = 0.66

  Precompute distance table 2:
    d(q₂, c₀) = 0.85
    d(q₂, c₁) = 0.11  ← we'll use this
    d(q₂, c₂) = 0.41
    d(q₂, c₃) = 0.50

  Approximate distance = 0.11 + 0.11 = 0.22
  (True L2 distance would be ~0.24)
```

## Parameters

### M (Number of Subquantizers)

- **Typical values:** 8, 16, 32, 64
- **Tradeoff:**
  - Larger M → Better accuracy (finer granularity)
  - Larger M → More storage (more indices to store)
  - M must divide D evenly (D mod M = 0)

### K (Number of Centroids per Subspace)

- **Typical value:** 256 (fits in 1 byte)
- **Tradeoff:**
  - Larger K → Better accuracy (more representative centroids)
  - Larger K → More memory for codebooks
  - Larger K → Slower distance table precomputation

### D' (Subvector Dimensionality)

- **Formula:** D' = D / M
- **Typical values:** 4, 8, 16
- **Recommendation:** Keep D' ≥ 4 for good clustering quality

## Advantages

1. **Memory Efficiency:** 64-256× compression ratio
2. **Fast Search:** Distance computation is just M table lookups
3. **Scalability:** Works well with billions of vectors
4. **Integration:** Combines naturally with graph-based indexes (HNSW, IVF)

## Disadvantages

1. **Lossy Compression:** Distances are approximate
2. **Training Required:** Need to run K-means on sample data
3. **Quantization Error:** Accuracy decreases with higher compression
4. **Limited to L2:** Works best with Euclidean distance

## Accuracy Analysis

The quantization error depends on how well centroids represent the data:

```
Quantization error = Σ min ||x_i^(m) - c_k^(m)||²
```

## Integration with HNSW

Product Quantization integrates with HNSW through pre-training:

1. **Sample training data** from your vector dataset
2. **Train PQ** on the sample (learn codebooks)
3. **Pass trained PQ to HNSW** at construction time
4. **During Insert**: Vectors are encoded on-the-fly and stored as codes
5. **During Search**: PQ symmetric/asymmetric distances are used

**Key principle**: PQ is trained OUTSIDE of HNSW, then passed as a pre-trained object.

```python
# Step 1: Sample and train PQ
train_sample = vectors[:10000]  # Sample from your data
pq = PQCompression(M=8, K=256, D=128)
pq.Train(train_sample)

# Step 2: Create HNSW with pre-trained PQ
index = HNSWIndex(M=16, ef_construction=200, compression=pq)

# Step 3: Insert vectors (compressed automatically)
for vec in all_vectors:
    index.Insert(vec)  # Encoded to PQ codes internally
```

### Memory Comparison (1M vectors, D=128)

| Method | Memory |
|--------|--------|
| HNSW alone | ~512 MB |
| HNSW + PQ (M=8, K=256) | ~72 MB |
| HNSW + PQ (M=16, K=256) | ~80 MB |

## Advanced Variations

### Optimized Product Quantization (OPQ)

Applies a learned rotation matrix before splitting vectors to minimize quantization error.

### Additive Quantization (AQ)

Approximates vectors as sum of multiple codewords instead of Cartesian product.

### Polysemous Codes

Uses Hamming distance on PQ codes for initial filtering before computing accurate distances.

## Implementation Considerations

### Training Data

- Use 10K-100K random samples from your dataset
- Run K-means with K=256 for each subspace
- Initialize K-means with k-means++ for better convergence

### Distance Metrics

- L2 (Euclidean) distance is standard
- Inner product can be supported with normalization
- Asymmetric Distance Computation (ADC): quantize database but not queries

### Hardware Optimization

- SIMD instructions can accelerate table lookups
- Cache locality: store indices contiguously
- GPU acceleration: parallel distance table computation

## References

1. **Original Paper:** Jégou, H., Douze, M., & Schmid, C. (2011). "Product quantization for nearest neighbor search."

2. **Optimized PQ:** Ge, T., He, K., Ke, Q., & Sun, J. (2013). "Optimized product quantization."

3. **Polysemous Codes:** Douze, M., Jégou, H., & Perronnin, F. (2016). "Polysemous codes."

4. **Survey:** Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-scale similarity search with GPUs."

## Example Use Cases

1. **Image Search:** Compress CNN features (2048D → 32 bytes)
2. **Recommendation Systems:** Store user/item embeddings efficiently
3. **Document Retrieval:** Compress BERT embeddings (768D → 16 bytes)
4. **Video Search:** Store frame features for billion-video libraries

## Summary

Product Quantization is a powerful technique that enables:
- **64-256× memory reduction**
- **10-100× faster search** (due to cache efficiency)
- **Scalability to billions of vectors**
- **Minimal accuracy loss** (2-5% recall drop with proper tuning)

It's particularly effective when combined with graph-based indexes like HNSW for state-of-the-art performance on large-scale vector search.
