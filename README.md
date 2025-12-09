# Vector Search
My goal here is to try and implement different ANN algos without using specialized external libraries (ie. via pip install) and make them as simple as possible to read.

For obvious reasons python versions will be slow (just for readability).

## HNSW
I've implemented a python single thread version of hnsw. Made a small optimization to use lru_cache to retrieve precomputed distance to node during insertion.

Used the following to implement hnsw.
Git: https://github.com/nmslib/hnswlib/tree/master
Paper: https://arxiv.org/pdf/1603.09320

## PQ
I've implemented a python multithreaded product quantization.

The underlying code uses kmeans++ for better selection of centroids and kmeans for clustering and etc...
