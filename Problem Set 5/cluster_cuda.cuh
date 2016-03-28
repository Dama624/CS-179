#ifndef CUDA_CLUSTER_CUH
#define CUDA_CLUSTER_CUH

#define REVIEW_DIM 50

void cudaCluster(float *clusters, int *cluster_counts, int k,
		 float *data, int *output, int batch_size, 
		 cudaStream_t stream);

#endif
