#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "cluster_cuda.cuh"

using namespace std;

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
      gpuErrChk(cudaEventCreate(&start));       \
      gpuErrChk(cudaEventCreate(&stop));        \
      gpuErrChk(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrChk(cudaEventRecord(stop));                     \
      gpuErrChk(cudaEventSynchronize(stop));                \
      gpuErrChk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrChk(cudaEventDestroy(start));                   \
      gpuErrChk(cudaEventDestroy(stop));                    \
  }

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code

// Fills output with standard normal data
void gaussianFill(float *output, int size) {
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0.0, 1.0);
  for (int i=0; i < size; i++) {
    output[i] = distribution(generator);
  }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM floats.
void readLSAReview(string review_str, float *output) {
  stringstream stream(review_str);
  int component_idx = 0;

  for (string component; getline(stream, component, ','); component_idx++) {
    output[component_idx] = atof(component.c_str());
  }
  assert(component_idx == REVIEW_DIM);
}

// used to pass arguments to printerCallback
struct printerArg {
  int review_idx_start;
  int batch_size;
  int *cluster_assignments;
};

// Prints out which cluster each review in a batch was assigned to.
// DONE: Call with cudaStreamAddCallback (after completing D->H memcpy)
void printerCallback(cudaStream_t stream, cudaError_t status, void *userData) {
  printerArg *arg = static_cast<printerArg *>(userData);

  for (int i=0; i < arg->batch_size; i++) {
    printf("%d: %d\n", 
	   arg->review_idx_start + i, 
	   arg->cluster_assignments[i]);
  }

  delete arg;
}

void cluster(istream& in_stream, int k, int batch_size) {
  // cluster centers
  float *d_clusters;

  // how many points lie in each cluster
  int *d_cluster_counts;

  // allocate memory for cluster centers and counts
  gpuErrChk(cudaMalloc(&d_clusters, k * REVIEW_DIM * sizeof(float)));
  gpuErrChk(cudaMalloc(&d_cluster_counts, k * sizeof(int)));

  // randomly initialize cluster centers
  float *clusters = new float[k * REVIEW_DIM];
  gaussianFill(clusters, k * REVIEW_DIM);
  gpuErrChk(cudaMemcpy(d_clusters, clusters, k * REVIEW_DIM * sizeof(float),
		       cudaMemcpyHostToDevice));

  // initialize cluster counts to 0
  gpuErrChk(cudaMemset(d_cluster_counts, 0, k * sizeof(int)));
  
  // DONE: allocate copy buffers and streams

  /* Allocating pinned memory for data */
  // Allocating host buffer
  float *h_data0;
  float *h_data1;
  gpuErrChk(cudaMallocHost((void **) &h_data0,
    REVIEW_DIM * batch_size * sizeof(float)));
  gpuErrChk(cudaMallocHost((void **) &h_data1,
    REVIEW_DIM * batch_size * sizeof(float)));

  // Establishing an array of the host buffer
  float *h_array[2];
  h_array[0] = h_data0;
  h_array[1] = h_data1; 

  /* Allocating device input buffer */
  float *d_data0;
  float *d_data1;
  gpuErrChk(cudaMalloc((void **) &d_data0,
    REVIEW_DIM * batch_size * sizeof(float)));
  gpuErrChk(cudaMalloc((void **) &d_data1,
    REVIEW_DIM * batch_size * sizeof(float)));

  // Establishing an array of the device input buffer
  float *d_array[2];
  d_array[0] = d_data0;
  d_array[1] = d_data1;

  /* Allocating device output buffer */
  // Creating output arrays
  int *d_out0;
  int *d_out1;
  gpuErrChk(cudaMalloc((void **) &d_out0, batch_size * sizeof(int)));
  gpuErrChk(cudaMalloc((void **) &d_out1, batch_size * sizeof(int)));

  // Zero'ing the arrays
  cudaMemset(d_out0, 0, batch_size * sizeof(int));
  cudaMemset(d_out1, 0, batch_size * sizeof(int));

  // Establishing an array of the device output buffer
  int *d_outarray[2];
  d_outarray[0] = d_out0;
  d_outarray[1] = d_out1;

  /* Allocating host output data buffer */
  int h_out0[batch_size];
  int h_out1[batch_size];

  // Establishing an array of the host output buffer
  int *h_outarray[2];
  h_outarray[0] = &h_out0[0];
  h_outarray[1] = &h_out1[0];

  /* Creating data streams */
  cudaStream_t stream[2];
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);

  // main loop to process input lines (each line corresponds to a review)
  int review_idx = 0;
  // initialize the batchID variable to identify which buffer to use
  int batchID;

  // initialize timers
  float batch_latency = -1;
  float throughput = -1;

  // Creating events for analyzing the program
  cudaEvent_t batchcopy;
  cudaEventCreate(&batchcopy);
  cudaEvent_t throughputstart;
  cudaEventCreate(&throughputstart);
  cudaEvent_t throughputstop;
  cudaEventCreate(&throughputstop);

  for (string review_str; getline(in_stream, review_str); review_idx++) {

    // DONE: readLSAReview into appropriate storage

    // batchID determines which buffer to use
    batchID = ((review_idx + 1) / batch_size) % 2;
    // Check if batch is not filled
    if ((review_idx + 1) % batch_size != 0){
      // Start recording for latency of classifying a batch
      START_TIMER();

      readLSAReview(review_str, h_array[batchID] +
        ((review_idx % batch_size) * REVIEW_DIM)); // Offset for each review
    }
    else {
      // End recording for latency of classifying a batch
      STOP_RECORD_TIMER(batch_latency);

      // DONE: if you have filled up a batch, copy H->D, kernel, copy D->H,
      //       and set callback to printerCallback. Will need to allocate
      //       printerArg struct. Do all of this in a stream.

      // Synchronize the memcopy
      // Make sure previous memcopy happened before doing another
      cudaEventSynchronize(batchcopy);

      // Start recording for throughput
      cudaEventRecord(throughputstart, stream[batchID]);

      // Copy from H -> D
      cudaMemcpyAsync(d_array[batchID], h_array[batchID],
        REVIEW_DIM * batch_size * sizeof(float), cudaMemcpyHostToDevice,
        stream[batchID]);

      // Memcopy event
      cudaEventRecord(batchcopy, stream[batchID]);

      // Kernel call
      cudaCluster(d_clusters, d_cluster_counts, k, d_array[batchID],
        d_outarray[batchID], batch_size, stream[batchID]);

      // Copy from D -> H
      cudaMemcpyAsync(h_outarray[batchID], d_outarray[batchID],
        batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream[batchID]);

      // Allocate printerArg struct
      struct printerArg *pA_output = new struct printerArg;
      pA_output -> review_idx_start = review_idx;
      pA_output -> batch_size = batch_size;
      pA_output -> cluster_assignments = h_outarray[batchID];

      // Setting callback to printerCallback
      cudaStreamAddCallback(stream[batchID], printerCallback,
        pA_output, 0);

      // Stop recording for throughput
      cudaEventRecord(throughputstop, stream[batchID]);
      cudaEventSynchronize(throughputstart);
      cudaEventSynchronize(throughputstop);
      cudaEventElapsedTime(&throughput, throughputstart, throughputstop);
      cudaEventDestroy(throughputstart);
      cudaEventDestroy(throughputstop);
    }
  }

  // wait for everything to end on GPU before final summary
  gpuErrChk(cudaDeviceSynchronize());

  // retrieve final cluster locations and counts
  int *cluster_counts = new int[k];
  gpuErrChk(cudaMemcpy(cluster_counts, d_cluster_counts, k * sizeof(int), 
		       cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(clusters, d_clusters, k * REVIEW_DIM * sizeof(int),
		       cudaMemcpyDeviceToHost));

  // print cluster summaries
  for (int i=0; i < k; i++) {
    printf("Cluster %d, population %d\n", i, cluster_counts[i]);
    printf("[");
    for (int j=0; j < REVIEW_DIM; j++) {
      printf("%.4e,", clusters[i * REVIEW_DIM + j]);
    }
    printf("]\n\n");
  }

  // free cluster data
  gpuErrChk(cudaFree(d_clusters));
  gpuErrChk(cudaFree(d_cluster_counts));
  delete[] cluster_counts;
  delete[] clusters;

  // DONE: finish freeing memory, destroy streams
  gpuErrChk(cudaFreeHost(h_data0));
  gpuErrChk(cudaFreeHost(h_data1));
  gpuErrChk(cudaFree(d_data0));
  gpuErrChk(cudaFree(d_data1));
  gpuErrChk(cudaFree(d_out0));
  gpuErrChk(cudaFree(d_out1));
  gpuErrChk(cudaStreamDestroy(stream[0]));
  gpuErrChk(cudaStreamDestroy(stream[1]));

  // Destroying Events
  gpuErrChk(cudaEventDestroy(batchcopy));

  // Print timer results
  printf("Latency of classifying a batch: %f ms\n", batch_latency);
  printf("Latency of processing 1 batch: %f ms\n", throughput);

}

int main(int argc, char** argv) {
  int k = 50;
  int batch_size = 32;

  if (argc == 1) {
    cluster(cin, k, batch_size);
  } else if (argc == 2) {
    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    cluster(buffer, k, batch_size);
  }
}
