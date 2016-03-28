#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "classify_cuda.cuh"

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
  // seed generator to 2015
  std::default_random_engine generator(2015);
  std::normal_distribution<float> distribution(0.0, 0.1);
  for (int i=0; i < size; i++) {
    output[i] = distribution(generator);
  }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM + 1 floats.
void readLSAReview(string review_str, float *output, int stride) {
  stringstream stream(review_str);
  int component_idx = 0;

  for (string component; getline(stream, component, ','); component_idx++) {
    output[stride * component_idx] = atof(component.c_str());
  }
  assert(component_idx == REVIEW_DIM + 1);
}


void classify(istream& in_stream, int batch_size) {

  // grad is the value of (-1 / batch_size) * the sum grad for all
  // points in the batch
  float *grad;
  gpuErrChk(cudaMalloc(&grad, (REVIEW_DIM + 1) * batch_size * sizeof(float)));

  // num is the numerator of the grad point calculation
  float *num;
  gpuErrChk(cudaMalloc(&num, REVIEW_DIM * batch_size * sizeof(float)));

  // DONE: randomly initialize weights, allocate and initialize buffers on
  //       host & device

  // Initialize weights
  float *weights = new float[REVIEW_DIM];
  gaussianFill(weights, REVIEW_DIM);

  /* Allocating pinned memory for data */
  // Allocating host buffer
  float *h_data0;
  float *h_data1;
  gpuErrChk(cudaMallocHost((void **) &h_data0,
    (REVIEW_DIM + 1) * batch_size * sizeof(float)));
  gpuErrChk(cudaMallocHost((void **) &h_data1,
    (REVIEW_DIM + 1) * batch_size * sizeof(float)));

  // Establishing an array of the host buffer
  float *h_array[2];
  h_array[0] = h_data0;
  h_array[1] = h_data1; 

  /* Allocating device input buffer */
  float *d_data0;
  float *d_data1;
  gpuErrChk(cudaMalloc((void **) &d_data0,
    (REVIEW_DIM + 1) * batch_size * sizeof(float)));
  gpuErrChk(cudaMalloc((void **) &d_data1,
    (REVIEW_DIM + 1) * batch_size * sizeof(float)));

  // Establishing an array of the device input buffer
  float *d_array[2];
  d_array[0] = d_data0;
  d_array[1] = d_data1;

  /* Allocating device output buffer */
  // Creating output arrays
  float *d_weights;
  gpuErrChk(cudaMalloc((void **) &d_weights, REVIEW_DIM * sizeof(float)));

  gpuErrChk(cudaMemcpy(d_weights, weights, REVIEW_DIM * sizeof(float),
    cudaMemcpyHostToDevice));

  /* Creating data streams */
  cudaStream_t stream[2];
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);

  // initialize the batchID variable to identify which buffer to use
  int batchID;

  // Creating events for analyzing the program
  cudaEvent_t batchcopy;
  cudaEventCreate(&batchcopy);
  cudaEvent_t throughputstart;
  cudaEventCreate(&throughputstart);
  cudaEvent_t throughputstop;
  cudaEventCreate(&throughputstop);

  // initialize timers
  float throughput = -1;

  // Initialize errors
  float errors = -1.0;

  // main loop to process input lines (each line corresponds to a review)
  int review_idx = 0;
  for (string review_str; getline(in_stream, review_str); review_idx++) {
    // DONE: process review_str with readLSAReview
    // batchID determines which buffer to use
    batchID = ((review_idx + 1) / batch_size) % 2;

    // Check if batch is not filled
    readLSAReview(review_str, h_array[batchID] +
      (review_idx % batch_size), batch_size);

    if ((review_idx + 1) % batch_size == 0) {
      // Synchronize the memcopy
      // Make sure previous memcopy happened before doing another
      cudaEventSynchronize(batchcopy);

      // Start recording for throughput
      cudaEventRecord(throughputstart, stream[batchID]);

      // DONE: if batch is full, call kernel

      // Copy from H -> D
      cudaMemcpyAsync(d_array[batchID], h_array[batchID],
        (REVIEW_DIM + 1) * batch_size * sizeof(float), cudaMemcpyHostToDevice,
        stream[batchID]);

      // Memcopy event
      cudaEventRecord(batchcopy, stream[batchID]);

      // Kernel call
      errors = cudaClassify(d_array[batchID], batch_size, 1.0, d_weights,
        num, grad, stream[batchID]);



      // Copy from D -> H
      cudaMemcpyAsync(weights, d_weights,
        REVIEW_DIM * sizeof(float), cudaMemcpyDeviceToHost, stream[batchID]);

      // Stop recording for throughput
      cudaEventRecord(throughputstop, stream[batchID]);
      cudaEventSynchronize(throughputstart);
      cudaEventSynchronize(throughputstop);
      cudaEventElapsedTime(&throughput, throughputstart, throughputstop);
      cudaEventDestroy(throughputstart);
      cudaEventDestroy(throughputstop);
      }
  }

  // DONE: print out weights
  printf("Weights:\n");
  for (int i = 0; i < REVIEW_DIM; i++){
    printf("%.4e\n", weights[i]);
  }
  // DONE: free all memory
  gpuErrChk(cudaFreeHost(h_data0));
  gpuErrChk(cudaFreeHost(h_data1));
  gpuErrChk(cudaFree(d_data0));
  gpuErrChk(cudaFree(d_data1));
  gpuErrChk(cudaFree(d_weights));
  gpuErrChk(cudaStreamDestroy(stream[0]));
  gpuErrChk(cudaStreamDestroy(stream[1]));
  // Destroying Events
  gpuErrChk(cudaEventDestroy(batchcopy));

  // Print timer results
  printf("Latency of processing 1 batch: %f ms\n", throughput);
  // Print error rate
  printf("Errors per batch: %f\n", errors);
}

int main(int argc, char** argv) {
  int batch_size = 2048;

  if (argc == 1) {
    classify(cin, batch_size);
  } else if (argc == 2) {
    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    classify(buffer, batch_size);
  }
}
