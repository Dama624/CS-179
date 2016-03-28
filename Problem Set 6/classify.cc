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
  // Initialize error
  float err = -1.0;

  // grad is the value of (-1 / batch_size) * the sum grad for all
  // points in the batch
  float *grad;
  gpuErrChk(cudaMalloc(&grad, (REVIEW_DIM + 1) * batch_size * sizeof(float)));
  

  // num is the numerator of the grad point calculation
  float *num;
  gpuErrChk(cudaMalloc(&num, REVIEW_DIM * batch_size * sizeof(float)));

  // DONE: randomly initialize weights, allocate and initialize buffers on
  //       host & device

  // Initialize and allocate weights
  float *d_weights;
  float *weights = new float[REVIEW_DIM];
  gaussianFill(weights, REVIEW_DIM);
  gpuErrChk(cudaMalloc(&d_weights, REVIEW_DIM * sizeof(float)));
  gpuErrChk(cudaMemcpy(d_weights, weights, REVIEW_DIM * sizeof(float),
    cudaMemcpyHostToDevice));

  // Allocate host data
  float *h_data = (float *) malloc((REVIEW_DIM + 1) * batch_size *
                                    sizeof(float));

  // Allocate device data
  float *d_data;
  gpuErrChk(cudaMalloc(&d_data,
    (REVIEW_DIM + 1) * batch_size * sizeof(float)));

  // Creating events for analyzing the program
  cudaEvent_t throughputstart;
  cudaEventCreate(&throughputstart);
  cudaEvent_t throughputstop;
  cudaEventCreate(&throughputstop);

  // initialize timers
  float throughput = -1;
  float IO = -1;

  // main loop to process input lines (each line corresponds to a review)
  int review_idx = 0;
  START_TIMER();
  for (string review_str; getline(in_stream, review_str); review_idx++) {

    // DONE: process review_str with readLSAReview
    readLSAReview(review_str, h_data +
      (review_idx % batch_size), batch_size);

    // DONE: if batch is full, call kernel
    if ((review_idx + 1) % batch_size == 0){
      // Start Timer
      cudaEventRecord(throughputstart);

      // Copy from H -> D
      gpuErrChk(cudaMemcpy(d_data, h_data,
        (REVIEW_DIM + 1) * batch_size * sizeof(float),
        cudaMemcpyHostToDevice));

      // Kernel call
      err = cudaClassify(d_data, batch_size, 0.1, d_weights,
        num, grad);
      // Print error
      printf("Error rate of batch %d: %f\n", review_idx / batch_size, err);

      // Copy from D -> H
      cudaMemcpy(weights, d_weights, REVIEW_DIM * sizeof(float),
        cudaMemcpyDeviceToHost);

      // End Timer
      cudaEventRecord(throughputstop);
      cudaEventSynchronize(throughputstart);
      cudaEventSynchronize(throughputstop);
      cudaEventElapsedTime(&throughput, throughputstart, throughputstop);
      cudaEventDestroy(throughputstart);
      cudaEventDestroy(throughputstop);

    }
  }
  STOP_RECORD_TIMER(IO);

  // DONE: print out weights
  printf("Weights:\n");
  for (int i = 0; i < REVIEW_DIM; i++){
    printf("%.4e\n", weights[i]);
  }

  // Print timer results
  printf("Latency of processing 1 batch: %f ms\n", throughput);
  printf("Total time: %f ms\n", IO);

  // DONE: free all memory
  gpuErrChk(cudaFree(grad));
  gpuErrChk(cudaFree(num));
  delete[] weights;
  gpuErrChk(cudaFree(d_weights));
  free(h_data);
  gpuErrChk(cudaFree(d_data));

}

int main(int argc, char** argv) {
  int batch_size = 65536;

  if (argc == 1) {
    classify(cin, batch_size);
  } else if (argc == 2) {
    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    classify(buffer, batch_size);
  }
}
