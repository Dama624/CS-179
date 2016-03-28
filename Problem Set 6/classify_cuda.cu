#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "classify_cuda.cuh"

/*
 * Arguments:
 * data: Memory that contains both the review LSA coefficients and the labels.
 *       Format decided by implementation of classify.
 * batch_size: Size of mini-batch, how many elements to process at once
 * step_size: Step size for gradient descent. Tune this as needed. 1.0 is sane
 *            default.
 * weights: Pointer to weights vector of length REVIEW_DIM.
 * errors: Pointer to a single float used to describe the error for the batch.
 *         An output variable for the kernel. The kernel can either write the
 *         value of loss function over the batch or the misclassification rate
 *         in the batch to errors.
 */
__global__
void trainLogRegKernel(float *data, int batch_size, int step_size,
		       float *weights, float *errors,
               float *num, float *grad) {
    /* 
     * Reminder to self: accessing data element X for thread Y is
     * = data[Y + (X * batch_size)]
     */

    // DONE: write me
    // For each batch of reviews
    unsigned int review = blockIdx.x * blockDim.x + threadIdx.x;

    // For each review
    while (review < batch_size){
        /* Applying the gradient descent */
        // These float initializations are for calculating the denominator
        float exponent = 0;
        float denom = 0;
        // Initialize int to keep track of errors
        int errorcount = 0;
        // Calculating the numerator and denominator
        for (int i = 0; i < REVIEW_DIM; i++){
            // Calculating the numerator
            num[review + (i * batch_size)] = data[review + (i * batch_size)] *
                data[review + (REVIEW_DIM * batch_size)];
            // Calculating the denominator
            exponent += weights[i] * data[review + (i * batch_size)];
        }
        denom = (1 + exp(data[review + (REVIEW_DIM * batch_size)] *
                exponent)) * -batch_size;
        // Calculating the gradient
        for (int i = 0; i < REVIEW_DIM; i++){
            atomicAdd(&grad[review + (i * batch_size)],
              num[review + (i * batch_size)] / denom); 
        }


        /* Calculating prediction */
        if ((exponent > 0 && data[review + (REVIEW_DIM * batch_size)] < 0)
            || (exponent < 0 && data[review + (REVIEW_DIM * batch_size)] > 0)){
            errorcount++; 
        }

        // Sync threads before updating weight
        __syncthreads();

        /* Updating the weight */
        for (int i = 0; i < REVIEW_DIM; i++){
            atomicAdd(&weights[i], -step_size * 
                grad[review + (i * batch_size)]);
        }
        // Calculate final error
        atomicAdd(errors, (float) errorcount / batch_size);
        review += blockDim.x * gridDim.x;
    }
}

/*
 * All parameters have the same meaning as in docstring for trainLogRegKernel.
 * Notably, cudaClassify returns a float that quantifies the error in the
 * minibatch. This error should go down as more training occurs.
 */
float cudaClassify(float *data, int batch_size,
                   float step_size, float *weights,
                   float *num, float *grad) {
  int block_size = (batch_size < 512) ? batch_size : 512;

  // grid_size = CEIL(batch_size / block_size)
  int grid_size = (batch_size + block_size - 1) / block_size;
  int shmem_bytes = 0;

  float *d_errors;
  cudaMalloc(&d_errors, sizeof(float));
  cudaMemset(d_errors, 0, sizeof(float));
  cudaMemset(grad, 0, (REVIEW_DIM + 1) * batch_size * sizeof(float));
  trainLogRegKernel<<<grid_size, block_size, shmem_bytes>>>(data,
                                                                    batch_size,
                                                                    step_size,
                                                                    weights,
                                                                    d_errors,
                                                                    num,
                                                                    grad);



  float h_errors = -1.0;
  cudaMemcpy(&h_errors, d_errors, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_errors);
  return h_errors;
}
