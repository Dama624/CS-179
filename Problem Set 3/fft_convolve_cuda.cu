/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve_cuda.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {


    /* DONE: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < padded_length){
        out_data[i].x = ((raw_data[i].x * impulse_v[i].x)
            - (raw_data[i].y * impulse_v[i].y)) / padded_length;
        out_data[i].y = ((raw_data[i].x * impulse_v[i].y)
            + (raw_data[i].y * impulse_v[i].x)) / padded_length;
        i += blockDim.x * gridDim.x;
    }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* DONE 2: 

    ** EXPLANATION **

    Reduction method: sequential addressing. Split the array of values
    by half, then use that number of threads to read from array index
    [i] and index [i + (half the array size)]. Doing so prevents the bank
    conflicts that a "binary tree" reduction causes, and also is not
    inefficient in the number of warps used. 

    */
    extern __shared__ float shd[];

    // Loading output data onto shared memory
    unsigned int thread = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    

    // The reduction (sequential addressing)
    while (i < padded_length){
        shd[thread] = fabs(out_data[i].x);
        __syncthreads();
        for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1){
            if (thread < j){
                if (shd[thread] < shd[thread + j]){
                    shd[thread] = shd[thread + j];
                }
            }
            __syncthreads();
        }
        // Atomic max to find the max among all blocks
        if (threadIdx.x == 0){ // Only one thread executes atomicMax
            atomicMax(max_abs_val, shd[0]);
        }
        i += blockDim.x * gridDim.x;
    }

}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* DONE 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < padded_length){
        out_data[i].x /= *max_abs_val;
        i += blockDim.x * gridDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* DONE: Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>
        (raw_data, impulse_v, out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        

    /* DONE 2: Call the max-finding kernel. */
    cudaMaximumKernel<<<blocks, threadsPerBlock,
        threadsPerBlock * sizeof(float)>>>
        (out_data, max_abs_val, padded_length);
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* DONE 2: Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>>
        (out_data, max_abs_val, padded_length);
}
