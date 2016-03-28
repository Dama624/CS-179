/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Cuda1DFDWave_cuda.cuh"


/* DONE: You'll need a kernel here, as well as any helper functions
to call it */

/* 
 * Solves the wave equation for time t+1 given data at time t & t-1.
 * old_data: Array of data at time t-1
 * curr_data: Array of data at time t
 * new_data: Array of data for time t+1. We are writing our output here.
 * dx: The change in x
 * dt: The change in t
 * courantsq: (c*dt / dx) ^ 2 factor
 * numberOfNodes: Self explanatory.
 */

__global__
void Wave_Solve(float *old_data, float *curr_data, float *new_data,
    float dx, float dt, float courantsq, size_t numberOfNodes) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
    while (index < (numberOfNodes - 1)){
        new_data[index] = (2 * curr_data[index]) - old_data[index]
            + (courantsq * (curr_data[index + 1] - 2 * curr_data[index]
                + curr_data[index - 1]));
        index += blockDim.x * gridDim.x;
    }
}

void cudaWave_Solve(float *old_data, float *curr_data, float *new_data,
    float dx, float dt, float courantsq, size_t numberOfNodes,
    unsigned int blocks, unsigned int threadsPerBlock){

    Wave_Solve<<<blocks, threadsPerBlock>>>
        (old_data, curr_data, new_data, dx, dt, courantsq,
            numberOfNodes);
}