#include <cstdio>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>

#define TIMESTEPS 1000
#define THREADS 1000
#define MAXTIME 100
#define nBLOCKS 200
#define nTHREADS 512

// Initializing the rate constants
#define b 10
#define g 1
#define Kon 0.1
#define Koff 0.9

void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    }
    else
    {
        // fprintf(stderr, "No kernel error detected\n");
    }
}

/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/*
 * Performs a single iteration of the Gilelspie algorithm
 *
 * random: An array of randomly generated floats from 0 to 1. The array is
 *         size THREADS
 * conc: An array that represents the current concentration of X. The
 *       size is THREADS * TIMESTEPS
 * currenttime: The current time. The array size is THREADS
 * on_off: An array of ON/OFF states for each thread ID. The array is
 *         size THREADS
 * decider: An array of randomly generated floats from 0 to 1 that decides
 *          what reaction proceeds
 * flag: Determines whether the kernel should be run again later.
 */

__global__
void Kernel_Gillespie(float *random, unsigned int *conc, float *currenttime,
    unsigned int *on_off, float *decider, unsigned int *flag){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < THREADS){
        if (currenttime[i] < MAXTIME){
            if (on_off[i] == 1){ // If the current state is ON
                // Calculating total rate
                int timeindex = (int)(currenttime[i] /
                    ((float)MAXTIME / THREADS));
                float lambda = Koff + b + (conc[i +
                    (timeindex * THREADS)] * g);
                // Calculating time step
                float timestep = -log(random[i]) / lambda;
                // Deciding what reaction to do
                float ontooff = Koff / lambda;
                float incconc = b / lambda;
                currenttime[i] += timestep;
                if (decider[i] < ontooff){ // ON -> OFF
                    on_off[i] = 0;
                }
                else if (decider[i] >= ontooff &&
                            decider[i] < (ontooff + incconc)){
                    // [X]++
                    int newtimeindex = (int)(currenttime[i] /
                    ((float)MAXTIME / THREADS));
                    for (int k = newtimeindex; k < TIMESTEPS; k++){
                        conc[i + (k * THREADS)] += 1;
                    }
                }
                else{ // [X]--
                    int newtimeindex = (int)(currenttime[i] /
                    ((float)MAXTIME / THREADS));
                    for (int k = newtimeindex; k < TIMESTEPS; k++){
                        if (conc[i + (k * THREADS)] > 0){
                            conc[i + (k * THREADS)] -= 1;
                        }
                    }
                }
                if (currenttime[i] < MAXTIME){
                    *flag = 1;
                }
            }
            else{ // If the current state is OFF
                // Calculate the total rate
                int timeindex = (int)(currenttime[i] /
                    ((float)MAXTIME / THREADS));
                float lambda = Kon + (conc[i +
                    (timeindex * THREADS)] * g);
                // Calculate the time step
                float timestep = -log(random[i]) / lambda;
                // Deciding what reaction to do
                float offtoon = Kon / lambda;
                currenttime[i] += timestep;
                if (decider[i] < offtoon){ // OFF -> ON
                    on_off[i] = 1;
                }
                else{ // [X]--
                   int newtimeindex = (int)(currenttime[i] /
                    ((float)MAXTIME / THREADS));
                    for (int k = newtimeindex; k < TIMESTEPS; k++){
                        if (conc[i + (k * THREADS)] > 0){
                            conc[i + (k * THREADS)] -= 1;
                        }
                    }
                }
                if (currenttime[i] < MAXTIME){
                    *flag = 1;
                }
            }
        }
    i += blockDim.x * gridDim.x;
    }
}

/* 
 * Calculates the expected concentration at each timepoint.
 *
 * conc: The 2D array of each thread's concentration at each timepoint
 * expect: Array of expected concentration at each timepoint. Size is 
 *         MAXTIME
 */

__global__
void Kernel_ExpectedConc(unsigned int *conc, float *expect){
    extern __shared__ float shd[];

    // Loading data onto shared memory
    unsigned int thread = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // The reduction (sequential addressing)
    while (i < THREADS){
        shd[thread] = (float)conc[i];
        __syncthreads();
        for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1){
            if (thread < j){
                shd[thread] += shd[thread + j];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0){ // Only one thread executes atomicAdd
            atomicAdd(expect, shd[0]);
            *expect /= THREADS;
        }
        i += blockDim.x * gridDim.x;
    }
}



/* 
 * Calculates the variance at each timepoint.
 *
 * conc: The 2D array of each thread's concentration at each timepoint
 * variance: Array of variance at each timepoint. Size is MAXTIME.
 * expect: Array of expected concentration at each timepoint. Size is 
 *         MAXTIME
 */

__global__
void Kernel_Variance(unsigned int *conc, float *variance,
    float *expect){
    extern __shared__ float shd[];

    // Loading data onto shared memory
    unsigned int thread = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // The reduction (sequential addressing)
    while (i < THREADS){
        shd[thread] = pow((float)conc[i] - *expect, 2);
        __syncthreads();
        for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1){
            if (thread < j){
                shd[thread] += shd[thread + j];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0){ // Only one thread executes atomicAdd
            atomicAdd(variance, shd[0]);
        }
        i += blockDim.x * gridDim.x;
    }
}

int main(void){
    // Generate the array (of length THREADS) of randomly-generated floats
    // This decides the time step
    float *rand_pts;
    gpuErrchk(cudaMalloc((void **) &rand_pts, THREADS * sizeof(float)));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    

    // Generate the randomly-generated array for deciding what reaction
    // happens
    float *decider;
    gpuErrchk(cudaMalloc((void **) &decider, THREADS * sizeof(float)));

    // Initialize array of ON/OFF states to 0
    // 0: OFF state
    // 1: ON state
    unsigned int *on_off;
    gpuErrchk(cudaMalloc((void **) &on_off, THREADS * sizeof(unsigned int)));
    cudaMemset(on_off, 0, THREADS * sizeof(unsigned int));

    // Initialize the current time & concentration
    float *currenttime;
    gpuErrchk(cudaMalloc((void **) &currenttime, THREADS * sizeof(float)));
    cudaMemset(currenttime, 0, THREADS * sizeof(float));

    unsigned int *conc;
    gpuErrchk(cudaMalloc((void **) &conc, THREADS *
        TIMESTEPS * sizeof(unsigned int)));
    cudaMemset(conc, 0, THREADS * TIMESTEPS * sizeof(unsigned int));

    // Initialize the flag that determines whether the kernel is run again
    unsigned int *flag;
    unsigned int h_flag = 1;
    gpuErrchk(cudaMalloc((void **) &flag, sizeof(unsigned int)));
    cudaMemset(flag, 1, sizeof(unsigned int));
    
    // Call Gillespie kernel
    while (h_flag == 1){
        cudaMemset(flag, 0, sizeof(unsigned int));
        curandGenerateUniform(gen, rand_pts, THREADS);
        curandGenerateUniform(gen, decider, THREADS); 
        Kernel_Gillespie<<<nBLOCKS, nTHREADS>>>
            (rand_pts, conc, currenttime, on_off, decider, flag);
        cudaMemcpy(&h_flag, flag, sizeof(unsigned int),
            cudaMemcpyDeviceToHost);
    }

    // Initialize the expected concentration array
    float *expect;
    float h_expect[TIMESTEPS];
    gpuErrchk(cudaMalloc((void **) &expect, TIMESTEPS * sizeof(float)));

    // Call Expected Value kernel
    for (int k = 0; k < TIMESTEPS; k++){
        Kernel_ExpectedConc<<<nBLOCKS, nTHREADS, THREADS * sizeof(float)>>>
            (&conc[k * THREADS], &expect[k]);
    }
    checkCUDAKernelError();

    cudaMemcpy(h_expect, expect, TIMESTEPS * sizeof(float),
        cudaMemcpyDeviceToHost);

    // Initialize the expected variance array
    float *variance;
    float h_variance[TIMESTEPS];
    gpuErrchk(cudaMalloc((void **) &variance, TIMESTEPS * sizeof(float)));

    // Call Variance kernel
    for (int k = 0; k < TIMESTEPS; k++){
        Kernel_Variance<<<nBLOCKS, nTHREADS, THREADS * sizeof(float)>>>
            (&conc[k * THREADS], &variance[k], &expect[k]);
    }
    checkCUDAKernelError();

    cudaMemcpy(h_variance, variance, TIMESTEPS * sizeof(float),
        cudaMemcpyDeviceToHost);

    for (int i = 0; i < TIMESTEPS; i++){
        h_variance[i] /= THREADS;
        printf("Timestep %.2f: \n", (i * ((float)MAXTIME / TIMESTEPS)));
        printf("Expectation: %f \n", h_expect[i]);
        printf("Variance: %f\n\n", h_variance[i]);
    }

    // Free all memory
    cudaFree(rand_pts);
    cudaFree(decider);
    cudaFree(on_off);
    cudaFree(currenttime);
    cudaFree(conc);
    cudaFree(flag);
    cudaFree(expect);
    cudaFree(variance);
    
    return 0;
}