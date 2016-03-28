/* CUDA blur
 * Kevin Yuh, 2014 */

#ifndef CUDA_BLUR_CUH
#define CUDA_BLUR_CUH

void cudaCallBlurKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const float *raw_data,
        const float *blur_v,
        float *out_data,
        const unsigned int N,
        const unsigned int blur_v_size);

#endif
