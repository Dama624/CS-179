
/* 
Based off work by Nelson, et al.
Brigham Young University (2010)

Adapted by Kevin Yuh (2015)
*/


#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>

#define PI 3.14159265358979

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

// Creating the kernel for frequency scaling
__global__
void
cudaFreqScaling(cufftComplex *input_data, int width, int angles) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < width * angles){
        if ((i % width) < (width / 2)) {
            input_data[i].x *= (float) (i % width) / (width / 2);
            input_data[i].y *= (float) (i % width) / (width / 2);
        }
        else {
            input_data[i].x *= (float) (width - (i % width)) / (width / 2);
            input_data[i].y *= (float) (width - (i % width)) / (width / 2);
        }
        i += blockDim.x * gridDim.x;
    }
}

// Creating the kernel for extracting real of sinogram to float
__global__
void
cudaRealtoFloat(cufftComplex *input_data, float *output_data,
    int width, int angles) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < width * angles){
        output_data[i] = input_data[i].x;
        i += blockDim.x * gridDim.x;
    }
}

// Creating the kernel for accelerated backprojection
__global__
void
cudaBackprojection(float *output_data, float* output_image,
    int imagewidth, int angles, int sinowidth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float m; // Slope
    float x_i;
    float y_i;
    float d;
    while (i < imagewidth * imagewidth){
        for (int theta = 0; theta < angles; theta++){ 
            m = -cosf(theta * PI / angles) / sinf(theta * PI / angles);
            int x_0 = (i % imagewidth) - (imagewidth / 2);
            int y_0 = (imagewidth / 2) - (i / imagewidth);
            float q = - 1 / m;
            x_i = (y_0 - m * x_0) / (q - m);
            y_i = q * x_i;
            if (theta == 0){
                d = x_0;
                q = -1;
                x_i = 1;
            }
            else if (abs((float) theta / angles - 0.5) < 0.0001){
                d = y_0;
            }
            else {
                d = powf(powf(x_i, 2) + powf(y_i, 2), 0.5);
            }
            if ((q > 0 && x_i < 0) || (q < 0 && x_i > 0)){
                int j = (sinowidth / 2) - d + (theta * sinowidth);
                output_image[i] += output_data[j];
            }
            else {
                int j = (sinowidth / 2) + d + (theta * sinowidth);
                output_image[i] += output_data[j];
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

/* Check errors on cuFFT functions */
void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        printf("Failed FFT call, error code %d\n", errval);
    }
}


/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}




int main(int argc, char** argv){

    if (argc != 7){
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Sinogram filename > \n \
        < Width or height of original image, whichever is larger > \n \
        < Number of angles in sinogram >\n \
        < threads per block >\n \
        < number of blocks >\n \
        < output filename >\n");
        exit(EXIT_FAILURE);
    }


    /********** Parameters **********/

    int width = atoi(argv[2]);
    int height = width;
    int sinogram_width = (int)ceilf( height * sqrt(2) );

    int nAngles = atoi(argv[3]);


    int threadsPerBlock = atoi(argv[4]);
    int nBlocks = atoi(argv[5]);


    /********** Data storage *********/


    // GPU DATA STORAGE
    cufftComplex *dev_sinogram_cmplx;
    float *dev_sinogram_float; 
    float* output_dev;  // Image storage


    cufftComplex *sinogram_host;

    size_t size_result = width*height*sizeof(float);
    float *output_host = (float *)malloc(size_result);




    /*********** Set up IO, Read in data ************/

    sinogram_host = (cufftComplex *)malloc(  sinogram_width*nAngles*sizeof(cufftComplex) );

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Sinogram file missing\n");
        exit(EXIT_FAILURE);
    }

    FILE *outputFile = fopen(argv[6], "w");
    if (outputFile == NULL){
        fprintf(stderr, "Output file cannot be written\n");
        exit(EXIT_FAILURE);
    }

    int j, i;

    for(i = 0; i < nAngles * sinogram_width; i++){
        fscanf(dataFile,"%f",&sinogram_host[i].x);
        sinogram_host[i].y = 0;
    }

    fclose(dataFile);


    /*********** Assignment starts here *********/

    /* DONE: Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */
    cudaMalloc((void **) &dev_sinogram_cmplx, sinogram_width
        * nAngles * sizeof(cufftComplex));
    cudaMalloc((void **) &dev_sinogram_float, sinogram_width
        * nAngles * sizeof(float));

    cudaMemcpy(dev_sinogram_cmplx, sinogram_host, sinogram_width
        * nAngles * sizeof(cufftComplex), cudaMemcpyHostToDevice);


    /* DONE 1: Implement the high-pass filter:
        - Use cuFFT for the forward FFT
        - Create your own kernel for the frequency scaling.
        - Use cuFFT for the inverse FFT
        - extract real components to floats
        - Free the original sinogram (dev_sinogram_cmplx)

        Note: If you want to deal with real-to-complex and complex-to-real
        transforms in cuFFT, you'll have to slightly change our code above.
    */
    // Creating the forward & inverse FFT plan
    cufftHandle plan;
    int batch = nAngles; // nAngles # of transformations
    cufftPlan1d(&plan, sinogram_width, CUFFT_C2C, batch);   

    // Executing the forward FFT
    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx,
        CUFFT_FORWARD);

    // Executing the frequency scaling kernel
    cudaFreqScaling<<<nBlocks, threadsPerBlock>>>
        (dev_sinogram_cmplx, sinogram_width, nAngles); 

    // Executing the inverse FFT
    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx,
        CUFFT_INVERSE);

    // Extracting real components to floats kernel
    cudaRealtoFloat<<<nBlocks, threadsPerBlock>>>
        (dev_sinogram_cmplx, dev_sinogram_float, sinogram_width, nAngles);

    // Freeing the original sinogram
    cudaFree(dev_sinogram_cmplx);

    // Destroying the FFT plan;
    cufftDestroy(plan);

    /* TODO 2: Implement backprojection.
        - Allocate memory for the output image.
        - Create your own kernel to accelerate backprojection.
        - Copy the reconstructed image back to output_host.
        - Free all remaining memory on the GPU.
    */
    // Allocating memory for the output image
    cudaMalloc((void **) &output_dev, size_result);

    // Zeroing out output_dev
    cudaMemset(output_dev, 0, size_result);

    // Executing the kernel to accelerate backprojection
    cudaBackprojection<<<nBlocks, threadsPerBlock>>>
        (dev_sinogram_float, output_dev, width, nAngles, sinogram_width);

    // Check for error
    checkCUDAKernelError();

    // Copying the image back to output_host
    cudaMemcpy(output_host, output_dev, size_result,
        cudaMemcpyDeviceToHost);

    // Free all remaining memory on GPU
    cudaFree(dev_sinogram_float);
    cudaFree(output_dev);
    
    /* Export image data. */

    for(j = 0; j < width; j++){
        for(i = 0; i < height; i++){
            fprintf(outputFile, "%e ",output_host[j*width + i]);
        }
        fprintf(outputFile, "\n");
    }


    /* Cleanup: Free host memory, close files. */

    free(sinogram_host);
    free(output_host);

    fclose(outputFile);

    return 0;
}



