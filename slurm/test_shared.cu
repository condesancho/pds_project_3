#include "cu_util_shared.h"

__global__ void filter_image(float *filtered, float *padded, int im_rows, int im_cols, float *gaussian_kernel, int patch_size, float sigma){

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int elem_indx = bx*blockDim.x + tx;

    int pixel_row = elem_indx/im_cols;
    int pixel_col = elem_indx%im_cols;

    filtered[pixel_row*im_cols + pixel_col] = cuNonLocalMeans(padded, gaussian_kernel, patch_size, sigma, im_rows , im_cols);   
    
}

int main(int argc, char* argv[]){

    float patch_sigma = 5/2;
    float filt_sigma = 0.01;

    int rows = 0;
    int cols = 0;
    int patch_size = 0;
    
    if(argc < 2){
        printf("\nUsage: %s [No. dimensions] [Patch size]\n", argv[0]);
        exit(-1);
    }

    rows = atoi(argv[1]);
    cols = atoi(argv[1]);
    patch_size = atoi(argv[2]);

    float* h_X = (float*)malloc(rows*cols*sizeof(float));

    for (int i=0; i<rows*cols; i++){
        h_X[i] = (float)rand()/RAND_MAX;
    }

    printf("\nCurrently in %s running random image with %d dimension and patch size %d.\n", argv[0], rows, patch_size);
    

    float** gaussian = gaussian_Kernel(patch_size, patch_sigma);
    float* h_gaussian = matToRowMajor(gaussian, patch_size, patch_size);


    // Make padded
    float* h_padded = pad_array(h_X, rows, cols, patch_size);
    int padded_rows = rows + patch_size -1;
    int padded_cols = cols + patch_size -1;

    float* d_padded;
    float* d_X_filtered;
    float* d_gaussian;

    struct timespec begin, end;

    // Starting the clock
    clock_gettime(CLOCK_MONOTONIC, &begin);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_padded, padded_rows*padded_cols*sizeof(float));
    cudaMalloc(&d_X_filtered, rows*cols*sizeof(float));
    cudaMalloc(&d_gaussian, patch_size*patch_size*sizeof(float));


    cudaMemcpy(d_padded, h_padded, padded_rows*padded_cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gaussian, h_gaussian, patch_size*patch_size*sizeof(float), cudaMemcpyHostToDevice);

    int noBlocks = rows;
    int noThreads = cols;
    int shared_memory = patch_size*patch_size*sizeof(float) + (noThreads+patch_size-1)*patch_size*sizeof(float);

    filter_image<<<noBlocks, noThreads, shared_memory>>>(d_X_filtered, d_padded, rows, cols, d_gaussian, patch_size, filt_sigma);

    cudaMemcpy(h_X, d_X_filtered, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_padded);
    cudaFree(d_gaussian);
    cudaFree(d_X_filtered);

    cudaDeviceSynchronize();

    // Stopping the clock
    clock_gettime(CLOCK_MONOTONIC, &end);
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;

    printf("\nTime elapsed: %.5f seconds.\n\n", elapsed);

    printf("Done with %s that run a random image with dimension %d and patch size %d.\n\n", argv[0], rows, patch_size);

    free(h_X);
    free(h_padded);
    free(h_gaussian);

    return 0;
}