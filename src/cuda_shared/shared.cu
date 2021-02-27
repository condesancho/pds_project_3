#include "cu_util_shared.h"


__global__ void filter_image(float *filtered, float *padded, int im_rows, int im_cols, float *gaussian_kernel, int patch_size, float sigma){

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int elem_indx = bx*blockDim.x + tx;

    int pixel_row = elem_indx/im_cols;
    int pixel_col = elem_indx%im_cols;

    filtered[pixel_row*im_cols + pixel_col] = cuNonLocalMeans(padded, gaussian_kernel, patch_size, sigma, im_rows , im_cols);   
    
}

int main(void){

    int rows = 64, cols = 64;
    int patch_size = 5;
    float patch_sigma = 5/2;
    float filt_sigma = 0.1;

    float* h_X = read_csv2("/home/bill/Downloads/THMMY/0.1/distorted_image.csv", rows, cols);

    float** gaussian = gaussian_Kernel(patch_size, patch_sigma);
    float* h_gaussian = matToRowMajor(gaussian, patch_size, patch_size);

    // Make padded
    float* h_padded = pad_array(h_X, rows, cols, patch_size);
    int padded_rows = rows + patch_size -1;
    int padded_cols = cols + patch_size -1;

    float* d_padded;
    float* d_X_filtered;
    float* d_gaussian;

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

    // print_array(h_X, rows, cols);
    create_csv("denoised.csv", h_X, rows, cols);

    free(h_X);
    free(h_padded);
    free(h_gaussian);

    return 0;
}