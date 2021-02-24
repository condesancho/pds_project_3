#include "CU_util.h"


__global__ void filter_image(float *filtered, float *padded, int im_rows, int im_cols, float *gaussian_kernel, int patch_size, float sigma){

    int row = blockIdx.x;
    int col = threadIdx.x;

    filtered[row*im_cols + col] = cuNonLocalMeans(padded, gaussian_kernel, patch_size, sigma, im_rows , im_cols);   
    
}

float* denoise_image(float* h_X, int rows, int cols, int patch_size, float patch_sigma, float filter_sigma){
    
    float** gaussian = gaussian_Kernel(patch_size, patch_sigma);
    float* h_gaussian = matToRowMajor(gaussian, patch_size, patch_size);

    // Make padded
    float* h_padded = pad_array(h_X, rows, cols, patch_size);
    int padded_rows = rows + patch_size -1;
    int padded_cols = cols + patch_size -1;

    float* h_filtered = (float*)malloc(rows*cols*sizeof(float));

    float* d_padded;
    float* d_X_filtered;
    float* d_gaussian;

    cudaMalloc(&d_padded, padded_rows*padded_cols*sizeof(float));
    cudaMalloc(&d_X_filtered, rows*cols*sizeof(float));
    cudaMalloc(&d_gaussian, patch_size*patch_size*sizeof(float));

    cudaMemcpy(d_padded, h_padded, padded_rows*padded_cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gaussian, h_gaussian, patch_size*patch_size*sizeof(float), cudaMemcpyHostToDevice);

    filter_image<<<rows, cols>>>(d_X_filtered, d_padded, rows, cols, d_gaussian, patch_size, filter_sigma);

    cudaMemcpy(h_filtered, d_X_filtered, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_padded);
    cudaFree(d_gaussian);
    cudaFree(d_X_filtered);

    free(h_padded);
    free(h_gaussian);

    return h_filtered;
}



int main(void){

    int rows = 64, cols = 64;
    int patch_size = 5;
    float patch_sigma = 5/2;
    float filter_sigma = 0.02;

    float* image = read_csv2("/home/bill/Downloads/THMMY/0.02/distorted_image.csv", rows, cols);

    float* new_image = denoise_image(image, rows, cols, patch_size, patch_sigma, filter_sigma);

    // print_array(h_X, rows, cols);
    create_csv("denoised.csv", new_image, rows, cols);

    free(new_image);
    free(image);

    return 0;
}