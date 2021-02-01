#include "CU_util.h"


__global__ void filter_image(float *filtered, float *padded, int im_rows, int im_cols, float *gaussian_kernel, int patch_size, float sigma){

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < im_rows && col < im_cols){
        filtered[row*im_cols + col] = cuNonLocalMeans(padded, gaussian_kernel, patch_size, sigma, im_rows , im_cols);
    }
    
}

int main(void){

    int rows = 5, cols = 5;
    int patch_size = 3;
    float patch_sigma = 2;
    float filt_sigma = 1;

    float* h_X = (float*)malloc(rows*cols*sizeof(float));

    for (int i=0; i<rows*cols; i++){
        h_X[i] = i+1;
    }
    print_array(h_X, rows, cols);

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

    dim3 grid(1,1);
    dim3 block(rows, cols);
    filter_image<<<grid, block>>>(d_X_filtered, d_padded, rows, cols, d_gaussian, patch_size, filt_sigma);

    cudaMemcpy(h_X, d_X_filtered, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_padded);
    cudaFree(d_gaussian);
    cudaFree(d_X_filtered);

    // print_array(h_X, rows, cols);

    free(h_X);
    free(h_padded);
    free(h_gaussian);

    return 0;
}