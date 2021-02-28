#include "../utilities.h"

/**
 * This file contains two functions that implement the Non Local Means algorithm to filter an image using CUDA
 *  and taking advantage of the shared memory of the threads.
 */

/**
 * Function that filters the value of a single pixel as described in the exercise and the MatLab code
 * 
 * Inputs:
 *      float* filtered             --> The filtered image stored in the global memory
 *      float* padded             --> The padded image array in the global memory
 *      float *gaussian_kernel      --> The array that stores the gaussian weights in the global memory
 *      int im_rows                 --> The image rows
 *      int im_cols                 --> The image cols
 *      int patch_size              --> The dimension of the square patch
 *      float filter_sigma          --> The filter sigma used in non local means
 *      
 * Works like the function in the util_global.h file.
 * The difference is that the threads use the shared memory to find the patch values.
 * The way it is divided is that the threads inside a block belong in the same row of the picture.
 * That means that the threads called depend on the pixel columns in the image, which lead to problems if the column size is big (>1024).
 * The shared memory takes a strip of the padded array so that it contains all the pixel patches of the row.
 * Furthermore, in the shared memory these stored the gaussian kernel to reduce calls from the global memory.
 * 
 */
__global__ void filter_image(float *filtered, float *padded, float *gaussian_kernel, int im_rows, int im_cols, int patch_size, float filter_sigma){
    
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // The position of the pixel to be filtered is based on the position of the thread in the grid
    int elem_indx = bx*blockDim.x + tx;

    // Find the position of the pixel in the original image
    int pixel_row = elem_indx/im_cols;
    int pixel_col = elem_indx%im_cols;

    int patch_elements = patch_size*patch_size;

    int padded_cols = im_rows + patch_size - 1;

    // Keeps the index of the first element of the image in the padded array
    int first_im_elem = (patch_size/2)*padded_cols + patch_size/2;

    // The first element of the patch in the global padded array
    int first_element = pixel_row*padded_cols + pixel_col;

    // The value to be returned
    float new_pixel_val = 0;

    // Keeps the first element of the temporary pilex's patch
    int first_element_temp = 0;

    float temp_w = 0;
    float Z = 0;
    float norm;
    float val = 0;
    float patch_val;
    float temp_patch_val;
    float gaussian_filt;

    int corresponding_thread = 0;

    // The shared memory is dynamically defined
    extern __shared__ float patches_kernel[];
    float* kernel = patches_kernel;
    float* patches = &kernel[patch_elements];
    
    // The thread with id 0 brings all the gaussian kernel elements in the shared memory
    if (tx == 0){
        for (int i=0; i<patch_elements; i++){
            kernel[i] = gaussian_kernel[i];
        }
    }

    // All threads bring their patch in the shared memory
    for (int i=0; i<patch_size; i++){
        for (int j=0; j<patch_size; j++){
            patches[tx + i*padded_cols + j] = padded[first_element + i*padded_cols +j];
        }
    }    

    __syncthreads();


    // Find the weights for every element in the image
    for (int i=0; i<im_rows; i++){
        for (int j=0; j<im_cols; j++){
            norm = 0;

            // Finds the first element of the temp patch in the padded array
            first_element_temp = i*padded_cols + j;

            // If the temp pixel is the same with the current pixel do not take it into account
            if (first_element_temp == first_element){
                continue;
            }

            // Runs for every element of patch
            for (int k=0; k<patch_size; k++){
                for (int l=0; l<patch_size; l++){
                    // Find the values from the shared array
                    patch_val = patches[tx + k*padded_cols + l];

                    // If temp pixel is in the same block
                    if (first_element_temp >= bx*blockDim.x  &&  first_element_temp < (bx+1)*blockDim.x){
                        // Find its thread
                        corresponding_thread = first_element_temp % padded_cols;
                        //Take the value from the shared memory
                        temp_patch_val = patches[corresponding_thread + k*padded_cols + l];
                    }
                    else{
                        temp_patch_val = padded[first_element_temp + k*padded_cols + l];
                    }

                    // Get the gaussian weight from the shared memory
                    gaussian_filt = kernel[k*patch_size + l];

                    norm += gaussian_filt*gaussian_filt*(patch_val - temp_patch_val)*(patch_val - temp_patch_val);
                }
            }
            
            // Find the weight with the temporary patch
            temp_w = expf(-norm/filter_sigma);
                
            Z = Z + temp_w;

            // Get the value of the central pixel of the temp patch
            val = padded[first_im_elem + first_element_temp];

            new_pixel_val = new_pixel_val + temp_w*val;

        }
    }

    new_pixel_val = new_pixel_val/Z;

    filtered[pixel_row*im_cols + pixel_col] = new_pixel_val;
}

/**
 * Function that takes an image and filters the noise.
 * Same with the global implementation
 * 
 * Inputs:
 *      float* h_X          --> The image to be filtered
 *      int rows            --> The pixel rows of the image
 *      int cols            --> The pixel cols of the image
 *      int patch_size      --> The dimension of the square patch dezired
 *      int patch_sigma     --> The sigma used to find the gaussian kernel weights
 *      int filter_sigma    --> The sigma used in the non local means algorithm
 * 
 * Output:
 *  The new filtered image
 * 
 * Works like the function in the util_global.h file.
 * The threads initialized MUST be equal to the number of columns on the image.
 */
float* denoise_image(float* h_X, int rows, int cols, int patch_size, float patch_sigma, float filter_sigma){
    
    float** gaussian = gaussian_Kernel(patch_size, patch_sigma);
    float* h_gaussian = matToRowMajor(gaussian, patch_size, patch_size);

    // Make padded
    float* h_padded = pad_array(h_X, rows, cols, patch_size);
    int padded_rows = rows + patch_size -1;
    int padded_cols = cols + patch_size -1;

    float* h_filtered = (float*)malloc(rows*cols*sizeof(float));

    float* d_padded;
    float* d_filtered;
    float* d_gaussian;

    // Allocate GPU memory
    cudaMalloc(&d_padded, padded_rows*padded_cols*sizeof(float));
    cudaMalloc(&d_filtered, rows*cols*sizeof(float));
    cudaMalloc(&d_gaussian, patch_size*patch_size*sizeof(float));

    // Pass the padded array and the gaussian weights in the GPU
    cudaMemcpy(d_padded, h_padded, padded_rows*padded_cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gaussian, h_gaussian, patch_size*patch_size*sizeof(float), cudaMemcpyHostToDevice);

    int noBlocks = rows;
    int noThreads = cols;
    int shared_memory = patch_size*patch_size*sizeof(float) + (noThreads+patch_size-1)*patch_size*sizeof(float);

    filter_image<<<noBlocks, noThreads, shared_memory>>>(d_filtered, d_padded, d_gaussian, rows, cols, patch_size, filter_sigma);

    // Return the filtered image in the CPU
    cudaMemcpy(h_filtered, d_filtered, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_padded);
    cudaFree(d_gaussian);
    cudaFree(d_filtered);

    cudaDeviceSynchronize();

    free(h_padded);
    free(h_gaussian);

    return h_filtered;
}