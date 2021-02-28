#include "../utilities.h"

/**
 * This file contains two functions that implement the Non Local Means algorithm to filter an image using CUDA.
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
 * This function uses CUDA for programming in parallel and filters an image throught the non local means algorithm.
 * Stored in the global memory is the padded array of the image, the gaussian kernel and the filtered image.
 * Depending on the position of the thread in the grid, each thread calculates the new value of the pixel it deals with.
 * When the new pixel value is acquired it is stored in the filtered array to be passed in the CPU.
 * 
 */
__global__ void filter_image(float *filtered, float *padded, float *gaussian_kernel, int im_rows, int im_cols, int patch_size, float filter_sigma){
    
    // The position of the pixel to be filtered is based on the position of the thread in the grid
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int elem_indx = bx*blockDim.x + tx;

    //The index must be in the image
    if (elem_indx < im_rows*im_cols){

        // Find the position of the pixel in the original image
        int pixel_row = elem_indx/im_cols;
        int pixel_col = elem_indx%im_cols;
    
        int padded_cols = im_rows + patch_size - 1;

        // Keeps the index of the first element of the image in the padded array
        int first_im_elem = (patch_size/2)*padded_cols + patch_size/2;

        // The first element of the pixel's patch
        int first_element = pixel_row*padded_cols + pixel_col;

        // The value to be returned
        float new_pixel_val = 0;

        float temp_w = 0;
        float Z = 0;
        float norm;
        float val = 0;
        float patch_val;
        float temp_patch_val;
        float gaussian_filt;

        // Keeps the first element of the temporary pilex's patch
        int first_element_temp = 0;

        // Find the weights for every element in the image
        for (int i=0; i<im_rows; i++){
            for (int j=0; j<im_cols; j++){
                // Reset the norm
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
                        
                        // Find the values from the padded array
                        patch_val = padded[first_element + k*padded_cols + l];
                        temp_patch_val = padded[first_element_temp + k*padded_cols + l];

                        // Get the gaussian weight
                        gaussian_filt = gaussian_kernel[k*patch_size + l];

                        norm += gaussian_filt*gaussian_filt*(patch_val - temp_patch_val)*(patch_val - temp_patch_val);
                    }
                }

                // Find the weight with the temporary patch
                temp_w = expf(-norm/filter_sigma);
                
                Z += temp_w;

                // Get the value of the central pixel of the temp patch
                val = padded[first_im_elem + first_element_temp];

                new_pixel_val += temp_w*val;

            }
        }

        new_pixel_val /= Z;

        // Put the new value in the filtered array
        filtered[pixel_row*im_cols + pixel_col] = new_pixel_val;
    }
}

