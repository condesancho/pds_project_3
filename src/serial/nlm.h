#include "../utilities.h"

/**
 * This file contains two functions that implement the Non Local Means algorithm to filter an image.
 */

/**
 * Function that filters the value of a single pixel as described in the exercise and the MatLab code
 * 
 * Inputs:  
 *      float* F                    --> The original image array
 *      float **Gaussian_Patches    --> A 2d matrix that each row stores a patch that is gaussian filtered
 *      int patch_size              --> The dimension of the square patch
 *      int im_rows                 --> The image rows
 *      int im_cols                 --> The image cols
 *      float filter_sigma          --> The filter sigma used in non local means
 *      int pixel_row               --> The row of the pixel to be filtered in the image
 *      int pixel_col               --> The col of the pixel to be filtered in the image
 *      
 * Output:
 *  The new value of the filtered image
 * 
 */
float nonLocalMeans(float* F, float** Gaussian_Patches, int patch_size, int im_rows, int im_cols, float filter_sigma, int pixel_row, int pixel_col){
   
    // The value to be returned
    float new_pixel_val = 0;
    
    // The patch of the pixel to be examined
    float* Current_Patch = Gaussian_Patches[pixel_row*im_cols + pixel_col];
    
    float* Temp_Patch;
    float Norm2 = 0;

    // The weight and the sum of them
    float W = 0;
    float Z = 0;

    //Gaussian Patches is a row major Patches storage matrix for each pixel
    for (int i = 0; i < im_rows*im_cols; i++){

        // If the Temp_Patch is the same with the Current_Patch do not take it into account
        if (i == pixel_row*im_cols + pixel_col){
            continue;
        }

        // Change the temporary patch and reset the norm
        Temp_Patch = Gaussian_Patches[i];
        Norm2 = 0;
        
        // Calculate the norm of the two patches
        for (int j = 0; j < patch_size*patch_size; j++){
            Norm2 += (Current_Patch[j]-Temp_Patch[j]) * (Current_Patch[j]-Temp_Patch[j]);
        }

        // Calculate the weight
        W = expf(-Norm2/filter_sigma);


        // Find the sum of the weights
        Z += W;
        new_pixel_val += W*F[i];
    }

    new_pixel_val /= Z;

    return new_pixel_val;
}

/**
 * Function that takes an image and filters the noise.
 * 
 * Inputs:
 *      float* F            --> The image to be filtered
 *      int rows            --> The pixel rows of the image
 *      int cols            --> The pixel cols of the image
 *      int patch_size      --> The dimension of the square patch dezired
 *      int patch_sigma     --> The sigma used to find the gaussian kernel weights
 *      int filter_sigma    --> The sigma used in the non local means algorithm
 * 
 * Output:
 *  The new filtered image
 * 
 * This function takes an image, pads it and using the nonLocalMeans function above filters the pixels one by one.
 * It uses a 2d matrix that stores all the patches of the image multiplied by the gaussian kernel for reduced complexity.
 */
float* denoise_image(float* F, int rows, int cols, int patch_size, float patch_sigma, float filter_sigma){
    // Make the padded array
    float *pad = pad_array(F, rows, cols, patch_size);

    
    int number_of_pixels = rows*cols;
    
    // Store the gaussian kernel
    float **kernel = gaussian_Kernel(patch_size, patch_sigma);

    // Make the array that stores all the patches of the pixels
    float** Gaussian_Patches = (float**)malloc(number_of_pixels*sizeof(float*));
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            Gaussian_Patches[i*cols+ j] = patch_finder(pad, patch_size, i, j, rows, cols);
            Gaussian_Patches[i*cols + j] = gaussian_Filtering(Gaussian_Patches[i*cols + j], kernel, patch_size, patch_sigma);
        }  
    }

    // For every pixel of the image find the new value
    float* f_new = (float*)malloc(rows*cols*sizeof(float));
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            f_new[i*cols+j] = nonLocalMeans(F, Gaussian_Patches, patch_size, rows, cols, filter_sigma, i, j);
        }
    }

    // Free all the unnecessary arrays
    for (int i = 0; i < number_of_pixels; i++){
        free(Gaussian_Patches[i]);
    }
    free(Gaussian_Patches);
    free(pad);
    for (int i = 0; i < patch_size; i++){
        free(kernel[i]);
    }
    
    free(kernel);


    return f_new;
}