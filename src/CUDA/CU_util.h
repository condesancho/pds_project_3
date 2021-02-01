#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#ifndef RAND_MAX
#define RAND_MAX ((int) ((unsigned) ~0 >> 1))
#endif

#ifndef PATCH_ELEM
#define PATCH_ELEM 9
#endif

// General functions used by the CPU

__device__ void d_print_array(float *array, int rows, int cols){
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            printf("%.3f ", array[i*cols+j]);
        }
        printf("\n");
    }
}

void print_array(float *array, int rows, int cols){
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            printf("%.3f ", array[i*cols+j]);
        }
        printf("\n");
    }
}


//ARRAY PADDING 
//extends F by patch_size / 2 from each side and uses the mirroring technique
float* pad_array(float *F, int rows, int cols, int patch_size){

    int rows_new = rows + patch_size - 1;
    int cols_new = cols + patch_size - 1;

    // The extended array to be returned
    float *pad_array = (float*)malloc(rows_new*cols_new*sizeof(float));

    int extension = patch_size/2;

    float *upper_extension = (float*)malloc(extension*cols*sizeof(float));

    int index = 0;
    for (int i=extension-1; i>-1; i--){
        for (int j=0; j<cols; j++){
            upper_extension[i*cols+j] = F[index];
            index++;

        }
    }

    float *lower_extension = (float*)malloc(extension*cols*sizeof(float));
    // Start from the last the element 
    index = rows*cols-1;
    for (int i=0; i<extension; i++){
        for (int j= cols - 1; j>-1; j--){
            lower_extension[i*cols+j] = F[index];
            index--;
        }
    }
    /*
        Upper left and Right and Lower left and Right extensions of F are symmmetrical to upper_extension and lower_extension respectively.    
    */
    for (int i = 0; i < rows_new; i++){
        for (int j = 0; j < cols_new; j++){
            //For the left side
            if (j < extension){
                // Upper left
                if (i < extension){
                    pad_array[i*cols_new + j] = upper_extension[i*cols + extension - 1 - j];
                }
                // Middle left
                else if (i >= extension && i < rows + extension){
                    pad_array[i*cols_new + j] = F[(i-extension)*cols + extension - 1 - j];
                }
                // Bottom left
                else{
                    pad_array[i*cols_new +j] = lower_extension[(i-extension-rows)*cols + extension - 1 - j];
                }
                
            }
            // For the middle cols
            else if (j >= extension  && j < extension + cols){ 
                int new_j = j -extension;
                // Upper
                if (i < extension){ 
                    pad_array[i*cols_new + j] = upper_extension[i*cols + new_j];
                }
                // Middle
                else if (i >= extension && i < rows + extension){
                    pad_array[i*cols_new + j] = F[(i-extension)*cols + new_j];
                }
                // Bottom
                else{
                    pad_array[i*cols_new + j] = lower_extension[(i-extension-rows)*cols + new_j];
                }
            }
            // For the right side
            else{
                int new_j = j -extension - cols + 1;
                // Upper right
                if (i < extension){ 
                    pad_array[i*cols_new + j] = upper_extension[i*cols + cols - new_j];
                }
                // Middle right
                else if (i >= extension && i < rows + extension){
                    pad_array[i*cols_new + j] = F[(i-extension)*cols + cols - new_j];
                }
                // Bottom right
                else{
                    pad_array[i*cols_new + j] = lower_extension[(i-extension-rows)*cols + cols - new_j];
                }
            }
        }
    }
    

    free(upper_extension);
    free(lower_extension);

    return pad_array;
}

float *matToRowMajor(float** matrix, int n, int m){
    float *RowMajor = (float*)malloc(n*m*sizeof(float));
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            RowMajor[i*m+j] = matrix[i][j];
        }        
    }

    for (int i=0; i<n; i++){
        free(matrix[i]);
    }
    free(matrix);

    return RowMajor; 
}


float** gaussian_Kernel(int size, float sigma){
    
    // Make the two-dimensional kernel
    float **kernel2d = (float**)malloc(size*sizeof(float*));
    for (int i = 0; i < size; i++){
        kernel2d[i] = (float*)malloc(size*sizeof(float));
    }
    
    float sum = 0;
    float mu = size/2;
    float alpha = 1 / (2 * M_PI * sigma * sigma);

    // Get the_kernel
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
           kernel2d[i][j] = alpha * expf(-((j - mu)*(j - mu) + (i - mu)*(i - mu)) / (2*sigma*sigma));
           sum += kernel2d[i][j];
        }
    }

    
    // NORMALIZE by dividing with the sum and the max
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            kernel2d[i][j] /= sum;
        }
    }

    // Find the max
    float max = kernel2d[0][0];
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            if(max < kernel2d[i][j])
                max = kernel2d[i][j];
        }
    }

    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            kernel2d[i][j] /= max;
        }
    }

    return kernel2d;

}


__device__ void filter_patch(float *patch, float *gaussian_kernel, int patch_size){
    for(int i=0; i<patch_size*patch_size; i++){
        patch[i] = patch[i]*gaussian_kernel[i];
    }
}


__device__ float cuNonLocalMeans(float *padded_F, float *gaussian_kernel, int patch_size, float filter_sigma, int im_rows, int im_cols){
    
    int central_row = blockIdx.x*blockDim.x + threadIdx.x;
    int central_col = blockIdx.y * blockDim.y + threadIdx.y;
    
    int padded_cols = im_rows + patch_size - 1;

    // Keeps the index of the first element of the image in the padded array
    int first_im_elem = (patch_size/2)*padded_cols + patch_size/2;

    int first_element = central_row*padded_cols + central_col;

    const int patch_elements = PATCH_ELEM;

    float current_patch[patch_elements];
    float temp_patch[patch_elements];

    // The value to be returned
    float new_pixel_val = 0;

    if (central_row < im_rows && central_col < im_cols){
        // printf("I'm working on the pixel (%d, %d)\n", central_row, central_col);

        // Find the patch and filter it
        for (int i=0; i<patch_size; i++){
            for (int j=0; j<patch_size; j++){
                current_patch[i*patch_size+j] = padded_F[first_element + i*padded_cols + j];
            }
        }
        filter_patch(current_patch, gaussian_kernel, patch_size);


        // Useful variables
        float temp_w = 0;
        float Z = 0;
        float norm;
        float max_w = FLT_MIN;
        float val = 0;

        // Find the weights for every element in the image
        for (int i=0; i<im_rows; i++){
            for (int j=0; j<im_cols; j++){
                norm = 0;

                // Finds the first element of the temp patch in the padded array
                first_element = i*padded_cols + j;

                /// Get the value of the central pixel of the temp patch
                val = padded_F[first_im_elem + first_element];

                // Get the temp patch and filter it
                for (int p=0; p<patch_size; p++){
                    for (int q=0; q<patch_size; q++){
                        temp_patch[p*patch_size+q] = padded_F[first_element + p*padded_cols + q];
                    }
                }
                filter_patch(temp_patch, gaussian_kernel, patch_size);
                
                // Calculate the norm between the two patches
                for (int k=0; k<patch_elements; k++){
                    norm += (current_patch[k]-temp_patch[k]) * (current_patch[k]-temp_patch[k]);
                }
                
                temp_w = expf(-norm/filter_sigma);
                


                // The patches must be different
                if (first_element != central_row*padded_cols + central_col){
                    Z += temp_w;
                    if (max_w < temp_w){
                        max_w = temp_w;
                    }
                    new_pixel_val += temp_w*val;
                    
                }
            }
        }
        val = padded_F[first_im_elem + central_row*padded_cols + central_col];
        new_pixel_val += max_w*val;
        Z += max_w;
        new_pixel_val /= Z;
    }

    return new_pixel_val;
}