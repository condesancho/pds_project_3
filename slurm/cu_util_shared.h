#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#ifndef RAND_MAX
#define RAND_MAX ((int) ((unsigned) ~0 >> 1))
#endif

// General functions used by the CPU and GPU

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

float *read_csv2(char* filename, int rows, int cols){

    float *X = (float*)malloc(rows*cols*sizeof(float));
    
    FILE *matFile = fopen(filename, "r");

    if (matFile == NULL){
        printf("Couldn't open file\n");
        exit(1);
    }

    float num;

    int i=0,j=0;

    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){
            if(fscanf(matFile,"%f,",&num)==EOF) break;
            X[i*cols+j]=num;
        }
    }

    fclose(matFile);

    return X;
}

void create_csv(char *filename, float *array, int rows, int cols){

    FILE *f = fopen(filename,"w");
    
    if (f == NULL){
        printf("Couldn't open file.\n");
        exit(1);
    }
 
    for(int i=0; i<rows; i++){
 
        for(int j=0; j<cols-1; j++){
            fprintf(f, "%f,", array[i*cols + j]);
        }
        fprintf(f, "%f\n", array[(i+1)*cols-1]);
 
    }
    fclose(f);
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


__device__ float cuNonLocalMeans(float *padded_F, float *gaussian_kernel, int patch_size, float filter_sigma, int im_rows, int im_cols){
    
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int elem_indx = bx*blockDim.x + tx;

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

    int first_element_temp = 0;

    // Useful variables
    float temp_w = 0;
    float Z = 0;
    float norm;
    float val = 0;
    float patch_val;
    float temp_patch_val;
    float gaussian_filt;

    int corresponding_thread = 0;

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
            patches[tx + i*padded_cols + j] = padded_F[first_element + i*padded_cols +j];
        }
    }    

    __syncthreads();

    // if (elem_indx == 13){
    //     // d_print_array(kernel, patch_size, patch_size);
    //     printf("%d ", elem_indx%tx);
    // }

    first_element_temp = 0;


    // if (first_element_temp > (bx*blockDim.x-1) && first_element_temp < (bx+1)*blockDim.x && tx ==0){
    //     corresponding_thread = first_element_temp;
    //     printf("%d ", corresponding_thread);
    // }

    // Find the weights for every element in the image
    for (int i=0; i<im_rows; i++){
        for (int j=0; j<im_cols; j++){
            norm = 0;

            // Finds the first element of the temp patch in the padded array
            first_element_temp = i*padded_cols + j;

            if (first_element_temp == first_element){
                continue;
            }

            for (int k=0; k<patch_size; k++){
                for (int l=0; l<patch_size; l++){
                    patch_val = patches[tx + k*padded_cols + l];

                    // If temp pixel is in the same block
                    if (first_element_temp >= bx*blockDim.x  &&  first_element_temp < (bx+1)*blockDim.x){
                        corresponding_thread = first_element_temp % padded_cols;
                        temp_patch_val = patches[corresponding_thread + k*padded_cols + l];
                    }
                    else{
                        temp_patch_val = padded_F[first_element_temp + k*padded_cols + l];
                    }

                    gaussian_filt = kernel[k*patch_size + l];

                    norm += gaussian_filt*gaussian_filt*(patch_val - temp_patch_val)*(patch_val - temp_patch_val);
                }
            }
                
            temp_w = expf(-norm/filter_sigma);
                
            Z = Z + temp_w;

            // Get the value of the central pixel of the temp patch
            val = padded_F[first_im_elem + first_element_temp];

            new_pixel_val = new_pixel_val + temp_w*val;

        }
    }

    new_pixel_val = new_pixel_val/Z;

    return new_pixel_val;
}