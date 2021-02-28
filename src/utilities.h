#ifndef UTILITIES_H
#define UTILITIES_H


#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

#ifndef RAND_MAX
#define RAND_MAX ((int) ((unsigned) ~0 >> 1))
#endif


/** 
 * This file contains many useful functions that we used to implement the programs.
 * 
 * Contains:
 *  i) General purpose functions:
 *      1) void print_array(float *array, int rows, int cols)
 *      2) __device__ void d_print_array(float *array, int rows, int cols)
 *      3) float *matToRowMajor(float** matrix, int n, int m)
 *      4) float *read_csv2(char* filename, int rows, int cols)
 *      5) void create_csv(char *filename, float *array, int rows, int cols)
 * 
 *  ii) Functions used for the non local means algorithm:
 *      6) float* pad_array(float *F, int rows, int cols, int patch_size)
 *      7) float* patch_finder(float *padded_array, int patch_size, int row_central, int col_central, int rows, int cols)
 *      8) float** gaussian_Kernel(int size, float sigma)
 *      9) float* gaussian_Filtering(float* P, float** kernel,int size, float patch_sigma)
 *  
 */


/*****************************************************************************/
/*                        General purpose functions                          */
/*****************************************************************************/

/**
 * Simple function that takes an array in row major format and prints it in the terminal.
 * Used to corroborate the results.
 */
void print_array(float *array, int rows, int cols){
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            printf("%f ", array[i*cols+j]);
        }
        printf("\n");
    }
}

/**
 * Function that takes a two-dimensional matrix and returns it in row major format.
 */
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
 
/**
 * Takes a file and stores it in an array that is returned.
 * The inputs are the path of the file and the rows and cols to be read.
 */
float *read_csv2(char* filename, int rows, int cols){

    float* X = (float*)malloc(rows*cols*sizeof(float));
    if (X == NULL){
        printf("Error: Couldn't allocate memory.\n");
        exit(-1);
    }
    
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

/**
 * Takes a one dimensional array in row major format and stores it in a csv file
 */
void create_csv(char *filename, float *array, int rows, int cols){

    strcat(filename, ".csv");
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

/*****************************************************************************/
/*              Functions used for the non local means algorithm             */
/*****************************************************************************/


/**
 * This function pads the image in order to fit the patches
 * 
 * Inputs:  float*F         --> The array to be padded. Corresponds to the starting image
 *          int rows        --> The rows of the array
 *          int cols        --> The cols of the array
 *          int patch_size  --> The size of the patch used
 * 
 * According to the patch size, this function extends the array up, left, right and down
 * by patch_size/2 using the mirroring method, since it is used in the MatLab code.
 * Two extensions are made. One above the array(image) and one below.
 * Based on those extensions and the starting image the left and right extensions are made too.
 * There 8 different cases to make the padded array depending on the position of the element.
 * 
 */
float* pad_array(float *F, int rows, int cols, int patch_size){

    int rows_new = rows + patch_size - 1;
    int cols_new = cols + patch_size - 1;

    // The extended array to be returned
    float *pad_array = (float*)malloc(rows_new*cols_new*sizeof(float));

    // The extension cols and rows are the integer part of the patch_size/2
    int extension = patch_size/2;

    // Array that stores the mirrored part above the image
    float *upper_extension = (float*)malloc(extension*cols*sizeof(float));

    int index = 0;

    // Start from the bottom row of the extension and ascend
    // The original array is accessed in the correct order using the index variable
    for (int i=extension-1; i>-1; i--){
        for (int j=0; j<cols; j++){
            upper_extension[i*cols+j] = F[index];
            index++;

        }
    }

    // Array that stores the mirrored part below the image
    float *lower_extension = (float*)malloc(extension*cols*sizeof(float));
    
    index = rows*cols-1;

    // Start from the first row of the extension but from the last element in that row
    // The original image is accessed backwards starting from the last element using the index variable
    for (int i=0; i<extension; i++){
        for (int j= cols - 1; j>-1; j--){
            lower_extension[i*cols+j] = F[index];
            index--;
        }
    }

    // Run every element of the padded array and find its value
    for (int i = 0; i < rows_new; i++){
        for (int j = 0; j < cols_new; j++){
            
            //For the left side
            if (j < extension){
                // Upper left takes its mirrored values from the upper extension
                if (i < extension){
                    pad_array[i*cols_new + j] = upper_extension[i*cols + extension - 1 - j];
                }
                // Middle left takes its value from the original array
                else if (i >= extension && i < rows + extension){
                    pad_array[i*cols_new + j] = F[(i-extension)*cols + extension - 1 - j];
                }
                // Bottom left takes its mirrored values from the bottom extension
                else{
                    pad_array[i*cols_new +j] = lower_extension[(i-extension-rows)*cols + extension - 1 - j];
                }
                
            }

            // For the middle cols
            else if (j >= extension  && j < extension + cols){ 
                int new_j = j -extension;
                // Upper takes its mirrored values from the upper extension
                if (i < extension){ 
                    pad_array[i*cols_new + j] = upper_extension[i*cols + new_j];
                }
                // Middle takes its value from the original array
                else if (i >= extension && i < rows + extension){
                    pad_array[i*cols_new + j] = F[(i-extension)*cols + new_j];
                }
                // Bottom takes its mirrored values from the bottom extension
                else{
                    pad_array[i*cols_new + j] = lower_extension[(i-extension-rows)*cols + new_j];
                }
            }

            // For the right side
            else{
                int new_j = j -extension - cols + 1;
                // Upper right takes its mirrored values from the upper extension
                if (i < extension){ 
                    pad_array[i*cols_new + j] = upper_extension[i*cols + cols - new_j];
                }
                // Middle right takes its value from the original array
                else if (i >= extension && i < rows + extension){
                    pad_array[i*cols_new + j] = F[(i-extension)*cols + cols - new_j];
                }
                // Bottom right takes its mirrored values from the bottom extension
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


/**
 * Returns the values of a patch given the central pixel positions in the image
 */
float* patch_finder(float *padded_array, int patch_size, int row_central, int col_central, int rows, int cols){
    // Check if the pixel belongs in the image
    if (row_central<0 || row_central>rows-1 || col_central<0 || col_central>cols-1){
        printf("Invalid centre for patch.\n");
        exit(-1);
    }

    // Allocate space for the patch
    float* Patch = (float *)malloc(patch_size*patch_size* sizeof(float));
    
    // Useful for changing row in the padded array
    int padded_cols = cols + patch_size - 1;

    // Stores the index of the first element of the patch in the padded aray
    int first_elem = row_central*padded_cols + col_central;

    // Fill the patch
    for (int i = 0; i < patch_size; i++){
        for (int j = 0; j < patch_size; j++){
            Patch[i*patch_size + j] = padded_array[first_elem + i*padded_cols + j];
        }
    }

    return Patch;
}

/**
 * Function that returns the gaussian kernel weights depending on the patch size and the patch sigma.
 * The weights are returned in a 2d matrix.
 */
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
           kernel2d[i][j] = alpha * expf(-((j - mu)*(j - mu) + (i - mu)*(i - mu)) / (2 * sigma * sigma));
           sum += kernel2d[i][j];
        }
    }

    // NORMALIZE by dividing with the sum and the max according to the MatLab code
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

/**
 * Takes a patch and multiplies it with the gaussian kernel.
 * It returns the new patch.
 */
float* gaussian_Filtering(float* P, float** kernel,int size, float patch_sigma){
    
    float **Patch = (float **)malloc(size*sizeof(float*));
    for (int i = 0; i < size; i++){
        Patch[i] = (float*)malloc(size*sizeof(float));
    }
    
    //Convert patch take to 2d array from row major
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            Patch[i][j] = P[size*i + j];
        }
    }
    free(P);

    // Multiply patch and kernel
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            Patch[i][j] = Patch[i][j]*kernel[i][j];
        }
    }

    // Convert patch to one-dimensional array
    float* new_patch = matToRowMajor(Patch, size, size);
    
    return new_patch;    
}

#endif