#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#ifndef RAND_MAX
#define RAND_MAX ((int) ((unsigned) ~0 >> 1))
#endif

//#include "G:\Program Files\MATLAB\R2018a\extern\include\mat.h"

// All matrices are in row major format

// float *F = matOpen("../code/data/house.mat",)

void print_array(float *array, int rows, int cols){
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            printf("%f ,", array[i*cols+j]);
        }
        printf("\n");
    }
}

/*Returns the values of a patch given the central pixel positions in the image*/
float* patch_finder(float *padded_array, int patch_size, int row_central, int col_central, int rows, int cols){

    if (row_central<0 || row_central>rows-1 || col_central<0 || col_central>cols-1){
        printf("Invalid centre for patch.\n");
        exit(-1);
    }

    // Allocate space for the patch
    float* Patch = (float *)malloc(patch_size*patch_size* sizeof(float));
    
    // Useful for changing line in the padded array
    int padded_cols = cols + patch_size - 1;

    // Stores the index of the first element of the patch in the padded aray
    int first_elem = row_central*padded_cols + col_central;

    for (int i = 0; i < patch_size; i++){
        for (int j = 0; j < patch_size; j++){
            Patch[i*patch_size + j] = padded_array[first_elem + i*padded_cols + j];
        }
    }

    return Patch;
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
float randomBounded(float low, float high) {
    double r;

    r = (float) rand() / ((float) RAND_MAX + 1);
    return (low + r * (high - low));
}


float *matToRowMajor(float** matrix, int n, int m){
        float *RowMajor = (float*)malloc(n*m*sizeof(float));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                RowMajor[i*m+j] = matrix[i][j];
            }
            
        }
        
        return RowMajor; 
}
 

float *read_csv2(char* filename, int rows, int cols){
    float *X = malloc(rows*cols*sizeof(float));
    FILE *matFile = fopen(filename, "r");

    if (matFile == NULL){
        printf("Couldn't open file\n");
        exit(1);
    }

    float num,temp;

    int i=0,j=0;

    for(i=0; i<rows; i++){
        fscanf(matFile,"%f",&num);
        for(j=0; j<cols; j++){
            if(fscanf(matFile,",%f",&num)==EOF) break;
            X[i*cols+j]=num;
        }
        fscanf(matFile,"%[^\n]\n");
    }

    
    fclose(matFile);
    return X;
}
void convolution_2D(float **P, float **kernel , float **result, int size) {

// find center position of kernel (half of kernel size)
int kCenterX = size / 2;
int kCenterY = size / 2;

    for (int i = 0; i < size; ++i)              // rows
    {
        for (int j = 0; j < size; ++j)          // columns
        {
            for (int m = 0; m < size; ++m)     // kernel rows
            {
                int mm = size - 1 - m;      // row index

                for (int n = 0; n < size; ++n) // kernel columns
                {
                    int nn = size - 1 - n;  // column index

                    // index of input signal, used for checking boundary
                    int ii = i + (m - kCenterY);
                    int jj = j + (n - kCenterX);

                    // ignore input samples which are out of bound
                    if (ii >= 0 && ii < size  && jj >= 0 && jj < size)
                        result[i][j] += P[ii][jj] * kernel[mm][nn];
                }
            }
        }
    }
}
float** gaussian_Kernel(int size, float sigma){
    float **kernel2d = (float**)malloc(size*sizeof(float*));
    for (int i = 0; i < size; i++)
    {
        kernel2d[i] = malloc(size*sizeof(float));
    }
    
    float sum = 0;
    float mu = size/2;
    //get_kerneL
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            
           kernel2d[i][j] = (1 / (2 * M_PI * sigma * sigma)) * expf(-((j - mu) * (j - mu) + (i - mu) * (i - mu)) / (2 * sigma * sigma));
           sum += kernel2d[i][j];
        }
        
    }
    

    
    //NORMALIZE /sum and /max
  

    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            kernel2d[y][x] /= sum;
            
        }
        
    }
    float max = kernel2d[0][0];
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if(max < kernel2d[i][j])
                max = kernel2d[i][j];
        }
        
    }
    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            kernel2d[y][x] /= max;
        }
        
    }
    return kernel2d;
}
float* gaussian_Filtering(float* P, float** kernel,int size, float patch_sigma){
    float **Patch = (float **)malloc(size*sizeof(float*));
    for (int i = 0; i < size; i++)
    {
        Patch[i] = (float*)malloc(size*sizeof(float));
    }
    
    //Convert patch take to 2d array from row major
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            Patch[i][j] = P[size*i + j];
        }
        
    }
    free(P);

    
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            Patch[i][j] = Patch[i][j]*kernel[i][j];
        }
        
    }
    float* new_patch = matToRowMajor(Patch,size,size);
    for (int i = 0; i < size; i++)
    {
        free(Patch[i]);
    }
    free(Patch);
    
    return new_patch;    
}


float nonLocalMeans(float* F, float **Gaussian_Patches, int patch_size, int im_rows, int im_cols, float filter_sigma, int pixel_row, int pixel_col){
    // The value to be returned
    float new_pixel_val = 0;
    
    // The patch to be examined
    float* Current_Patch = Gaussian_Patches[pixel_row*im_cols + pixel_col];
    float* Temp_Patch ;
    float Norm2 = 0;

    // Variable to replace weight with itself according to Matlab
    float max = FLT_MIN;

    // The weights and the sum of them
    float* W = (float*)malloc(im_rows*im_cols*sizeof(float));
    float Z = 0;

    //Gaussian Patches is a row major Patches storage matrix for each pixel
    for (int i = 0; i < im_rows*im_cols; i++){
        // Change the temporary patch and reset norm
        Temp_Patch = Gaussian_Patches[i];
        Norm2 = 0;
        
        // Calculate the norm of the two patches
        for (int j = 0; j < patch_size*patch_size; j++){
            Norm2 += (Current_Patch[j]-Temp_Patch[j]) * (Current_Patch[j]-Temp_Patch[j]);
        }

        // Calculate the weight
        W[i] = expf(-Norm2/filter_sigma);

        // Find the max weight excluding the weight of the pixel with itself
        if (i != pixel_row*im_cols + pixel_col){
            Z += W[i];
            if (max < W[i]){
                max = W[i];
            }
        }
    }

    // Replace the weight of the pixel with itself (according to the Matlab code)
    W[pixel_row*im_cols + pixel_col] = max;

    // Adjust the sum of the weights
    Z += max;

    // Find the new value of the pixel
    for (int i = 0; i < im_rows*im_cols; i++){
        new_pixel_val += W[i]*F[i]/Z;
    }
   
    free(W);

    return new_pixel_val;
}


