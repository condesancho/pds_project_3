#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#ifndef RAND_MAX
#define RAND_MAX ((int) ((unsigned) ~0 >> 1))
#endif


__global__ void cuNonLocalMeans(float dev_new_pixel,float *F, float **Gaussian_Patches, int patch_size, int im_rows, int im_cols, float filter_sigma, int pixel_row, int pixel_col){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < im_rows && col < im_cols)
    {
        float *Current_Patch = Gaussian_Patches[row*im_cols + col];
        float *Temp = Gaussian_Patches[row];
        Norm2 = 0;
        for ( i = 0; i < count; i++)
        {
            /* code */
        }
        

    }
    
}