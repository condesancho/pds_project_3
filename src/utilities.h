#include <stdlib.h>
#include <math.h>
#include <stdio.h>
//#include "G:\Program Files\MATLAB\R2018a\extern\include\mat.h"

//all matrices are in row major format

// float *F = matOpen("../code/data/house.mat",)

/*patch finder function*/

float* patch_finder(float *F, int patch_size, int x_central, int y_central, int x_border, int y_border){

    //allocate space for the patch
    float* Patch = (float *)malloc(patch_size*patch_size* sizeof(float));
    //find the patch

    /*todo implemnt different algorithms for searching patch depending on the central pixel position*/
    
    /*if the patch is not getting out of bounds*/
    int x0_patch = x_central + 1 - patch_size / 2;
    int y0_patch = y_central + 1 - patch_size / 2;
    if (abs(x_central - x_border) > patch_size / 2 && abs(y_central - y_border) > patch_size / 2 && (x_central-(patch_size / 2)) > 0 && (y_central-(patch_size / 2)) > 0)
    {
        for (int i = 0; i < patch_size; i++)
        {
            for (int j = 0; j < patch_size; j++)
            {
                //to complex to analyze this algorithm but it works trust us we are engineers 
                Patch[i*patch_size + j] = F[x0_patch*i + y0_patch + j + patch_size*i + 1 + patch_size];
            }
        }
    }else if (/* condition */)
    {
        //todo mirror the patch parts that are out of bounds.  
    }
    
    
    
    

    return Patch;
}






double* nonLocalMeans(float **F, int patch_size, float filter_sigma, float patch_sigma){
    float** Patch = (float *)malloc(patch_size * sizeof(float*));
    for (int i = 0; i <//toimport size of the image; i++)
    {
        Patch[i] = malloc(patch_size * sizeof(float));
    }
    
    Patch = patch_finder(F,patch_size);


    //apply gaussian filter to patch 
    
    Gaussian_Patch = gaussian_Filtering(Patch, patch_size, patch_sigma);


    free(Patch);
    
}


float** gaussian_Filtering(float** P, int size, float sigma){
    float* gauss = malloc(size*size*sizeof(float));
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            float rows[size] = P[]
            float columns[size] = 
        }
        
    }
    
}
// for (int i = 0; i < COLS; i++)
// {
//     for (int j = 0; j < ROWS; j++)
//     {
//         printf("%f", a[i*COLS + j])
//     }
    
    
// }
