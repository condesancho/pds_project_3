#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include 




/*patch finder function*/

float** patch_finder(float **F, int patch_size, int x_central, int y_central){

    //allocate space for the patch
    float** Patch = (float *)malloc(patch_size * sizeof(float*));
    for (int i = 0; i < count; i++)
    {
        Patch[i] = malloc(patch_size * sizeof(float));
    }
    //find the patch
    for (int i = 0; i < patch_size; i++)
    {
        for (int j = 0; j < patch_size; j++)
        {
            Patch[i][j] = F[x_central - patch_size / 2 + i][y_central - patch_size / 2 + j];
        }
        
    }
    
    

    return Patch;
}






double* nonLocalMeans(float **F, int patch_size, float filter_sigma, float patch_sigma){
    float** Patch = (float *)malloc(patch_size * sizeof(float*));
    for (int i = 0; i < count; i++)
    {
        Patch[i] = malloc(patch_size * sizeof(float));
    }
    
    Patch = patch_finder(F,patch_size);


    //apply gaussian filter to patch 
    
    Gaussian_Patch = gaussian_Filtering(Patch, patch_size, patch_sigma);

    
    for (int i = 0; i < patch_size; i++)
    {
        free(Patch[i]);
    }
    
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
