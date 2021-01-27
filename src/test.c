#include "utilities.h"
#include <time.h>

int main(void){

    int size = 5;
    float *X = (float*)malloc(size*size*sizeof(float));

    srand(time(NULL));

    for(int i=0; i<size*size; i++){
        X[i] = i+1;
    }

    print_array(X, size, size);

    int patch_size = 5;
    int row = 2;
    int col = 2;

    float *pad = pad_array(X,size,size,patch_size);

    // printf("\n");
    // for (int i=0; i<size+patch_size-1; i++){
    //     for (int j=0; j<size+patch_size-1; j++){
    //         printf("|%f |", pad[i*(size+patch_size-1) + j]);
    //     }
    //     printf("\n");
    // }

    // float *patch = patch_finder(pad, patch_size, row, col, size, size);
    
    // print_array(patch, patch_size, patch_size);

    int numberOfPixels = size*size;
    int numberOfPixelsInPatch = patch_size*patch_size;

    float** Gaussian_Patches = (float**)malloc(numberOfPixels*sizeof(float*));
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            Gaussian_Patches[i*size + j] = patch_finder(pad,patch_size,i,j,size,size);
            Gaussian_Patches[i*size + j] = gaussian_Filtering(Gaussian_Patches[i*size + j],patch_size,2);
        }  
    }
    

    // float result = nonLocalMeans(X, Gaussian_Patches, patch_size, size, size, 0.1, 2, 2);
    // printf("%f\n", result);

    float* f_new = (float*)malloc(size*size*sizeof(float));
    for (int i = 0; i < size ; i++){
        for (int j = 0; j < size; j++){
            f_new[i*size+j] = nonLocalMeans(X, Gaussian_Patches, patch_size, size, size, 1, i, j);
        }
    }
    
    printf("\n");
    print_array(f_new, size, size);

    
    for (int i = 0; i < numberOfPixels; i++)
    {
        free(Gaussian_Patches[i]);
    }
    free(Gaussian_Patches);
    free(f_new);
    free(pad);
    free(X);
    
    return 0;
}