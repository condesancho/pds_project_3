#include "utilities.h"

int main(){

    float X[25] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};

    int size = 5;

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
    // float *patch = patch_finder(pad,patch_size,row,col,size,size);
    // for (int i=0; i<patch_size; i++){
    //     for (int j=0; j<patch_size; j++){
    //         printf("|%f |", patch[i*patch_size + j]);
    //     }
    //     printf("\n");
    // }
    int numberOfPixels = size*size;
    int numberOfPixelsInPatch = patch_size*patch_size;
    float** Gaussian_Patches = (float**)malloc(size*size*sizeof(float*));
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            Gaussian_Patches[i*size + j] = patch_finder(pad,patch_size,i,j,size,size);
            Gaussian_Patches[i*size + j] = gaussian_Filtering(Gaussian_Patches[i*size + j],patch_size,2);       
        }
        
    }
    for (int i = 0; i < numberOfPixels; i++)
    {
        for (int j = 0; j < numberOfPixelsInPatch; j++)
        {
            printf(" %.2f ", Gaussian_Patches[i][j]);
        }
        printf("\n");
    }
    
    
    float* NLM_filteredX = malloc(size*size*sizeof(float));
    float** Temp = (float**)malloc(size*sizeof(float*));
    for (int i = 0; i < size; i++)
    {
        Temp[i] = malloc(size*sizeof(float));
    }
    
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            Temp[i][j] = nonLocalMeans(X,Gaussian_Patches,patch_size,size,size,0.1,i,j);
        }
        
    }
    NLM_filteredX = matToRowMajor(Temp,size,size);
  
    
    
    printf("--------------NON LOCAL MEANS FILTERED MATRIX-------\n");
    for(int i = 0; i < size; i++)
    {   
        for (int j = 0; j < size; j++)
        {
            printf(" %f ", NLM_filteredX[i*size + j]); 
        }
        
        printf("\n");
               
    }
    
    
    
    for (int i = 0; i < numberOfPixels; i++)
    {
        free(Gaussian_Patches[i]);
    }
    free(Gaussian_Patches);
    for (int i = 0; i < size; i++)
    {
        free(Temp[i]);
    }
    free(Temp);
    free(pad);
    free(NLM_filteredX);
    return 0;
}