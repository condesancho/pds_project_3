#include "./Serial/utilities.h"
#include <time.h>

float* denoise_image(float *F, int rows, int cols, int patch_size, float patch_sigma,  float filter_sigma){
    // Make the padded array
    float *pad = pad_array(F, rows, cols, patch_size);

    // Make the array that stores all the patches of the pixels
    int number_of_pixels = rows*cols;
    
    float **kernel = gaussian_Kernel(patch_size, patch_sigma);

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

    for (int i = 0; i < number_of_pixels; i++){
        free(Gaussian_Patches[i]);
    }
    free(Gaussian_Patches);
    free(pad);
    for (int i = 0; i < patch_size; i++)
    {
        free(kernel[i]);
    }
    
    free(kernel);
    return f_new;
}

int main(void){

    int size = 5;
    float *X = (float*)malloc(size*size*sizeof(float));

    srand(time(NULL));

    for(int i=0; i<size*size; i++){
        X[i] = i+1;
    }

    int patch_size = 3;
    int row = 2;
    int col = 2;

    // printf("\n");
    // for (int i=0; i<size+patch_size-1; i++){
    //     for (int j=0; j<size+patch_size-1; j++){
    //         printf("|%f |", pad[i*(size+patch_size-1) + j]);
    //     }
    //     printf("\n");
    // }

    // float *patch = patch_finder(pad, patch_size, row, col, size, size);
    
    // print_array(patch, patch_size, patch_size);

    // float result = nonLocalMeans(X, Gaussian_Patches, patch_size, size, size, 0.1, 2, 2);
    // printf("%f\n", result);

    float* f_new = denoise_image(X, size, size, patch_size, 1, 2);
    
    
    printf("\n");
    print_array(f_new, size, size);
    // float* Image = malloc(64*64*sizeof(float));
    // Image = read_csv2("../data/house.csv",64,64);
    // print_array(Image,64,64);

    // free(Image);
    free(f_new);
    free(X);
    
    return 0;
}