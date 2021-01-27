#include "utilities.h"
#include <time.h>

float* denoise_image(float *F, int rows, int cols, int patch_size, float patch_sigma,  float filter_sigma){
    // Make the padded array
    float *pad = pad_array(F, rows, cols, patch_size);

    // Make the array that stores all the patches of the pixels
    int number_of_pixels = rows*cols;

    float** Gaussian_Patches = (float**)malloc(number_of_pixels*sizeof(float*));
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            Gaussian_Patches[i*cols+ j] = patch_finder(pad, patch_size, i, j, rows, cols);
            Gaussian_Patches[i*cols + j] = gaussian_Filtering(Gaussian_Patches[i*cols + j], patch_size, patch_sigma);
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

    return f_new;
}

int main(void){

    int size = 5;
    float *X = (float*)malloc(size*size*sizeof(float));

    srand(time(NULL));
    for(int i=0; i<size*size; i++){
        X[i] = (float)rand()/RAND_MAX;
    }

    // print_array(X, size, size);

    int patch_size = 3;
    float filter_sigma = 0.1;
    float patch_sigma = 1;

    float *f_new = denoise_image(X, size, size, patch_size, patch_sigma, filter_sigma);

    // print_array(f_new, size, size);

    free(f_new);
    free(X);
    
    return 0;
}