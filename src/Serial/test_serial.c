#include "utilities.h"
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

int main(int argc, char* argv[]){

    float patch_sigma = 5/2;
    float filt_sigma = 0.01;

    int patch_size = 0;
    int rows = 0;
    int cols = 0;

    if(argc < 2){
        printf("\nUsage: %s [No. dimensions] [Patch size]\n", argv[0]);
        exit(-1);
    }

    rows = atoi(argv[1]);
    cols = atoi(argv[1]);
    patch_size = atoi(argv[2]);

    float *X = (float*)malloc(rows*cols*sizeof(float));

    srand(time(NULL));

    for(int i=0; i<rows*cols; i++){
        X[i] = (float)rand()/RAND_MAX;
    }

    printf("\nCurrently in %s running random image with %d dimension and patch size %d.\n", argv[0], rows, patch_size);

    struct timespec begin, end;

    // Starting the clock
    clock_gettime(CLOCK_MONOTONIC, &begin);

    float* f_new = denoise_image(X, rows, cols, patch_size, patch_sigma, filt_sigma);

    // Stopping the clock
    clock_gettime(CLOCK_MONOTONIC, &end);
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;

    printf("\nTime elapsed: %.5f seconds.\n\n", elapsed);
    
    printf("Done with %s that run a random image with dimension %d and patch size %d.\n\n", argv[0], rows, patch_size);
    
    free(f_new);
    free(X);
    
    return 0;
}