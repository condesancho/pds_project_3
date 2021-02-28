#include "utilities.h"
#include <time.h>

float* denoise_image(float *F, int rows, int cols, int patch_size, float patch_sigma, float filter_sigma){
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

    int rows = 0;
    int cols = 0;
    int patch_size = 0;

    float patch_sigma = 5/2;
    float filter_sigma = 0.1;

    //If flag == 0 then the user gave a file as an input
    int flag = 0;

    float* image;

    if(argc < 4){
        printf("\nNot enough arguments!\nUsage: %s [Path to file] [No. image rows] [No. image cols] [Patch size]\n", argv[0]);
        printf("\t*****OR*****");
        printf("\nUsage: %s [No. image rows] [No. image cols] [Patch size]\n\n", argv[0]);
        exit(-1);
    }

    if (argc == 5){
        rows = atoi(argv[2]);
        cols = atoi(argv[3]);
        patch_size = atoi(argv[4]);
    }
    else if (argc==4){
        rows = atoi(argv[1]);
        cols = atoi(argv[2]); 
        patch_size = atoi(argv[3]);

        //The user wants to use a random array
        flag = 1;
    }
    else{
        printf("Too many arguments\n");
        exit(-1);
    }

    if(!flag){
        printf("\nCurrently in %s running %s with %d pixel rows, %d pilex cols and patch size %d.\n", argv[0], argv[1], rows, cols, patch_size);
        image = read_csv2(argv[1], rows, cols);
        printf("\nArray in file successfully read\n");
    }
    else{
        printf("\nCurrently in %s running random image with %d pixel rows, %d pilex cols and patch size %d.\n", argv[0], rows, cols, patch_size);
        
        srand(time(NULL));

        image = (float*)malloc(rows*cols*sizeof(float));

        for (int i=0; i<rows*cols; i++){
            image[i] = (float)rand()/RAND_MAX;
        }
        printf("\nRandom array complete\n");
    }


    struct timespec begin, end;

    // Starting the clock
    clock_gettime(CLOCK_MONOTONIC, &begin);

    float* new_image = denoise_image(image, rows, cols, patch_size, patch_sigma, filter_sigma);

    // Stopping the clock
    clock_gettime(CLOCK_MONOTONIC, &end);
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;

    printf("\nTime elapsed: %.5f seconds.\n\n", elapsed);

    if(!flag){
        
        char denoised_name[] = "denoised.csv";

        create_csv(denoised_name, new_image, rows, cols);

        float* residual = (float*)malloc(rows*cols*sizeof(float));

        for (int i=0; i<rows*cols; i++){
            residual[i] = new_image[i]-image[i];
        }

        char residual_name[] = "residual.csv";

        create_csv(residual_name, residual, rows, cols);

        free(residual);

        printf("Done with %s that run %s with %d pixel rows, %d pilex cols and patch size %d.\n", argv[0], argv[1], rows, cols, patch_size);
    }
    else{
        printf("Done with %s that run a random image with %d pixel rows, %d pilex cols and patch size %d.\n", argv[0], rows, cols, patch_size);
    }

    free(new_image);
    free(image);

    return 0;
}