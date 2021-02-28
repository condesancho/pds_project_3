#include "../utilities.h"
#include "util_global.h"


/**
 * Function that takes an image and filters the noise.
 * 
 * Inputs:
 *      float* h_X          --> The image to be filtered
 *      int rows            --> The pixel rows of the image
 *      int cols            --> The pixel cols of the image
 *      int patch_size      --> The dimension of the square patch dezired
 *      int patch_sigma     --> The sigma used to find the gaussian kernel weights
 *      int filter_sigma    --> The sigma used in the non local means algorithm
 * 
 * Output:
 *  The new filtered image
 * 
 * The CPU pads the image and calulates the gaussian kernel. These arrays are copied to
 * the global memory of the GPU. There is also allocated an array of equal size to the image,
 * where the filtered image is stored. The kernel above is called and then the filtered image
 * is passed to the CPU.
 */
 float* denoise_image(float* h_X, int rows, int cols, int patch_size, float patch_sigma, float filter_sigma){
    
    float** gaussian = gaussian_Kernel(patch_size, patch_sigma);
    float* h_gaussian = matToRowMajor(gaussian, patch_size, patch_size);

    // Make padded
    float* h_padded = pad_array(h_X, rows, cols, patch_size);
    int padded_rows = rows + patch_size -1;
    int padded_cols = cols + patch_size -1;

    float* h_filtered = (float*)malloc(rows*cols*sizeof(float));

    float* d_padded;
    float* d_filtered;
    float* d_gaussian;

    // Allocate GPU memory
    cudaMalloc(&d_padded, padded_rows*padded_cols*sizeof(float));
    cudaMalloc(&d_filtered, rows*cols*sizeof(float));
    cudaMalloc(&d_gaussian, patch_size*patch_size*sizeof(float));

    // Pass the padded array and the gaussian weights in the GPU
    cudaMemcpy(d_padded, h_padded, padded_rows*padded_cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gaussian, h_gaussian, patch_size*patch_size*sizeof(float), cudaMemcpyHostToDevice);

    // Number of thread and blocks
    int noThreads = 20;
    int noBlocks = (rows*cols)/noThreads +1;
    
    filter_image<<<noBlocks, noThreads>>>(d_filtered, d_padded, d_gaussian, rows, cols, patch_size, filter_sigma);

    // Return the filtered image in the CPU
    cudaMemcpy(h_filtered, d_filtered, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_padded);
    cudaFree(d_gaussian);
    cudaFree(d_filtered);

    cudaDeviceSynchronize();

    free(h_padded);
    free(h_gaussian);

    return h_filtered;
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
        
        char denoised_name[13] = "denoised";

        create_csv(denoised_name, new_image, rows, cols);

        float* residual = (float*)malloc(rows*cols*sizeof(float));

        for (int i=0; i<rows*cols; i++){
            residual[i] = new_image[i]-image[i];
        }

        char residual_name[13] = "residual";

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