#include "../utilities.h"
#include "nlm.h"
#include <time.h>


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