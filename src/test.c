#include "utilities.h"

int main(void){

    float X[25] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};

    int size = 5;

    int patch_size = 5;
    int row = 0;
    int col = 0;

    float *pad = pad_array(X,5,5,5);

    printf("\n");
    // for (int i=0; i<size+patch_size-1; i++){
    //     for (int j=0; j<size+patch_size-1; j++){
    //         printf("%f ", pad[i*(size+patch_size-1) + j]);
    //     }
    //     printf("\n");
    // }
    float *patch = patch_finder(pad,patch_size,row,col,size,size);
    for (int i=0; i<patch_size; i++){
        for (int j=0; j<patch_size; j++){
            printf("%f ", patch[i*patch_size + j]);
        }
        printf("\n");
    }
    free(pad);
    free(patch);
    return 0;
}