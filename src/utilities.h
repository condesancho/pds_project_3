#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

//#include "G:\Program Files\MATLAB\R2018a\extern\include\mat.h"

// All matrices are in row major format

// float *F = matOpen("../code/data/house.mat",)

/*patch finder function*/

float* patch_finder(float *F, int patch_size, int row_central, int col_central, int rows, int cols){

    //allocate space for the patch
    float* Patch = (float *)malloc(patch_size*patch_size* sizeof(float));
    //find the patch

    /*todo implemnt different algorithms for searching patch depending on the central pixel position*/
    
    // Finding the position of the top left patch point in the F array
    int first_row = row_central - patch_size / 2;
    int first_col = col_central - patch_size / 2;
    int last_row = row_central + patch_size / 2 + 1;
    int last_col = col_central + patch_size / 2 + 1;
    // Stores the index of the first element of the patch in F
    int first_elem = first_row*rows + first_col;

    /*When the patch is not getting out of bounds*/
    if (first_col > -1 && first_row > -1 && last_col < cols+1 && last_row < rows+1){
        for (int i = 0; i < patch_size; i++){
            for (int j = 0; j < patch_size; j++){
                Patch[i*patch_size + j] = F[first_elem + i*rows + j];
            }
        }
    }

    else{

        // The patch exceeds the upper left corner
        if(first_row < 0 && first_col < 0){
            // Helpful variables for making the extensions
            int index = 0;
            int extension_cols = patch_size+first_col;
            int extension_rows = patch_size+first_row;

            float *upper_extension = (float*)malloc(abs(first_row)*extension_cols*sizeof(float));
            for(int i=abs(first_row)-1; i>-1; i--){
                for(int j=0; j<extension_cols; j++){
                    upper_extension[i*extension_cols+j] = F[index];
                    index++;
                }
                // Goes to the next row
                index += cols-extension_cols;
            }

            index = 0;
            float *up_left_extension = (float*)malloc(abs(first_col)*abs(first_row)*sizeof(float));
            for(int i=abs(first_row)-1; i>-1; i--){
                for(int j=abs(first_col)-1; j>-1; j--){
                    up_left_extension[i*abs(first_col)+j] = F[index];
                    index++;
                }
                // Goes to the next row
                index += cols+first_col;
            }

            index = 0;
            float *left_extension = (float*)malloc(abs(first_col)*extension_rows*sizeof(float));
            for(int i=0; i<extension_rows; i++){
                for(int j=abs(first_col)-1; j>-1; j--){
                    left_extension[i*abs(first_col)+j] = F[index];
                    index++;
                }
                // Goes to the next row
                index += cols+first_col;
            }

            index = 0;
            for (int i = 0; i < patch_size; i++){
                for (int j = 0; j < patch_size; j++){
                    if(i<abs(first_row) && j<abs(first_col)){
                        Patch[i*patch_size + j] = up_left_extension[i*abs(first_col)+j];
                    }
                    else if(i<abs(first_row) || j<abs(first_col)){
                        if(i<abs(first_row)){
                            Patch[i*patch_size + j] = upper_extension[index];
                            index++;
                        } 
                        else{
                            Patch[i*patch_size + j] = left_extension[index];
                            index++;
                        }
                        
                    }
                    else{
                        Patch[i*patch_size + j] = F[first_elem + i*rows + j];
                    }
                }
                if(i==abs(first_row)-1) index = 0;
            }
            
            free(upper_extension);
            free(up_left_extension);
            free(left_extension);

        }
        // The patch exceeds the upper right corner
        else if(first_row < 0 && last_col > cols){
            // Helpful variables for making the extensions
            int index = 0;
            int extension_cols = patch_size-(last_col-cols);
            int extension_rows = patch_size+first_row;

            index = first_col;
            float *upper_extension = (float*)malloc(abs(first_row)*extension_cols*sizeof(float));
            for(int i=abs(first_row)-1; i>-1; i--){
                for(int j=0; j<extension_cols; j++){
                    upper_extension[i*extension_cols+j] = F[index];
                    index++;
                }
                // Goes to the next row
                index += cols-extension_cols;
            }

            index = cols - (last_col-cols);
            float *up_right_extension = (float*)malloc(abs(first_col)*abs(first_row)*sizeof(float));
            for(int i=abs(first_row)-1; i>-1; i--){
                for(int j=abs(first_col)-1; j>-1; j--){
                    up_right_extension[i*abs(first_col)+j] = F[index];
                    index++;
                }
                // Goes to the next row
                index += cols+first_col;
            }

            index = cols - (last_col-cols);
            float *right_extension = (float*)malloc((last_col-cols)*extension_rows*sizeof(float));
            for(int i=0; i<extension_rows; i++){
                for(int j=(last_col-cols)-1; j>-1; j--){
                    right_extension[i*abs(first_col)+j] = F[index];
                    index++;
                }
                // Goes to the next row
                index += cols-(last_col-cols);
            }

            index = 0;
            int upper_right_index = 0;
            for (int i = 0; i < patch_size; i++){
                for (int j = 0; j < patch_size; j++){
                    if(i<abs(first_row) && j>last_col-cols){
                        Patch[i*patch_size + j] = up_right_extension[upper_right_index];
                        upper_right_index++;
                    }
                    else if(i<abs(first_row) || j>last_col-cols){
                        if(i<abs(first_row)){
                            Patch[i*patch_size + j] = upper_extension[index];
                            index++;
                        } 
                        else{
                            Patch[i*patch_size + j] = right_extension[index];
                            index++;
                        }
                        
                    }
                    else{
                        Patch[i*patch_size + j] = F[first_elem + i*rows + j];
                    }
                }
                if(i==abs(first_row)-1) index = 0;
            }
            
            free(upper_extension);
            free(up_right_extension);
            free(right_extension);
        }
        // The patch exceed upper row
        else if(first_row < 0){
            // Helpful variables for making the extensions
            int index = 0;
            int extension_cols = patch_size+first_col;

            float *upper_extension = (float*)malloc(abs(first_row)*extension_cols*sizeof(float));
            index = first_col;
            for(int i=abs(first_row)-1; i>-1; i--){
                for(int j=0; j<extension_cols; j++){
                    upper_extension[i*extension_cols+j] = F[index];
                    index++;
                }
                // Goes to the next row
                index += cols-extension_cols;
            }

            index=0;
            for (int i = 0; i < patch_size; i++){
                for (int j = 0; j < patch_size; j++){
                    if(i<abs(first_row)){
                        Patch[i*patch_size + j] = upper_extension[index];
                        index++;
                    }
                    else{
                        Patch[i*patch_size + j] = F[first_elem + i*rows + j];
                    }
                }
            }

            free(upper_extension);
        }
        else{
            printf("I didn't do anything.\n");
        }
        
    }

    return Patch;
}

float *matToRowMajor(float** matrix, int n, int m){
        float *RowMajor = (float*)malloc(n*m*sizeof(float));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                RowMajor[i*n+j] = matrix[i][j];
            }
            
        }
        
        return RowMajor; 
}
 
float* read_csv(int row, int col, char *filename){
	FILE *file;
	file = fopen(filename, "r");
    if (file == NULL)
    {
       fprintf(stderr,"error fopen(): Failed to open CSV file");
    }
    
    float **data = (float **)malloc(row*sizeof(float*));
    for (int i = 0; i < row; i++)
    {
        data[i] = (float*)malloc(row*sizeof(float));
    }
    float *dataRowMajor = (float*)malloc(row*col*sizeof(float));
            
    int count = 0;
    char c;
    for (c = getc(file); c != EOF; c = getc(file)){ 
    // Increment count for this character 
        count = count + 1; 
    }
    char buffer[count];
    char *record, *line;
    int i=0,j=0;
    while((line=fgets(buffer,sizeof(buffer),file))!=NULL)
    {
        record = strtok(line,",");
        while(record != NULL)
        {
            printf("%s\t",record) ;    //here you can put the record into the array as per your requirement.
            printf("\n");
            data[i][j++] = atof(record);
            record = strtok(NULL,",");
            
        }
        ++i ;
    }

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%f,", data[i][j]);
        }
        printf("\n");
    }
    
    
    dataRowMajor = matToRowMajor(data, row, col);
    for (int i = 0; i < row; i++)
    {
        free(data[i]);
    }
    free(data);
    return dataRowMajor;
    
}






// double* nonLocalMeans(float **F, int patch_size, float filter_sigma, float patch_sigma){
//     float** Patch = (float *)malloc(patch_size * sizeof(float*));
//     for (int i = 0; i <//toimport size of the image; i++)
//     {
//         Patch[i] = malloc(patch_size * sizeof(float));
//     }
    
//     Patch = patch_finder(F,patch_size);


//     //apply gaussian filter to patch 
    
//     Gaussian_Patch = gaussian_Filtering(Patch, patch_size, patch_sigma);


//     free(Patch);
    
// }


// float** gaussian_Filtering(float** P, int size, float sigma){
//     float* gauss = malloc(size*size*sizeof(float));
//     float sum = 0;
//     for (int i = 0; i < size; i++)
//     {
//         for (int j = 0; j < size; j++)
//         {
//             float rows[size] = P[]
//             float columns[size] = 
//         }
        
//     }
    
// }
// for (int i = 0; i < COLS; i++)
// {
//     for (int j = 0; j < ROWS; j++)
//     {
//         for (int j = 0; j < size; j++)
//         {
//             float rows[size] = P[]
//             float columns[size] = 
//         }
        
//     }
    
// }
// // for (int i = 0; i < COLS; i++)
// // {
// //     for (int j = 0; j < ROWS; j++)
// //     {
// //         printf("%f", a[i*COLS + j])
// //     }
    
    
// // }
