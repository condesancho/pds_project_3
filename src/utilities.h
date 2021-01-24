#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

//#include "G:\Program Files\MATLAB\R2018a\extern\include\mat.h"

//all matrices are in row major format

// float *F = matOpen("../code/data/house.mat",)

/*patch finder function*/

float* patch_finder(float *F, int patch_size, int x_central, int y_central, int x_border, int y_border){

    //allocate space for the patch
    float* Patch = (float *)malloc(patch_size*patch_size* sizeof(float));
    //find the patch

    /*todo implemnt different algorithms for searching patch depending on the central pixel position*/
    
    /*if the patch is not getting out of bounds*/
    int x0_patch = x_central + 1 - patch_size / 2;
    int y0_patch = y_central + 1 - patch_size / 2;
    if (abs(x_central - x_border) > patch_size / 2 && abs(y_central - y_border) > patch_size / 2 && (x_central-(patch_size / 2)) > 0 && (y_central-(patch_size / 2)) > 0)
    {
        for (int i = 0; i < patch_size; i++)
        {
            for (int j = 0; j < patch_size; j++)
            {
                //to complex to analyze this algorithm but it works trust us we are engineers 
                Patch[i*patch_size + j] = F[x0_patch*i + y0_patch + j + patch_size*i + 1 + patch_size];
            }
        }
    }//else if (/* condition */)
    // {
    //     //todo mirror the patch parts that are out of bounds.  
    // }
    
    
    
   

    return Patch;
}
float *matToRowMajor(float** matrix, int n, int m){
        float *RowMajor = malloc(n*m*sizeof(float));
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
    float *dataRowMajor = malloc(row*col*sizeof(float));
            
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
//         printf("%f", a[i*COLS + j])
//     }
    
    
// }
