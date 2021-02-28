SHELL := /bin/bash

CC = gcc -O3
NVCC = nvcc
RM = rm -f

EXEC = main global shared

all : $(EXEC)


main : ./serial/main.c
	$(CC) -o main ./serial/main.c -lm

shared: ./cuda_shared/shared.cu
	$(NVCC) -o shared ./cuda_shared/shared.cu -lm

global: ./cuda_global/global.cu
	$(NVCC) -o global ./cuda_global/global.cu -lm

clean:
		$(RM) *.o *~ $(EXEC)
default:
		all
