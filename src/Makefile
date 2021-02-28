SHELL := /bin/bash

CC = gcc -O3
NVCC = nvcc
RM = rm -f

EXEC = test_shared test_serial test_global shared NLM kernel

all : $(EXEC)

test_serial : test_serial.c 
	$(CC) -o test_serial .\Serial\test_serial.c -lm


test_shared:  test_shared.cu
	$(NVCC) -o test_shared .\cuda_shared\test_shared.cu -lm

test_global: test_global.cu	
	$(NVCC) -o test_global .\CUDA\test_global.cu -lm


NLM : NLM.c 
	$(CC) -o NLM .\Serial\NLM.c -lm


shared:  shared.cu
	$(NVCC) -o shared .\cuda_shared\shared.cu -lm

kernel: kernel.cu	
	$(NVCC) -o kernel .\CUDA\kernel.cu -lm

clean:
		$(RM) *.o *~ $(EXEC)
default:
		all
