
SHELL := /bin/bash

CC = gcc -O3
NVCC = nvcc
RM = rm -f

EXEC = test_shared test_serial test_shared

all : $(EXEC)

test_serial : test_serial.c 
	$(CC) $< -o ./$@  -lm


test_shared:  test_shared.cu
	$(NVCC) $< -o ./$@  -lm

test_global: test_global.cu	
	$(NVCC) $< -o ./$@  -lm

clean:
		$(RM) *.o *~ $(EXEC)
default:
		all


