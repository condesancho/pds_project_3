
SHELL := /bin/bash

CC = gcc -O3
NVCC = nvcc
RM = rm -f

EXEC = test_shared test_serial test_shared

all : $(EXEC)

test_serial : test_serial 
	$(CC) $< -o ./$@  -lm


test_shared:  test_shared
	$(NVCC) $< -o ./$@  -lm

test_global: test_global	
	$(NVCC) $< -o ./$@  -lm

clean:
		$(RM) *.o *~ $(EXEC)
default:
		all


