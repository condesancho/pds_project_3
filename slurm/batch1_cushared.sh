#!/bin/bash
#SBATCH --job-name=DeviceQuery
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00
module load gcc/7.3.0 cuda/10.0.130
make all 
for i  in 3 5 7  
do
    for k in 64 128 256
    do 
        ./test_shared $k $i
    done
done
