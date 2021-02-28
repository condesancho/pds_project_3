#!/bin/bash
#SBATCH --job-name=DeviceQuery
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00

sbatch batch1_cushared.sh 
sbatch batch2_cuglobal.sh
sbatch batch3_serial.sh
