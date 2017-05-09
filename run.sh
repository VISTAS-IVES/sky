#!/bin/bash
#SBATCH --job-name tinyAE
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-cpu 512
#SBATCH --partition long
#SBATCH --output=output.txt
module load Python/intel-python3.5.2
srun python tinyAE.py
