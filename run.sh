#!/bin/bash
#SBATCH --job-name cloudsphi
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 20
#SBATCH --mem-per-cpu 4096
#SBATCH --partition phi
#SBATCH --output=output.txt

module load Python/3.6.2/intel
echo $options
srun --unbuffered python train.py $options
