#!/bin/bash
#SBATCH --job-name clouds
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 20
#SBATCH --mem-per-cpu 2048
#SBATCH --partition medium
#SBATCH --output=output.txt

module load Python/intel-python3.5.2
echo $options
srun --unbuffered python net.py $options