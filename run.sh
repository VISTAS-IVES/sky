#!/bin/bash
#SBATCH --job-name clouds
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 20
#SBATCH --mem-per-cpu 2048
#SBATCH --partition long
#SBATCH --output=output.txt

module load Python/intel-python3.5.2
echo $options
srun --unbuffered python Tensor_flow_practice.py $options


