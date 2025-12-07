#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=00:12:00
#SBATCH --partition=tornado

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module load mpi/openmpi/4.1.6/gcc/11
mpirun temp_calc_mpi


mpicc temp_calc_mpi.c -o temp_calc_mpi -lm
mpirun ./temp_calc_mpi
