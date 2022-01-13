#!/bin/bash
#SBATCH -J dp
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -p kshdexclu07
#SBATCH -o unifold-%J.log
#SBATCH -e unifold-%J.err
#SBATCH -t 48:00:00
#SBATCH --gres=dcu:4
#SBATCH --mem-per-cpu=25g
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=1

source ./env.sh

/public/home/fold_test/LuDh/openmpi_4.0.3/bin/mpirun -np 8 ./wrapper.sh

