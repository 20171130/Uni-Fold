#!/bin/bash
module unload mpi
module load apps/singularity/3.7.3 

export MPI_DIR=/public/home/fold_test/LuDh/openmpi_4.0.3
export OMPI_DIR=$MPI_DIR
export PATH=$OMPI_DIR/bin:$PATH
export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export MANPATH=$OMPI_DIR/share/man:$MANPATH
export UNIFOLD_DIR=$(pwd)