#!/bin/bash

#######################################
# dependencies of feature processing  #
#######################################

conda install -y -c bioconda hmmer hhsuite==3.3.0 kalign2
conda install -y -c conda-forge \
      openmm=7.5.1 \
      pdbfixer

# work_path=/path/to/unifold-code
work_path=$(pwd)

# update openmm
a=$(which python)
cd $(dirname $(dirname $a))/lib/python3.8/site-packages
patch -p0 < $work_path/docker/openmm.patch


#######################################
# dependencies of training Uni-Fold   #
#######################################

conda install -y -c nvidia cudnn==8.0.4
pip install --upgrade pip \
    && pip install -r ./requirements.txt \
    && pip install jaxlib==0.1.67+cuda111 -f \
    https://storage.googleapis.com/jax-releases/jax_releases.html

# install openmpi. this is optional for single gpu tasks.
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz
gunzip -c openmpi-4.1.1.tar.gz | tar xf -
cd openmpi-4.1.1
./configure --prefix=/usr/local
make all install

