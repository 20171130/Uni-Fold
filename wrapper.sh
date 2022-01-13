#!/bin/bash
lrank=$OMPI_COMM_WORLD_LOCAL_RANK
case ${lrank} in
[0])
  export HIP_VISIBLE_DEVICES=0
  numactl --cpunodebind=0 --membind=0 singularity run --bind "/public/home/fold_test/LuDh,$UNIFOLD_DIR" /public/home/fold_test/LuDh/singularity-images/jax_0.2.27-rocm4.1.sif  python3 $UNIFOLD_DIR/train.py;;
[1])
  export HIP_VISIBLE_DEVICES=1
  numactl --cpunodebind=1 --membind=1 singularity run --bind "/public/home/fold_test/LuDh,$UNIFOLD_DIR" /public/home/fold_test/LuDh/singularity-images/jax_0.2.27-rocm4.1.sif  python3 $UNIFOLD_DIR/train.py;;
[2])
  export HIP_VISIBLE_DEVICES=2
  numactl --cpunodebind=2 --membind=2 singularity run --bind "/public/home/fold_test/LuDh,$UNIFOLD_DIR" /public/home/fold_test/LuDh/singularity-images/jax_0.2.27-rocm4.1.sif  python3 $UNIFOLD_DIR/train.py;;
[3])
  export HIP_VISIBLE_DEVICES=3
  numactl --cpunodebind=3 --membind=3 singularity run --bind "/public/home/fold_test/LuDh,$UNIFOLD_DIR" /public/home/fold_test/LuDh/singularity-images/jax_0.2.27-rocm4.1.sif  python3 $UNIFOLD_DIR/train.py;;
esac

