"""A demo of AlphaFold protein structure training script."""

# OS & MPI config. please config before any import of jax / tf.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='True'
from unifold.train.train_config import train_config
use_mpi = train_config.global_config.use_mpi
if use_mpi:   # configurating MPI. please do this before importing jax.
  from mpi4py import MPI
  mpi_comm = MPI.COMM_WORLD
  mpi_rank = mpi_comm.Get_rank()
  is_main_process = (mpi_rank == 0)
  #os.environ['HIP_VISIBLE_DEVICES']= str(mpi_rank)
  os.environ['CUDA_VISIBLE_DEVICES']= str(mpi_rank)
else:         # assume single gpu is used.
  mpi_comm = None
  mpi_rank = 0
  is_main_process = True
# external import


from absl import logging
import pdb
from multiprocessing import Queue

# internal import
from unifold.model.config import model_config as get_model_config
from unifold.train.data_system import DataSystem, GetBatchProcess
from unifold.train.utils import get_queue_item
from unifold.train.trainer import Trainer

def train(train_config):
  """
  main function of training (single gpu).
  """
  # get configs
  gc = train_config.global_config
  model_config = get_model_config(gc.model_name, is_training=True)
  # construct datasets
  logging.info("constructing train data ...")
  train_data = DataSystem(model_config, train_config.data.train)
  logging.info("constructing valid data ...")
  try:
    eval_data = DataSystem(model_config, train_config.data.eval)
  except:
    logging.warning("failed to load valid data. consider poor config.")
    eval_data = None
    
  # create batch processes
  train_queue = Queue(gc.max_queue_size)
  train_batch_proc = GetBatchProcess(
      queue=train_queue,
      data=train_data,
      num_batches=gc.end_step - gc.start_step + 1,  # add 1 for the initialization batch
      is_training=True,
      random_seed=gc.random_seed,
      mpi_rank=mpi_rank)                            # pass rank to generate different batches among mpi.
  train_batch_proc.start()
  
  if eval_data is not None:
    eval_queue = Queue(gc.max_queue_size)
    eval_batch_proc = GetBatchProcess(
        queue=eval_queue,
        data=eval_data,
        num_batches=(gc.end_step - gc.start_step) // gc.eval_freq + 1,
        is_training=False,
        random_seed=gc.random_seed,
        mpi_rank=mpi_rank)                          # pass rank to generate different batches among mpi.
    eval_batch_proc.start()

  # define and initialize trainer
  trainer = Trainer(
      global_config=gc,
      optim_config=train_config.optimizer,
      model_config=model_config,
      mpi_comm=mpi_comm)
  logging.info("initializing ...")
  _, init_batch = get_queue_item(train_queue)    # do NOT use the returned rng to initialize trainer.
  trainer.initialize(init_batch)
  
  # conduct training
  logging.info("training ...")
  for step in range(gc.start_step, gc.end_step):
    update_rng, batch = get_queue_item(train_queue)
    trainer.train_step(step, batch, update_rng, silent=(not is_main_process))
    if eval_data is not None and trainer.is_eval_step(step):
      eval_rng, batch = get_queue_item(eval_queue)
      trainer.eval_step(step, batch, eval_rng, silent=(not is_main_process))
  logging.info("finished training.")

  if train_batch_proc.is_alive():
    train_batch_proc.terminate()
  if eval_data is not None and eval_batch_proc.is_alive():
    eval_batch_proc.terminate()


if __name__ == "__main__":
  LOG_VERBOSITY = {
    'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG
  }
  if is_main_process:
    logging.set_verbosity(LOG_VERBOSITY[train_config.global_config.verbose.upper()])
  else:
    logging.set_verbosity(logging.ERROR)
  logging.info(train_config)
  train(train_config)

