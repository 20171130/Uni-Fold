# Uni-Fold: Training your own AlphaFold.

This package modifies the DeepMind AlphaFold v2.0 and provides training code to reproduce the results from scratch. See the 
[AlphaFold paper](https://doi.org/10.1038/s41586-021-03819-2), [Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf), and the [original repository](https://github.com/deepmind/unifold).

To train your own protein folding models, please follow the steps below:

## 1. Installing the environment

Uni-Fold has been tested for Python 3.8.10, CUDA 11.1 and OpenMPI 4.1.1. Run the following code to install the dependencies to run Uni-Fold:

```bash
  conda create -n unifold python=3.8.10 -y
  conda activate unifold
  ./install_dependencies.sh
```

## 2. Feature processing before training

#### 2.1 Datasets and external tools

#### 2.2 Running the code

## 3. Training Uni-Fold

#### 3.1 Configuration
You can adjust the training configuration and model configuration in `unifold/train/training_config.py` and `unifold/model/config.py` respectively. You should change the directories in the 'data' section in `train_config.py` to be the path to your dataset.

#### 3.2 Preprocessing input features from fasta files

#### 3.2 Running the code
To train the model on a single node without MPI, run
```bash
python train.py
```
You can also train the model using MPI (or workload managers that supports MPI, such as PBS or slurm).
```bash
mpiexec -N <number of nodes> --pernode <gpus per node> python train.py
```

## 4. Inference with Uni-Fold

#### 4.1 Inference from fasta files

#### 4.2 Inference from features.pkl

<!-- 
## 1. Installing the environment.

Create a Conda environment and install the dependencies via:
```bash
conda create -n unifold python=3.8.10 -y
conda activate unifold
bash install_on_local.sh
```

## 2. Specifying training configurations.
Before you conduct any actual training processes, please make sure that you correctly configured the code.

1. Modify the training configurations in `unifold/train/train_config.py`. We annotated the default configurations to reproduce AlphaFold in the script. Specifically, modify the data setups in `unifold/train/train_config.py`:
    
    ```json
    "data": {
        "train": {
            "features_dir": "where/training/protein/features/are/stored/",
            "mmcif_dir": "where/training/mmcif/files/are/stored/",
            "sample_weights": "which/specifies/proteins/for/training.json"
        },
        "eval": {
            "features_dir": "where/validation/protein/features/are/stored/",
            "mmcif_dir": "where/validation/mmcif/files/are/stored/",
            "sample_weights": "which/specifies/proteins/for/training.json"
        }
    }
    ```
    
    The specified data should be contained in two folders, namely a `features_dir` and a `mmcif_dir`. The former directory should contain pickle files with names `[pdb_id]_[model_id]_[chain_id]/features.pkl`, for example, `1ak0_1_A/features.pkl`. The latter should contain mmcif files with name `[pdb_id].cif`, for example, `1ak0.cif`. Make sure that two directories contain consistent proteins, that is, each protein with features should have its `pdb_id` included in the mmcif directory.

    If you want to specify the list of training data under the directories, write a json file and feed the path to `sample_weights`. This is optional, as you can leave it as `None` (and the program will attempt to use all entries under `features_dir` with uniform weights). The json file should be a dictionary contains the name of protein ([pdb_id]\_[model_id]\_[chain_id]) and the sample weight of each protein in the training process (Optional, integer or float), such as:
    ```json
    {"1am9_1_C": 82, "1amp_1_A": 291, "1aoj_1_A": 60, "1aoz_1_A": 552}
    ```
    or for uniform sampling, simply using a list of protein entries suffices:

    ```json
    ["1am9_1_C", "1amp_1_A", "1aoj_1_A", "1aoz_1_A"]
    ```

    Also, if you are a member of DP Technology and have access to the `oss` files, you can use the TrRosetta dataset with path `/tmp/ossfs-user/data/chenweijie/data/.`

## 3. Start training!
Start to train AlphaFold on a single GPU by running:
```bash
python train.py
```

## 4. Modify the model config
You can modify the configurations of model such as number of layers, number of channels in `unifold/model/config.py`. Other training details such as the print and save frequencies, the hyperparameters of optimizer, etc. are available in `unifold/train/train_config.py`.

You can also register your own model configurations. See the demo case of `MY_MODEL_CONFIG` (named as `'my_model_name'`) in the annotations of `unifold/model/config.py`. -->
