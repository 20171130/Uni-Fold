from ml_collections import ConfigDict


train_config = ConfigDict({
    'global_config':{
        # whether you are using MPI communication for multi-gpu training.
        'use_mpi': True,
        # This specifies a model config defined in `unifold/model/config.py`, 'model_1' to 'model_5' are the settings used in the AlphaFold2 paper.
        # You can also set this value to 'demo' or 'small' for fast demonstration or customize your own model in `unifold/model/config.py` and cite it here.
        'model_name': 'unifold',
        # verbosity of logging messages.
        'verbose': 'info',
        'debug': False,
        # initial step. if > 0, the model will attempt to auto-load ckpts from `auto_load_dir`.
        'start_step': 0,                # 0 by default
        # max steps for training. accumulated from 'start_step' instead of 0.
        'end_step': 1000,                # 80000 in af2
        # frequency of logging messages and the training loss curve.
        'logging_freq': 10,
        # frequency of validation.
        'eval_freq': 50,
        # frequency of saving ckpts.
        'save_freq': 100,
        # directory to save ckpts. used for auto-saving ckpts.
        'save_dir': './out/ckpt',
        # directory to load ckpts. used for auto-loading ckpts. ignored if start_step == 0.
        'load_dir': './out/ckpt',
        # precision. generally in ['fp32', 'fp16', 'bf16']. set for mixed precision training.
        'precision': 'fp32',
        # max queue size. specifies the queue size of the pre-processed batches. generally has little impact on code efficiency.
        'max_queue_size': 16,
        # random seed for initializing model parameters. ignored when attempting to auto load ckpts.
        'random_seed': 181129
    },
    'optimizer': {
        # optimizer class.
        'name': 'adam',                 # only 'adam' supported
        # learning rate. if warm up steps > 0, this specifies the peak learning rate. 
        'learning_rate': 1e-3,          # 1e-3 in af2
        # the number of warm-up steps.
        'warm_up_steps': 1000,            # 1000 in af2
        # learning rate decay configs.
        'decay':{
            'name': 'exp',              # only 'exp' supported
            'decay_rate': 0.95,         # 0.95 in af2
            'decay_steps': 2000           # 5000? in af2
        },
        # global clip norm of gradients.
        'clip_norm': 1e-1,
    },
    'data':{
        'train': {
            # directory to store features (features.pkl files)
            'features_dir': "./example_data/features",
            # directory to store labels (.mmcif files)
            'mmcif_dir': "./example_data/mmcif",
            # json file that specifies sampling weights of each sample.
            'sample_weights': "./example_data/sample_weights.json",
            # whether inavailable residues in pdb/mmcif were masked out in the input sequences. note that this param is set as true only for trrosetta data. please set it as false if other data are used.
            'use_mask':True
        },
        'valid': {
            # directory to store features (features.pkl files)
            'features_dir': "./example_data/features",
            # directory to store labels (.mmcif files)
            'mmcif_dir': "./example_data/mmcif",
            # json file that specifies sampling weights of each sample.
            'sample_weights': "./example_data/sample_weights.json",
            # whether inavailable residues in pdb/mmcif were masked out in the input sequences. note that this param is set as true only for trrosetta data. please set it as false if other data are used.
            'use_mask':True
        },
    }
}
)
