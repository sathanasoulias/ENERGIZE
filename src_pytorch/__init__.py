"""
PyTorch implementation of OpenNILM

This package provides PyTorch implementations of NILM (Non-Intrusive Load Monitoring)
models including CNN, GRU, and TCN architectures.
"""

from .models import CNN_NILM, GRU_NILM, TCN_NILM, get_model
from .data_loader import NILMDataset, DataLoaderNILM, SimpleNILMDataLoader
from .trainer import Trainer, train_model, EarlyStopping, ModelCheckpoint, TrainingHistory
from .tester import Tester, SimpleTester, load_model
from .config import (
    MODEL_CONFIGS,
    TRAINING,
    CALLBACKS,
    DATASET_CONFIGS,
    DATASET_SPLITS,
    REFIT_PARAMS,
    PLEGMA_PARAMS,
    get_appliance_params,
    get_model_config,
    get_dataset_config,
    get_dataset_split
)
from .utils import (
    set_seeds,
    create_experiment_directories,
    get_device,
    count_parameters,
    print_model_summary,
    save_checkpoint,
    load_checkpoint
)

__version__ = '1.0.0'
__all__ = [
    # Models
    'CNN_NILM',
    'GRU_NILM',
    'TCN_NILM',
    'get_model',

    # Data
    'NILMDataset',
    'DataLoaderNILM',
    'SimpleNILMDataLoader',

    # Training
    'Trainer',
    'train_model',
    'EarlyStopping',
    'ModelCheckpoint',
    'TrainingHistory',

    # Testing
    'Tester',
    'SimpleTester',
    'load_model',

    # Config
    'MODEL_CONFIGS',
    'TRAINING',
    'CALLBACKS',
    'DATASET_CONFIGS',
    'DATASET_SPLITS',
    'REFIT_PARAMS',
    'PLEGMA_PARAMS',
    'get_appliance_params',
    'get_model_config',
    'get_dataset_config',
    'get_dataset_split',

    # Utils
    'set_seeds',
    'create_experiment_directories',
    'get_device',
    'count_parameters',
    'print_model_summary',
    'save_checkpoint',
    'load_checkpoint',
]
