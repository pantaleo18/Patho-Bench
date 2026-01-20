import os
import shutil
import numpy as np
import torch
from pathlib import Path
from torch import nn
from patho_bench.experiments.FinetuningExperiment import FinetuningExperiment
from patho_bench.experiments.GeneralizabilityExperimentWrapper import GeneralizabilityExperimentWrapper
from patho_bench.TrainableSlideEncoder import (
    DefaultTrainableSlideEncoder, 
    im4MECTrainableSlideClassifier, 
    ABMIL_MILLAB_TrainableSlideClassifier,
    CLAMTrainableSlideClassifier
)
from patho_bench.SplitFactory import SplitFactory
from patho_bench.DatasetFactory import DatasetFactory
from patho_bench.helpers.GPUManager import GPUManager
from patho_bench.optim.NLLSurvLoss import NLLSurvLoss
from sklearn.utils.class_weight import compute_class_weight
from trident.slide_encoder_models.load import encoder_factory
import warnings
import json
from millab.builder import create_model
from itertools import product
import copy

import textwrap

COMBINE_TRAIN_VAL = False
TEST_EXTERNAL_ONLY = True

class ExperimentFactory:
                
    @staticmethod
    def finetune(split: str,
                 task_config: str,
                 patch_embeddings_dirs: list[str],
                 saveto: str,
                 combine_slides_per_patient: bool,
                 bag_size,
                 gradient_accumulation,
                 num_epochs,
                 balanced: bool,
                 save_which_checkpoints: str,
                 model_args: dict = {},
                 precision : str = None,
                 gpu : int = -1,
                 device_batch_size : int = 1, 
                 seed : int = None,
                 disable_cudnn : bool = False,
                 scheduler_config : dict = None,
                 optimizer_config : dict = None,
                 external_split: str = None,
                 external_saveto: str = None,
                 num_bootstraps: int = 100,
                 view_progress: str = 'bar',
                 color_map : str | dict = None,
                 lr_logging_interval : int = 1,
                 early_stop : bool = False,
                 early_stop_policy : str = "last-1",
                 patience : int = 3,
                 halt_training_on_folder_early_stop : bool = False,
                 **kwargs,
        ):

        if kwargs.get('batch_size'):
            warnings.warn(
                "The 'batch_size' argument is deprecated and has been renamed to 'device_batch_size'. "
                "It controls the physical batch size, i.e., the number of WSIs processed in parallel. "
                "The total effective batch size is calculated as device_batch_size * accumulation_steps. "
                "For backward compatibility, self.device_batch_size is set to the provided 'batch_size' value."
            )
            device_batch_size = kwargs['batch_size']

        base_learning_rate = model_args.pop('lr', 1e-3)

        split, task_info, internal_dataset = ExperimentFactory._prepare_internal_dataset(
            split_path=split,
            task_config=task_config,
            saveto=saveto,
            combine_slides_per_patient=combine_slides_per_patient,
            combine_train_val=COMBINE_TRAIN_VAL,
            patch_embeddings_dirs=patch_embeddings_dirs,
            gpu=gpu,
            bag_size=bag_size
        )

        loss = configure_loss(
            task_type = task_info['task_type'], 
            balanced = balanced,
            split = split,
            task_name = task_info['task_col']
        )
        
        model_constructor, classifier_args = configure_model(
            model_args,
            loss,
            task_info
        )
        
        optimizer_config = configure_optimizer(
            optimizer_config,
            base_learning_rate=base_learning_rate,
            batch_size=device_batch_size,
            gradient_accumulation=gradient_accumulation
        )
        
        scheduler_config = configure_scheduler(scheduler_config)
        
        if isinstance(color_map,str):
            with open(color_map,"r") as fp:
                color_map = json.load(fp)

        
        experiment = FinetuningExperiment(
            task_type = task_info['task_type'],
            dataset = internal_dataset,
            device_batch_size = device_batch_size,
            model_constructor = model_constructor,
            classifier_args = classifier_args,
            num_epochs = num_epochs,
            gradient_accumulation = gradient_accumulation,
            optimizer_config = optimizer_config,
            scheduler_config = scheduler_config,
            save_which_checkpoints = save_which_checkpoints,
            num_bootstraps = num_bootstraps,
            precision = get_precision_type(precision),
            device = f'cuda:{gpu if gpu != -1 else GPUManager.get_best_gpu(min_mb=500)}',
            results_dir = saveto,
            view_progress = view_progress,
            color_map = color_map,
            lr_logging_interval = lr_logging_interval,  
            early_stop = early_stop,
            early_stop_policy=early_stop_policy,
            patience = patience,
            halt_training_on_folder_early_stop=halt_training_on_folder_early_stop,
            seed=seed,
            disable_cudnn=disable_cudnn
        )
        
        if external_split is None:
            return experiment
        else:
            print('\033[91mWARNING: Generalizability experiment is not yet tested for finetuning. Proceed with caution.\033[0m')
            external_dataset = ExperimentFactory._prepare_external_dataset(
                                                                external_split,
                                                                task_config,
                                                                internal_dataset.num_folds,
                                                                patch_embeddings_dirs,
                                                                combine_slides_per_patient,
                                                                bag_size = bag_size)
            return GeneralizabilityExperimentWrapper(
                experiment,
                external_dataset=external_dataset,
                test_external_only=TEST_EXTERNAL_ONLY,
                saveto=external_saveto
            )

    @staticmethod
    def sweep(experiment_type: str,
              saveto_root: str,
              combine_slides_per_patient: bool,
              sweep_over: dict[list],
              gpu: int = -1,
              external_split: str = None,
              external_saveto: str = None,
              num_bootstraps: int = 100,
              view_progress: str = 'bar',
              color_map : str | dict = None,
              early_stop : bool = False,
              early_stop_policy : str = "best-val-loss",
              patience : int = 3,
              halt_training_on_folder_early_stop : bool = False,
              seed : bool = None,
              disable_cudnn : bool = False,
            ):
        
        num_configs = _sweep_welcome(sweep_over)
        
        # Build the base arguments to pass to the experiment factory.
        args = {
            'combine_slides_per_patient': combine_slides_per_patient,
            'gpu': gpu,
            'view_progress' : view_progress,
            'external_split': external_split,
            'external_saveto': external_saveto,
            'num_bootstraps': num_bootstraps,
            'color_map' : color_map,
            'early_stop' : early_stop,
            'patience' : patience,
            'early_stop_policy' : early_stop_policy,
            'halt_training_on_folder_early_stop' : halt_training_on_folder_early_stop,
            'seed' : seed,
            "disable_cudnn" : disable_cudnn
        }

        experiments_list = []
        # Iterate over all combinations of hyperparameters.
        for i, hyperparams in enumerate(generate_arg_combinations(sweep_over)):

            _sweep_section(f"STARTING CONFIGURATION {i + 1} / {num_configs}", 140)

            # Create a unique experiment directory from the hyperparameters.
            args['saveto'] = setup_folder_configs(
                saveto_root=saveto_root,
                hyperparams=hyperparams,
                id = i
            )

            # Unpacking dataset parameters
            dataset_cfg = hyperparams.pop('dataset')
            hyperparams['split'] = str(Path(dataset_cfg['split']).expanduser())
            hyperparams['task_config'] = str(Path(dataset_cfg['task_config']).expanduser())
            hyperparams['patch_embeddings_dirs'] = [
                str(Path(p).expanduser()) for p in dataset_cfg['patch_embeddings_dirs']
            ]

            hyperparams['balanced'] = dataset_cfg['balance_loss']

            # Unpacking batch size
            batch_size = hyperparams.pop('batch_size')
            if batch_size is not None:
                hyperparams['device_batch_size'] = batch_size['device_batch_size']
                hyperparams['gradient_accumulation'] = batch_size['gradient_accumulation']

            # Start fine tuning
            if args['saveto'] is not None:
                if experiment_type == 'finetune':
                    experiment = ExperimentFactory.finetune(**args, **hyperparams)
                else: 
                    raise NotImplementedError(
                        f'Experiment type {experiment_type} not recognized.'
                    )
            
                experiments_list.append(experiment.train())
            
        return experiments_list

    @staticmethod
    def _prepare_internal_dataset(split_path: str,
                                  task_config: str,
                                  saveto: str,
                                  combine_slides_per_patient: bool,
                                  combine_train_val: bool,
                                  patch_embeddings_dirs: list[str],
                                  pooled_embeddings_dir: str = None,
                                  model_name: str = None,
                                  model_kwargs: dict = {},
                                  bag_size: int = None,
                                  gpu: int = -1):
        """
        Helper method to prepare the internal dataset from slide embeddings or patch embeddings.
        """
        # Load split
        split, task_info = SplitFactory.from_local(split_path, task_config)
        if combine_train_val:
            split.replace_folds('val', 'train')
        split.save(os.path.join(saveto, 'split.csv'), row_divisor='slide_id')  # Save split to experiment folder for future reference
        
        # Load dataset
        if pooled_embeddings_dir is not None:
            dataset = DatasetFactory.from_slide_embeddings(
                split=split,
                task_name=task_info['task_col'],
                pooled_embeddings_dir=pooled_embeddings_dir,
                patch_embeddings_dirs=patch_embeddings_dirs,
                combine_slides_per_patient=combine_slides_per_patient,
                model_name=model_name,
                model_kwargs=model_kwargs,
                gpu=gpu
            )
        else:
            dataset = DatasetFactory.from_patch_embeddings(
                split=split,
                task_name=task_info['task_col'],
                patch_embeddings_dirs=patch_embeddings_dirs,
                combine_slides_per_patient=combine_slides_per_patient,
                bag_size=bag_size
            )
        return split, task_info, dataset

    @staticmethod
    def _prepare_external_dataset(external_split_path: str,
                                  task_config: str,
                                  internal_num_folds: int,
                                  patch_embeddings_dirs: list[str],
                                  combine_slides_per_patient: bool,
                                  external_pooled_embeddings_dir: str = None,
                                  model_name: str = None,
                                  model_kwargs: dict = {},
                                  bag_size: int = None,
                                  gpu: int = -1):
        """
        Helper method to prepare the external dataset (all test) from slide or patch embeddings for generalizability experiments.
        """
        external_split, task_info = SplitFactory.from_local(external_split_path, task_config)
        external_split.remove_all_folds()
        external_split.assign_folds(num_folds=internal_num_folds, test_frac=1, val_frac=0, method='monte-carlo')  # Reassign all samples to test
        
        if external_pooled_embeddings_dir is not None:
            return DatasetFactory.from_slide_embeddings(
                split=external_split,
                task_name=task_info['task_col'],
                pooled_embeddings_dir=external_pooled_embeddings_dir,
                patch_embeddings_dirs=patch_embeddings_dirs,
                combine_slides_per_patient=combine_slides_per_patient,
                model_name=model_name,
                model_kwargs=model_kwargs,
                gpu=gpu
            )
        else:
            return DatasetFactory.from_patch_embeddings(
                split=external_split,
                task_name=task_info['task_col'],
                patch_embeddings_dirs=patch_embeddings_dirs,
                combine_slides_per_patient=combine_slides_per_patient,
                bag_size=bag_size
            )

    @staticmethod
    def _get_model_constructor(model_name : str):
        if "millab" in model_name:
            if "abmil" in model_name:
                return ABMIL_MILLAB_TrainableSlideClassifier
            if "clam" in model_name:
                return CLAMTrainableSlideClassifier
        elif model_name == 'im4MEC':
            return im4MECTrainableSlideClassifier
        else :
            return DefaultTrainableSlideEncoder

def configure_model(model_args : dict, loss, task_info : dict):
    
    model_name = model_args['model_name']

    model_constructor = ExperimentFactory._get_model_constructor(model_name)
    model_name_cleaned = model_name.replace("millab:", "")
    num_classes = len(task_info['label_dict'])

    if model_constructor is ABMIL_MILLAB_TrainableSlideClassifier:
        
        slide_classifier = create_model(model_name_cleaned, from_pretrained=True, num_classes=num_classes)

        classifier_args = {
            'slide_classifier': slide_classifier,
            'post_pooling_dim': slide_classifier.model.classifier.in_features,
            'task_name': task_info['task_col'],
            'num_classes': num_classes,
            'loss': loss,
            'label_dict' : task_info['label_dict'],
            **model_args
        }
    
    else : 
        slide_encoder = encoder_factory( 
            pretrained = False, 
            freeze=False, 
            **model_args
        )

        classifier_args = {
            'slide_encoder': slide_encoder,
            'post_pooling_dim': slide_encoder.embedding_dim,
            'task_name': task_info['task_col'],
            'num_classes': num_classes,
            'loss': loss,
            'label_dict' : task_info['label_dict'],
            **model_args
        }

    return model_constructor, classifier_args
        
def parse_task_code(task_code):

    data_source, task_name = task_code.split('--')
    if '==' in data_source:
        train_source, test_source = data_source.split('==') # If running generalizability experiment, load split for internal dataset only
        assert train_source != test_source, f'train_source and test_source must be different when formatting task_code as "train_source==test_source--task_name". Did you mean to use {train_source}--{task_name} instead of {task_code}?'
        return train_source, test_source, task_name     
    else:
        train_source = data_source
        return train_source, None, task_name
    
def generate_exp_id(hyperparams):
    return '_'.join(sorted([f'{k}={v}' for k, v in hyperparams.items()]))
    
def generate_arg_combinations(variables):
    from itertools import product
    # If cost = 'auto', then automatically sweep over a range of costs (intended for linprobe)
    if 'auto' in make_list(variables.get('COST')):
        cost_list = list(np.logspace(np.log10(10e-6), np.log10(10e5), num=45))
        variables['cost'] = cost_list
        if len(make_list(variables['COST'])) != 1:
            raise ValueError(
                "If setting cost to 'auto', then only one cost value is allowed."
            )

    # Lowercase keys and ensure all values are lists
    variables = {k.lower(): make_list(v) for k, v in variables.items()}

    # Generate all combinations and deepcopy each value
    combinations = []
    for combo in product(*variables.values()):
        combo_dict = {k: copy.deepcopy(v) for k, v in zip(variables.keys(), combo)}
        combinations.append(combo_dict)

    return combinations
        
def make_list(x):
    return x if isinstance(x, list) else [x]

def configure_loss(
        task_type : str, 
        balanced : bool,
        split,
        task_name : str,
):
    if task_type == 'survival':
        loss = NLLSurvLoss(alpha=0.0, eps=1e-7, reduction='mean')
    elif balanced:
        # Balanced loss is a dict of losses for each fold
        fold_weights = {fold: compute_class_weight('balanced', classes = np.array(sorted(split.unique_classes(task_name))), y = split.y(task_name, fold, 'train')) for fold in range(split.num_folds)}
        loss = {fold: nn.CrossEntropyLoss(weight = torch.from_numpy(weights).float()) for fold, weights in fold_weights.items()}
    else:
        loss = nn.CrossEntropyLoss()

    return loss

def configure_scheduler(custom_config: dict = None):
    json_path = os.path.join(
        os.path.dirname(__file__),
        "config",
        "scheduler.json"
    )
    
    with open(json_path, "r") as f:
        default_configs = json.load(f)
    
    scheduler_type = custom_config.get("type", "cosine") if custom_config else "cosine"

    if scheduler_type not in default_configs:
        raise NotImplementedError(
            f"Scheduler type '{scheduler_type}' not implemented"
        )
    
    scheduler_config = default_configs[scheduler_type].copy()

    if custom_config:
        scheduler_config.update(custom_config)
    else:
        warnings.warn(
            f"No custom configuration set for scheduler. Using default configuration: {scheduler_config}",
            RuntimeWarning
        )
    
    return scheduler_config

def configure_optimizer(
    custom_config: dict = None,
    base_learning_rate: float = None,
    batch_size: int = None,
    gradient_accumulation: int = None
):
    json_path = os.path.join(
        os.path.dirname(__file__),
        "config",
        "optimizer.json"
    )

    with open(json_path, "r") as f:
        default_configs = json.load(f)

    optimizer_type = custom_config.get("type", "AdamW") if custom_config else "AdamW"

    if optimizer_type not in default_configs:
        raise NotImplementedError(
            f"Optimizer type '{optimizer_type}' not implemented"
        )

    # Config strutturale (JSON + override)
    optimizer_config = default_configs[optimizer_type].copy()

    # Adding learning rate
    if base_learning_rate is not None:
        optimizer_config["base_lr"] = base_learning_rate

    if custom_config:
        optimizer_config.update(custom_config)
    else:
        warnings.warn(
            f"No custom configuration set for optimizer. Using default configuration: {optimizer_config}",
            RuntimeWarning
        )

    # Special case for Patho-Bench custom optimizer
    if optimizer_type == 'gigapath':
        from patho_bench.optim.GigaPathOptim import param_groups_lrd
        optimizer_config = {
            'type': 'AdamW',
            'base_lr': base_learning_rate * ((batch_size * gradient_accumulation) / 256),
            'get_param_groups': param_groups_lrd,
            'param_group_args': {
                'layer_decay': custom_config.get('layer_decay',0),
                'no_weight_decay_list': [],
                'weight_decay': custom_config.get('weight_decay',0)
            },
        }

    return optimizer_config

def deep_dict_exact_equal(d1, d2):
    if type(d1) != type(d2):
        return False
    
    if isinstance(d1, dict):
        if d1.keys() != d2.keys():
            return False
        for k in d1:
            if not deep_dict_exact_equal(d1[k], d2[k]):
                return False
        return True
    
    elif isinstance(d1, list):
        if len(d1) != len(d2):
            return False
        return all(deep_dict_exact_equal(a, b) for a, b in zip(d1, d2))
    
    else:
        return d1 == d2 
    
def setup_folder_configs(saveto_root,id,hyperparams):
    # Directory per questa configurazione
    this_config_path = os.path.join(saveto_root, str(id))

    if os.path.exists(this_config_path):
        hyper_file = os.path.join(this_config_path, 'hyperparameters.json')
        if os.path.exists(hyper_file):
            with open(hyper_file, 'r') as f:
                saved_hyperparams = json.load(f)

            if deep_dict_exact_equal(saved_hyperparams, hyperparams):
                warnings.warn(
                    f"Exact same configuration already exists at {this_config_path}.\n"
                    f"Skipping this run.\n{hyperparams = }"
                )
                return None
            else:
                warnings.warn(
                    f"Folder {this_config_path} exists but configuration differs. "
                    f"Deleting folder and creating a new one.\n"
                    f"Saved hyperparams: {saved_hyperparams}\n"
                    f"Current hyperparams: {hyperparams}"
                )
                shutil.rmtree(this_config_path)  # rimuovi la vecchia cartella
                os.makedirs(this_config_path)
        else:
            warnings.warn(
                f"{this_config_path} exists but no hyperparameters.json found. Overwriting."
            )
            shutil.rmtree(this_config_path)
            os.makedirs(this_config_path)
    else:
        os.makedirs(this_config_path)

    # Save hyperparameters to hyperparameters.json
    with open(os.path.join(this_config_path, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    return this_config_path

def _sweep_section(title: str, width: int = 140):
    print("\n" + "-" * width)
    print(f"{title:^{width}}")
    print("-" * width)

def _sweep_welcome(sweep_over: dict, width: int = 140):
    # Header principale
    print("\n" + "=" * width)
    print(f"{'WELCOME TO SWEEP MODE':^{width}}")
    print("=" * width)

    # Descrizione
    description = (
        "Welcome to ComPaSIO's sweep mode.\n"
        "A grid search will be performed over the specified hyperparameters."
    )
    for line in description.split("\n"):
        print(f"{line:^{width}}")

    print("=" * width)

    # Sweep info
    sweep_keys = list(sweep_over.keys())
    num_configs = len(list(generate_arg_combinations(sweep_over)))

    print("\n" + "-" * width)
    print(f"{'SWEEP CONFIGURATION':^{width}}")
    print("-" * width)

    # Stampa ogni chiave e i valori puntati
    for k in sweep_keys:
        print(f"{k}:")
        for v in sweep_over[k]:
            # indent e bullet
            value_str = str(v)
            # wrap se troppo lungo
            wrapped = textwrap.wrap(value_str, width=width-6)
            for i, line in enumerate(wrapped):
                prefix = "    - " if i == 0 else "      "
                print(prefix + line)
        print()  # linea vuota tra chiavi

    print(f"Total configurations to run: {num_configs}")

    print("\n" + "=" * width + "\n")

    return num_configs

def get_precision_type(precision : str):
    if "bfloat16" == precision : 
        return torch.bfloat16
    elif "float16" == precision:
        return torch.float16
    elif "float64" == precision:
        return torch.float64
    elif "float32" == precision:
        return torch.float32
    else:
        # Default
        return torch.float32