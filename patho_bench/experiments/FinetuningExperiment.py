import os
import numpy as np
import torch
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import json
import warnings

from torch.optim import Adam
from torch.optim import SGD
from torch.optim import AdamW

from patho_bench.datasets.BaseDataset import BaseDataset
from patho_bench.experiments.BaseExperiment import BaseExperiment
from patho_bench.experiments.utils.LoggingMixin import LoggingMixin
from patho_bench.experiments.utils.ClassificationMixin import ClassificationMixin
from patho_bench.experiments.utils.SurvivalMixin import SurvivalMixin

import math 
import warnings
import textwrap

# Turn off tokenizer parallelism to avoid warnings from dataloader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FinetuningExperiment(LoggingMixin, ClassificationMixin, SurvivalMixin, BaseExperiment):

    EPOCH_W = 7
    PHASE_W = 12
    DURATION_W = 21
    WIDTH_EXP = 120
    WIDTH_FOLD = 100

    def __init__(self,
                 task_type: str,
                 dataset: BaseDataset,
                 device_batch_size: int,
                 model_constructor: callable,
                 classifier_args: dict,
                 num_epochs: int,
                 gradient_accumulation: int,
                 optimizer_config: dict,
                 scheduler_config: dict,
                 save_which_checkpoints: str,
                 num_bootstraps: int,
                 precision: torch.dtype,
                 device: str,
                 results_dir: str,
                 view_progress: str = 'bar',
                 seed: int = None,
                 disable_cudnn : bool = False,
                 color_map : dict = None,
                 early_stop : bool = False,
                 early_stop_policy : str = "last-1",
                 patience : int = 3,
                 halt_training_on_folder_early_stop : bool = False,
                 **kwargs):

        self.task_type = task_type
        self.dataset = dataset
        self.device_batch_size = device_batch_size
        self.gradient_accumulation = gradient_accumulation
        self._batch_size = self.device_batch_size * self.gradient_accumulation
        self.model_constructor = model_constructor
        self.model_kwargs = classifier_args
        self.num_epochs = num_epochs
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.num_bootstraps = num_bootstraps
        self.precision = precision
        self.device = device
        self.results_dir = results_dir
        self.view_progress = view_progress
        self.seed = seed
        self.disable_cudnn = disable_cudnn
        self.set_seed(self.seed,self.disable_cudnn)
        self.color_map = color_map
        self.early_stop = early_stop
        self.early_stop_policy = early_stop_policy
        self.patience = patience
        self.halt_training_on_folder_early_stop = halt_training_on_folder_early_stop
        self.target_score = None
        self.save_which_checkpoints = self._define_saving_policy(save_which_checkpoints=save_which_checkpoints)

        # Set kwargs as extra attributes for saving in config.json
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def _define_saving_policy(self,save_which_checkpoints):

        if save_which_checkpoints == 'best-val-loss':
            if self.dataset.get_subset(iteration=0, fold='val') is None:
                warnings.warn(
                    "'best-val-loss' requested but the validation dataset is empty. "
                    "Falling back to 'last-1'."
                )
                save_which_checkpoints = 'last-1'

        elif save_which_checkpoints.startswith("best-"):
            _save_policy = save_which_checkpoints[len("best-"):]
            
            if _save_policy == "step_on":
                if self.scheduler_config.get('type') == 'plateau':
                    self.target_score = self.scheduler_config.get('step_on')
                    save_which_checkpoints = f"best-{self.target_score}"
                else:
                    warnings.warn(
                        "'best-step_on' requested but the scheduler type is not 'plateau'. "
                        "Falling back to 'last-1'."
                    )
                    save_which_checkpoints = 'last-1'
            
            elif _save_policy in ClassificationMixin.SCALAR_SCORES:
                self.target_score = _save_policy
            
            else:
                warnings.warn(
                    f"Save policy '{_save_policy}' is invalid. Falling back to 'last-1'."
                )
                save_which_checkpoints = 'last-1'

        return save_which_checkpoints

    def train(self):

        self.save_config(os.path.join(self.results_dir, 'pathobench_config.json'))
        self.train_results_dir = self.results_dir 
        self.durations = {}
        
        self.experiment_report()
        
        ### Loop through folds
        for self.current_fold in range(self.dataset.num_folds):

            self.mode = "train"
            self.loggers = self.init_loggers(save_dir = os.path.join(self.results_dir, 'training_metrics', f'fold_{self.current_fold}'))

            ### Initialize train and val dataloaders
            self.dataloaders = {mode: self.dataset.get_dataloader(self.current_fold, mode, batch_size=self.device_batch_size, seed = self.seed) for mode in ['train', 'val']}
            self.num_phisical_batches_train = len(self.dataloaders['train'])
            self.num_batches_train = math.ceil(self.num_phisical_batches_train / self.gradient_accumulation)

            self.num_phisical_batches_val = len(self.dataloaders['val'])
            self.num_batches_val = math.ceil(self.num_phisical_batches_val / self.gradient_accumulation)
            
            ### Initialize model (type: TrainableSlideEncoder)
            self.model = self.model_constructor(**self.model_kwargs, device = self.device)
            self.save_model_architecture(self.model, os.path.join(self.results_dir, f'model.txt'))
            
            ### Initialize optimizer and scheduler
            self.optimizer = self._init_optimizer()
            self.scheduler = self._init_scheduler()

            self.global_opt_steps = 0 
            self._monitor_scaling_factors_ga = set() if self.gradient_accumulation > 1 else {1}

            ### Prepare grad scaler
            # Only use GradScaler for FP16 training. bfloat16 does not require GradScaler: https://discuss.pytorch.org/t/bfloat16-training-explicit-cast-vs-autocast/202618/8
            try:
                self.grad_scaler = torch.amp.GradScaler('cuda', enabled = (self.precision == torch.float16)) 
            except:
                # Legacy (torch 2.0.0) implementation for compatibility with Gigapath
                self.grad_scaler = torch.cuda.amp.GradScaler(enabled = (self.precision == torch.float16))

            ### Initialize metrics
            self.best_val_loss = float('inf')
            self.best_target_score = float('-inf') 
            self.best_smooth_rank = 0

            ### Prepare epoch loop
            if self.view_progress == 'bar':
                self.loop = tqdm(range(self.num_epochs))
            elif self.view_progress == 'verbose':
                self.loop = range(self.num_epochs)
            else:
                # Warning e fallback
                warnings.warn(f"view_progress must be 'bar' or 'verbose', got '{self.view_progress}'. Falling back to no progress display.")
                self.view_progress = None
                self.loop = range(self.num_epochs)

            impatient_counter = 0
            prev_gen_error = float('inf') 
            early_stop_trigger = False
            self.fold_training_epochs = 0 

            self.durations[self.current_fold] = []

            self.training_report(validation_scores = None)

            # TRAINING LOOP
            if self.view_progress == "verbose":
                self._print_training_header()
            
            for self.current_epoch in self.loop:
        
                self.fold_training_epochs += 1 

                # epoch = 0,1,...,num_epochs
                epoch_loss = {'train' : None, 'val' : None }
                new_best_loss = {'train' : None, 'val' : None}
                new_best_target = {'train' : None, 'val' : None}
                total_duration = {'train': 0, 'val': 0}

                for self.mode in ['train', 'val']:
                
                    if self.dataloaders[self.mode] is not None:
                        
                        if self.view_progress == 'bar':
                            self.loop.set_description(f'Epoch {self.current_epoch} {self.mode}')

                        
                        start = time.time()
                        new_best_loss[self.mode], epoch_loss[self.mode], new_best_target[self.mode], _ = \
                            self._run_single_epoch()
                        end = time.time()

                        self._lr_step(
                            update = (
                                (self.scheduler_config['step_on'] == "epoch") and # <--- if step @ epochs 
                                (self.scheduler_config['type'] != "plateau") and # <- plateau may need validation metrics
                                (self.mode == "train") # <-- only when training
                            ), 
                            metric = None,
                            step = self.current_epoch
                        )                        
    
                        duration = end - start
                        if self.view_progress == "verbose": 
                            self._print_training_row(
                                epoch=self.current_epoch + 1,
                                phase=self.mode,
                                duration_seconds=duration
                            )
                        # Store duration
                        total_duration[self.mode] += duration

                # Save duration
                self.durations[self.current_fold].append({
                        "train": total_duration['train'],
                        "val": total_duration['val'],
                })

                current_gen_err = epoch_loss['val'] - epoch_loss['train']
                
                if self.early_stop :

                    if self.early_stop_policy ==  "best-checkpoint":
                        impatient_counter = impatient_counter + 1 if not new_best_target['val'] else 0
                    
                    elif self.early_stop_policy == "best-val-loss" :
                        impatient_counter = impatient_counter + 1 if not new_best_loss['val'] else 0
                    
                    elif self.early_stop_policy == "er":
                        if not new_best_loss['val'] and current_gen_err > prev_gen_error:
                            impatient_counter += 1
                        else:
                            impatient_counter = 0

                    else :
                        warnings.warn(f"{self.early_stop_policy} not implemented yet. Early stop will have no effect")
                        impatient_counter = -1
                            
                    # Running out of patience?
                    early_stop_trigger = impatient_counter > self.patience
                    
                    prev_gen_error = current_gen_err
            
                    if early_stop_trigger:
                        warnings.warn(
                            f"Early stop criteria ({self.early_stop_policy}) met at "
                            f"fold =  {self.current_fold + 1}, epoch =  {self.current_epoch + 1}."
                        )
                        break

            # Deprecated
            if self.halt_training_on_folder_early_stop:
                warnings.warn(f"halt_training_on_early_stop is deprecated")

            # === VALIDATE THIS FOLD ===
            validation_scores = self._eval_single_fold(fold_idx=self.current_fold)
            self.training_report(validation_scores)
              
        json_path = os.path.join(self.results_dir, "durations.json")
        with open(json_path, "w") as f:
            json.dump(self.durations, f, indent=4)

        validation_summary = self.validate()

        self.experiment_report(
            validation_summary= validation_summary
        )

    def _eval_single_fold(self, fold_idx: int):

        # Carica dataloader del fold
        eval_dataloader = self.dataloaders.get('val')
        if eval_dataloader is None:
            return None

        # Directory checkpoint del fold
        checkpoint_dir = os.path.join(self.results_dir, 'checkpoints', f'fold_{fold_idx}')
        try:
            ckpt_path = self._pick_checkpoint(checkpoint_dir)
        except FileNotFoundError:
            warnings.warn(
                f"No checkpoint found for fold {fold_idx + 1}. Skipping evaluation."
            )
            return None

        # Load e freeze modello
        try:
            model = self.model_constructor(**self.model_kwargs, device=self.device)
            model = self.load_checkpoint(model, ckpt_path)
            model = self.freeze(model)
        except Exception as e:
            warnings.warn(
                f"Failed to load checkpoint for fold {fold_idx + 1}: {e}. Skipping."
            )
            return None

        # Accumula predizioni
        labels, preds, ids = self._accumulate_preds(eval_dataloader, model)

        # Caso standard: salva metriche per-fold
        per_fold_save_dir = os.path.join(
            self.results_dir, f'val_metrics', f'fold_{fold_idx}'
        )
        os.makedirs(per_fold_save_dir, exist_ok=True)

        scores = self._compute_metrics(labels, preds, per_fold_save_dir, ids=ids)
        return scores

    def test(self):
        '''
        Evaluate the model on the test set for each fold.
        '''
        self._eval(split='test')

    def validate(self):
        all_scores_across_folds = []

        for fold_idx in range(self.dataset.num_folds):
            metrics_path = os.path.join(
                self.results_dir, 'val_metrics', f'fold_{fold_idx}', 'metrics.json'
            )
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    scores = json.load(f)
                all_scores_across_folds.append(scores['overall'])  # Prendi solo il dict 'overall'

        if not all_scores_across_folds:
            warnings.warn("No per-fold metrics found. Summary cannot be computed.")
            return

        summary = {}

        for k, v in all_scores_across_folds[0].items():
            
            if isinstance(v, dict):
                continue

            vals = np.array([fold[k] for fold in all_scores_across_folds], dtype=float)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))
            se = float(std / np.sqrt(len(vals)))

            summary[k] = {
                "mean": mean,
                "std": std,
                "se": se,
                "formatted": f"{mean:.3f} ± {std:.3f}"
            }

        # Salva summary finale
        summary_path = os.path.join(self.results_dir, 'val_metrics_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Validation summary saved to {summary_path}")
        
        return summary
        
    def _eval(self, split: str):

        all_labels_across_folds = []
        all_preds_across_folds = []
        all_scores_across_folds = []

        ### Loop through folds which have been computed.
        loop = tqdm(range(self.dataset.num_folds))
        for self.current_fold in loop:
            ### Load the dataloader for this fold
            eval_dataloader = self.dataset.get_dataloader(self.current_fold, split, batch_size=1, seed = self.seed)
            if eval_dataloader is None:
                return
            loop.set_description(f'Running {split} split on {len(eval_dataloader.dataset)} samples')

            ### Get latest saved checkpoint for this fold
            checkpoint_dir = os.path.join(self.results_dir, 'checkpoints', f'fold_{self.current_fold}')
            try:
                ckpt_path = self._pick_checkpoint(checkpoint_dir)
            except FileNotFoundError:
                warnings.warn(f"No checkpoint found for fold {self.current_fold + 1}. Skipping this fold for {split} evaluation.")
                continue

            ### Load the model and freeze it
            try:
                model = self.model_constructor(**self.model_kwargs, device=self.device)
                model = self.load_checkpoint(model, ckpt_path)
                model = self.freeze(model)
            except Exception as e:
                warnings.warn(f"Failed to load checkpoint for fold {self.current_fold + 1}: {e}. Skipping this fold.")
                continue

            ### Gather labels and predictions
            labels, preds, ids = self._accumulate_preds(eval_dataloader, model)  # MV: ids added for downstream analyses

            ### Decide whether to report per-fold results (mean ± SD) or bootstrapped results (95% CI)
            if len(eval_dataloader.dataset) == 1 or self.dataset.num_folds == 1:
                # If only one fold or one sample per fold, will save results at end across all folds
                all_labels_across_folds.append(labels)
                all_preds_across_folds.append(preds)
            else:
                # If multiple folds and multiple samples per fold, save per-fold results
                per_fold_save_dir = os.path.join(self.results_dir, f'{split}_metrics', f'fold_{self.current_fold}')
                os.makedirs(per_fold_save_dir, exist_ok=True)  # MV added
                scores = self._compute_metrics(labels, preds, per_fold_save_dir, ids=ids)  # MV ids added
                all_scores_across_folds.append(scores)

        # After collecting all folds, either do bootstrapping or an average across folds
        summary = self._finalize_metrics(split, all_labels_across_folds, all_preds_across_folds, all_scores_across_folds)

        with open(os.path.join(self.results_dir, f'{split}_metrics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
    def _accumulate_preds(self, dataloader, model):

        labels_all = []
        preds_all = []
        ids_all = []

        model.eval()

        with torch.inference_mode(), torch.autocast(
            device_type='cuda',
            dtype=self.precision,
            enabled=self.precision != torch.float32
        ):
            for batch in dataloader:

                # ---------- IDS ----------
                raw_ids = batch.get('ids', None)
                if raw_ids is None:
                    batch_ids = ["UNKNOWN"] * len(next(iter(batch['labels'].values())))
                else:
                    batch_ids = [str(x) for x in raw_ids]

                ids_all.extend(batch_ids)

                # ---------- CLASSIFICATION ----------
                if self.task_type == 'classification':

                    # labels: (B,)
                    labels = batch['labels'][self.model_kwargs['task_name']]
                    labels = labels.cpu().int().numpy()

                    # logits: (B, C)
                    logits = model(batch, output='logits')
                    if isinstance(logits,tuple):
                        logits = logits[-1][0]['logits']

                    # probs: (B, C)
                    probs = torch.softmax(logits, dim=1)
                    probs = probs.cpu().numpy()

                    labels_all.extend(labels.tolist())
                    preds_all.extend(probs)

                # ---------- SURVIVAL ----------
                elif self.task_type == 'survival':

                    events = batch['labels']['extra_attrs'][
                        f'{self.model_kwargs["task_name"]}_event'
                    ].cpu().numpy()

                    times = batch['labels']['extra_attrs'][
                        f'{self.model_kwargs["task_name"]}_days'
                    ].cpu().numpy()

                    logits = model(batch, output='logits')  # (B, *)
                    risks = self._calculate_risk(logits)    # (B,)

                    risks = risks.cpu().numpy()

                    labels_all.extend(
                        [{"survival_event": e, "survival_time": t}
                        for e, t in zip(events, times)]
                    )
                    preds_all.extend(risks)

        # ---------- FINAL CONVERSION ----------
        if self.task_type == 'classification':
            labels_all = np.asarray(labels_all)
            preds_all = np.asarray(preds_all)

        elif self.task_type == 'survival':
            labels_all = {
                "survival_event": np.asarray([x["survival_event"] for x in labels_all]),
                "survival_time": np.asarray([x["survival_time"] for x in labels_all]),
            }
            preds_all = np.asarray(preds_all)

        return labels_all, preds_all, ids_all
    
    def _compute_metrics(self, labels, preds, save_dir = None, ids=None):  # MV ids added

        if self.task_type == 'classification':
            self.auc_roc(labels, preds, self.model_kwargs['num_classes'], 
                        saveto=os.path.join(save_dir, "roc_curves.png") if save_dir is not None else save_dir,
                        label_dict=self.model_kwargs['label_dict'], color_map = self.color_map)
            self.precision_recall(labels, preds, self.model_kwargs['num_classes'], 
                                saveto=os.path.join(save_dir, "pr_curves.png") if save_dir is not None else save_dir,
                                label_dict=self.model_kwargs['label_dict'],color_map = self.color_map)
            self.confusion_matrix(labels, preds, self.model_kwargs['num_classes'], 
                                saveto=os.path.join(save_dir, "confusion_matrix.png") if save_dir is not None else save_dir,
                                label_dict=self.model_kwargs['label_dict'])
            scores = self.classification_metrics(labels, preds, self.model_kwargs['num_classes'], 
                                                saveto=os.path.join(save_dir, "metrics.json") if save_dir is not None else save_dir,
                                                label_dict=self.model_kwargs['label_dict'])
            if save_dir is not None:
                np.savez_compressed(  # MV ADDED: Save labels and preds and ids together as a single .npz file
                    os.path.join(save_dir, "labels_preds.npz"),
                    labels=np.array([labels], dtype=object),
                    preds=np.array([preds], dtype=object),
                    ids=np.array([ids], dtype=object) if ids is not None else np.array([None], dtype=object),
                )
            return scores['overall']
        
        elif self.task_type == 'survival':
            scores = self.survival_metrics(
                labels['survival_event'], 
                labels['survival_time'], 
                preds, 
                saveto = os.path.join(save_dir, "metrics.json") if save_dir is not None else save_dir
            )
            # Optional MV added, functionality not checked yet: also store ids if present
            if ids is not None and save_dir is not None:
                np.savez_compressed(
                    os.path.join(save_dir, "labels_preds.npz"),
                    ids=np.array([ids], dtype=object),
                    survival_event=np.array([labels['survival_event']], dtype=object),
                    survival_time=np.array([labels['survival_time']], dtype=object),
                    risks=np.array([preds], dtype=object),
                )
            return scores

    def _finalize_metrics(self, split, labels_across_folds, preds_across_folds, scores_across_folds):

        if len(labels_across_folds) > 0:
            # Perform bootstrapping and calculate 95% CI (# They do?)
            bootstraps = self.bootstrap(labels_across_folds, preds_across_folds, self.num_bootstraps)
            if self.task_type == 'classification':
                scores_across_folds = [self.classification_metrics(labels, preds, self.model_kwargs['num_classes'])['overall'] for labels, preds in tqdm(bootstraps, desc=f'Computing {self.num_bootstraps} bootstraps')]
            elif self.task_type == 'survival':
                scores_across_folds = [self.survival_metrics(labels['survival_event'], labels['survival_time'], preds) for labels, preds in tqdm(bootstraps, desc=f'Computing {self.num_bootstraps} bootstraps')]
            
            # Save bootstraps
            folder_path = os.path.join(self.results_dir, f"{split}_metrics")
            os.makedirs(folder_path, exist_ok=True)  
            for idx, metrics_dict in enumerate(scores_across_folds):
                folder_path_curr = os.path.join(folder_path, f"bootstrap_{idx}")
                os.makedirs(folder_path_curr, exist_ok=True)  

                file_path = os.path.join(folder_path_curr, "metrics.json")
                with open(file_path, "w") as f:
                    json.dump(metrics_dict, f, indent=4)

            return self.get_95_ci(scores_across_folds)
        else:
            # Report mean ± SE across folds
            return self.get_mean_se(scores_across_folds)

    def _pick_checkpoint(self, checkpoint_dir):

        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
        available_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        
        if not available_checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
        if len(available_checkpoints) > 1:
            latest = max(available_checkpoints, key=lambda x: int(x.replace('.pt','').split('_')[-1]))
            warnings.warn(f"{len(available_checkpoints)} checkpoints found in {checkpoint_dir}. Using the latest checkpoint {latest}.")
            return os.path.join(checkpoint_dir, latest)
        else:
            return os.path.join(checkpoint_dir, available_checkpoints[0])
    
    def _run_single_epoch(self):
        
        # Set models to appropriate mode
        if self.mode == 'train':
            self.model.train()
            context_manager = torch.enable_grad()
            self.optimizer.zero_grad(set_to_none=True) 
        elif self.mode in ['val','test']:
            self.model.eval()
            context_manager = torch.inference_mode()
        else:
            raise ValueError('mode must be either "train", "val", or "test".')

        # Initialize epoch's values
        all_losses = []
        all_info = [] # This is additional info returned by the model along with the loss (e.g. predictions, targets, etc.)
        new_best_loss = False
        new_best_target_score = False
        new_best_smooth_rank = False
        num_samples_processed = 0
        optimizer_skipped = False
        scores = {}
        total_batches = len(self.dataloaders[self.mode])
    
        # Training Loop
        with context_manager:
            for batch_idx, batch in enumerate(self.dataloaders[self.mode]):
                num_samples_processed += len(batch['ids'])
                with torch.autocast(device_type='cuda', dtype=self.precision, enabled=self.precision != torch.float32):
                    loss, info = self.model(batch, output='loss')
                    assert isinstance(loss, torch.Tensor), f"Loss must be a tensor, got {loss} instead"
                    assert isinstance(info, list), f"Info must be a list on CPU, got {info} instead"
                
                # Update trackers
                all_losses.append(loss.cpu().detach().numpy())
                all_info.extend(info)

                if self.mode == 'train':
                    is_last_batch = (batch_idx + 1 == total_batches)
                    scale = self._compute_accumulation_scale(batch_idx) if self.gradient_accumulation > 1 else 1
                    self.grad_scaler.scale(loss / scale).backward()

                    if (batch_idx + 1) % self.gradient_accumulation == 0 or is_last_batch:
                        self.grad_scaler.step(self.optimizer)
                        current_scale = self.grad_scaler.get_scale()
                        self.grad_scaler.update()
                        optimizer_skipped = self.grad_scaler.get_scale() < current_scale
                        self.optimizer.zero_grad(set_to_none=True)

                        if optimizer_skipped:
                            warnings.warn(
                                f"Optimizer step skipped due to problems related with gradient underflow ({self.grad_scaler.get_scale()} < {current_scale}). "
                                f"Fold: {self.current_fold}, Epoch: {self.current_epoch}, "
                                f"Batch idx: {batch_idx}, Global opt step: {self.global_opt_steps}"
                            )
                        
                        # Count optimization step
                        self.global_opt_steps += not optimizer_skipped

                        self._lr_step(
                            update=(
                                self.scheduler_config['step_on'] == 'model_update'
                                and not optimizer_skipped
                            ),
                            metric = None,
                            step = self.global_opt_steps
                        )
                    

                if self.view_progress == 'bar':
                    self.loop.set_postfix(
                        batch=f'{batch_idx + 1}/{total_batches}',
                        samples=num_samples_processed,
                        loss=f'{loss.item():.4f}'
                    )

        
        # Save current epoch metrics
        self.current_epoch_metrics = {
            "loss": all_losses,
            "info": all_info,
            "avg_loss": np.mean(all_losses)
        }

        # Update best val loss
        if self.mode == 'val':
            avg_loss = self.current_epoch_metrics['avg_loss']
            labels, preds, ids = self._accumulate_preds(self.dataloaders[self.mode], self.model)
            
            scores = self._compute_metrics(
                labels = labels,
                preds = preds,
                save_dir=None,
                ids=ids
            )
            self.log_scores(self.current_epoch,scores)
            
            # Finding new best validation loss/target metric
            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                new_best_loss = True

            if self.target_score is not None and self.target_score != "val-loss":
                self.log_target_score(self.current_epoch, scores[self.target_score])
                if self._is_new_best_score(scores.get(self.target_score)):
                    self.best_target_score = scores[self.target_score]
                    new_best_target_score = True

            # Update is scheduler is plateau
            if self.scheduler_config['type'] == 'plateau':
                self._plateau_step(scores)

        # Update best smooth rank
        if isinstance(all_info[0], dict) and 'smooth_rank' in all_info[0].keys():
            smooth_rank = np.mean([info['smooth_rank'] for info in all_info])
            self.current_epoch_metrics['smooth_rank'] = smooth_rank
            if smooth_rank > self.best_smooth_rank:
                self.best_smooth_rank = smooth_rank
                new_best_smooth_rank = True
        else:
            assert self.save_which_checkpoints != 'best-smooth-rank', f"save_which_checkpoints cannot be 'best-smooth-rank' if smooth rank is not returned by the model."

        # Save checkpoints
        save_conditions = [self.save_which_checkpoints == 'all',
                           self.save_which_checkpoints == 'best-val-loss' and new_best_loss,
                           self.save_which_checkpoints == f'best-{self.target_score}' and new_best_target_score,
                           self.save_which_checkpoints == 'best-smooth-rank' and new_best_smooth_rank,
                           self.save_which_checkpoints.startswith('every-') and (self.current_epoch + 1) % int(self.save_which_checkpoints.split('-')[1]) == 0,
                           self.save_which_checkpoints.startswith('last-') and (self.current_epoch + 1) > self.num_epochs - int(self.save_which_checkpoints.split('-')[1])]
        if any(save_conditions):
            self.save_checkpoint(
                self.model, 
                self.save_which_checkpoints, 
                os.path.join(self.results_dir,
                'checkpoints', 
                f'fold_{self.current_fold}',
                f"epoch_{self.current_epoch}.pt")
            )
            
        self.log_loss(self.current_epoch) 
        # self.log_smooth_rank(self.current_epoch)

        return (
            new_best_loss, 
            self.current_epoch_metrics['avg_loss'],
            new_best_target_score,
            scores.get(self.target_score),
        )
    
    def _plateau_step(self, scores: dict):
        step_on = self.scheduler_config.get('step_on')

        if step_on is None:
            raise RuntimeError(
                f"'step_on' cannot be None when using plateau scheduler. Scheduler config: {self.scheduler_config}"
            )

        if step_on == 'val-loss':
            metric = self.current_epoch_metrics.get('avg_loss')
            if metric is None:
                raise RuntimeError(
                    f"'avg_loss' for validation is None. Cannot step scheduler on '{step_on}'. "
                    f"Current metrics: {self.current_epoch_metrics}"
                )

        elif step_on in scores:
            metric = scores[step_on]
            if metric is None:
                raise RuntimeError(
                    f" scores[{step_on}]' is invalid ({metric = }). Available scores: {list(scores.keys())}"
                )

        else:
            parts = step_on.split('-')
            if len(parts) == 2:
                agg, metric_name = parts
                metric_dict = scores.get(metric_name)
                if metric_dict is None:
                    raise RuntimeError(
                        f"Metric '{metric_name}' not found in scores. Available scores: {list(scores.keys())}"
                    )
                if agg not in metric_dict or metric_dict[agg] is None:
                    raise RuntimeError(
                        f"Aggregation '{agg}' not found or None in metric '{metric_name}'. "
                        f"Available keys: {list(metric_dict.keys())}, Scores: {scores}"
                    )
                metric = metric_dict[agg]
            else:
                raise RuntimeError(
                    f"Invalid 'step_on' value '{step_on}' "
                    f"Available solutions: {list(scores.keys()).extend('val')}"
                )

        self._lr_step(update=True, metric=metric,step = self.current_epoch)

    def _lr_step(self,update,metric = None, step  = None):
            if update : 
                self.log_lr(step)

                if metric is not None:
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            return
        
    def _compute_accumulation_scale(self, batch_idx : int):

            # 1. Identify the current accumulation block (e.g., 0-3 : 0, 4-7 : 4 for accumulation_steps = 4)
            current_block_start = (batch_idx // self.gradient_accumulation) * self.gradient_accumulation
            
            # 2. Determine the end of the current block
            # It's either the full accumulation step or the end of the dataset
            total_batches = len(self.dataloaders[self.mode])
            current_block_end = min(total_batches, current_block_start + self.gradient_accumulation)
            
            # 3. The true scale is the actual number of steps in this specific block
            update_window_length = current_block_end - current_block_start

            # Tracks different scales used, main for debugging purposes
            self._monitor_scaling_factors_ga.add(update_window_length)
            
            return update_window_length
    
    def _init_scheduler(self):
        """
        Crea lo scheduler principale e, se configurato, lo combina con un warmup.
        Gestisce automaticamente la sottrazione dei passi/epoch del warmup.
        """
        warmup_cfg = self.scheduler_config.get('warmup', None)
        warmup_scheduler = None
        warmup_steps = 0

        # Se c'è warmup, creo il LinearLR per warmup
        if warmup_cfg:
            warmup_steps = warmup_cfg.get('total_iters', 5)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=warmup_cfg.get('start_factor', 1/3),
                end_factor=warmup_cfg.get('end_factor', 1.0),
                total_iters=warmup_steps
            )

        main_scheduler = self._create_main_scheduler(warmup_steps=warmup_steps)

        # Se c'è warmup, concateno con SequentialLR
        if warmup_scheduler:
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
        else:
            return main_scheduler

    def _create_main_scheduler(self, warmup_steps=0):
        """
        Crea lo scheduler principale adattando eventuali parametri
        in base alla presenza di warmup.
        """
        scheduler_type = self.scheduler_config['type']
        step_on = self.scheduler_config.get('step_on', 'epoch')

        if scheduler_type == 'plateau':
            stepping_policy = self.scheduler_config.get('step_on', 'val')
            if stepping_policy not in ClassificationMixin.SCALAR_SCORES and stepping_policy != "val-loss":
                raise ValueError(f"'{stepping_policy}' is not a valid metric.")
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.scheduler_config['mode'],
                factor=self.scheduler_config['factor'],
                patience=self.scheduler_config['patience'],
                threshold=self.scheduler_config['threshold'],
                min_lr=self.scheduler_config.get('min_lr', 0)
            )

        elif scheduler_type == 'step':
            # milestone rimane invariato, non dipende dal warmup
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.scheduler_config['milestones'],
                gamma=self.scheduler_config['gamma']
            )

        elif scheduler_type == 'cosine':
            # T_max deve togliere i passi di warmup
            total_steps = self.num_epochs if step_on == 'epoch' else self.num_batches_train * self.num_epochs
            adjusted_T_max = max(total_steps - warmup_steps, 1)  # non può essere 0
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=adjusted_T_max,
                eta_min=self.scheduler_config['eta_min']
            )

        elif scheduler_type == 'cosine_warm_restart':
            # T_0 deve essere almeno 1 e considera warmup se necessario
            T_0 = max(self.scheduler_config['T_0'] - warmup_steps, 1)
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=self.scheduler_config['T_mult'],
                eta_min=self.scheduler_config['eta_min']
            )

        elif scheduler_type == 'exp':
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer,
                gamma=self.scheduler_config['gamma']
            )

        elif scheduler_type == 'linear':
            return torch.optim.lr_scheduler.LinearLR(
                optimizer=self.optimizer,
                start_factor=self.scheduler_config['start_factor']
            )

        elif scheduler_type == 'gigapath':
            from patho_bench.optim.GigaPathOptim import CustomLRScheduler
            default_scheduler_args = {
                'base_lr': self.optimizer_config['base_lr'],
                'max_epochs': self.num_epochs,
                'accumulation_steps': self.gradient_accumulation,
                'len_dataloader': len(self.dataloaders['train']),
                'warmup_steps': warmup_steps
            }
            return CustomLRScheduler(
                optimizer=self.optimizer,
                default_scheduler_args=default_scheduler_args,
                custom_scheduler_args=self.scheduler_config
            )

        elif callable(scheduler_type):
            default_scheduler_args = {
                'base_lr': self.optimizer_config['base_lr'],
                'max_epochs': self.num_epochs,
                'accumulation_steps': self.gradient_accumulation,
                'len_dataloader': len(self.dataloaders['train']),
                'warmup_steps': warmup_steps
            }
            return scheduler_type(
                optimizer=self.optimizer,
                default_scheduler_args=default_scheduler_args,
                custom_scheduler_args=self.scheduler_config
            )

        else:
            raise NotImplementedError(f"Scheduler type {scheduler_type} not implemented.")

    def _init_optimizer(self):
        optimizer_type = self.optimizer_config['type']
        extra_kwargs = {k: v for k, v in self.optimizer_config.items() if k not in ['type', 'get_param_groups', 'param_group_args', 'base_lr']}

        if 'get_param_groups' in self.optimizer_config:
            param_groups = self.optimizer_config['get_param_groups'](self.model, **self.optimizer_config['param_group_args'])
            assert isinstance(param_groups, list), "get_param_groups must return a list of dictionaries."
            assert len(param_groups) > 0, "get_param_groups must return a non-empty list of dictionaries."
        else:
            param_groups = self.model.parameters()

        if optimizer_type.lower() == "adam":
            return Adam(param_groups, self.optimizer_config['base_lr'], **extra_kwargs)
        elif optimizer_type.lower() == 'sgd':
            return SGD(param_groups, self.optimizer_config['base_lr'], **extra_kwargs)
        elif optimizer_type.lower() == "adamw":
            return AdamW(param_groups, self.optimizer_config['base_lr'], **extra_kwargs)
        else:
            raise NotImplementedError(f"Optimizer {optimizer_type} not implemented.")
                     
    def _is_new_best_score(self,candidate: float) -> bool:

        if candidate is None:
            return False
        
        lower_is_better = {"val-loss"}

        if self.target_score in lower_is_better:
            return candidate < self.best_target_score

        if self.target_score in ClassificationMixin.SCALAR_SCORES:
            return candidate > self.best_target_score

        # Fallback: unknown metric
        warnings.warn(
            f"Score type '{self.target_score}' not recognised as a valid metric."
            "Assuming higher is better.",
            RuntimeWarning,
        )
        return candidate > self.best_target_score

    def _compute_avg_durations(self):
        train_durations = []
        val_durations = []

        for fold_runs in self.durations.values():
            for run in fold_runs:
                train_durations.append(run["train"])
                val_durations.append(run["val"])

        avg_train = np.mean(train_durations)
        avg_val = np.mean(val_durations)

        return avg_train, avg_val

    def _fmt_duration(self, seconds):
        if seconds is None:
            return "N/A"
        return str(timedelta(seconds=int(seconds)))

    def get_device_info(self):
        """
        Returns a dictionary with detailed device info for the current device.
        """

        info = {}

        if 'cuda' in str(self.device) and torch.cuda.is_available():
            device = torch.device(self.device)
            device_idx = device.index if hasattr(device, 'index') else 0
            props = torch.cuda.get_device_properties(device_idx)

            info['Index'] = device_idx
            info['Name'] = props.name
            info['VRAM (GB)'] = f"{props.total_memory / (1024 ** 3):.2f}"
            info['MultiProcessor Count'] = props.multi_processor_count
        else:
            info['Index'] = "N/A"
            info['Name'] = str(self.device)
            info['VRAM (GB)'] = "N/A"
            info['MultiProcessor Count'] = "N/A"
        return info

    @staticmethod
    def _print_kv(key, value, width_key=40, width_val=70, indent=0):
        """Utility: print key/value with wrapping for long values."""
        indent_str = " " * indent
        value_str = str(value)
        # wrap long lines
        wrapped = textwrap.wrap(value_str, width=width_val)
        if wrapped:
            print(f"{indent_str}{key:<{width_key}}: {wrapped[0]}")
            for line in wrapped[1:]:
                print(f"{indent_str}{'':<{width_key}}  {line}")
        else:
            print(f"{indent_str}{key:<{width_key}}: {value_str}")

    @staticmethod
    def _print_section(title: str, width: int = 120, pad: str = "-"):
        title = f" {title} "
        side = max((width - len(title)) // 2, 0)
        line = pad * side + title + pad * (width - side - len(title))
        print(line)

    def experiment_report(self, validation_summary: dict = None):
        now = datetime.now()
        width_total = self.WIDTH_EXP

        print("\n" + "=" * width_total)
        title = "EXPERIMENT OVERVIEW" if validation_summary is None else "FINAL SUMMARY"
        print(f"{title:^{width_total}}")

        date_time_dir = f"Date: {now.strftime('%Y-%m-%d %H:%M:%S')} | Results Directory: {self.results_dir}"
        print(f"{date_time_dir:^{width_total}}")

        print("=" * width_total)

        if validation_summary is None:
            self._print_section("GENERAL INFO", width_total)
            info_lines = [
                ("Task Type", self.task_type),
                ("Number of Folds", self.dataset.num_folds),
                ("Epochs", self.num_epochs),
                ("(Phisical) Batch Size", self.device_batch_size),
                ("Gradient Accumulation Steps", self.gradient_accumulation),
                ("Batch Size", self._batch_size),
                ("Precision", self.precision),
                ("Seed", self.seed),
                ("Force determinism in cuda", self.disable_cudnn),
                ("Early Stop Enabled", self.early_stop),
            ]
            if self.early_stop:
                info_lines += [
                    ("Early Stop Policy", self.early_stop_policy),
                    ("Early Stop Patience", self.patience),
                ]

            for k, v in info_lines:
                if isinstance(v, type) or str(v).startswith("torch."):
                    continue
                self._print_kv(k, v)

            self._print_section("MODEL INFO", width_total)
            for k, v in self.model_kwargs.items():
                if isinstance(v, torch.nn.Module):
                    v = f"See model in {os.path.join(self.results_dir, 'model.txt')}"
                self._print_kv(k, v)

            self._print_section("OPTIMIZER INFO", width_total)
            for k, v in self.optimizer_config.items():
                self._print_kv(k, v)

            self._print_section("SCHEDULER INFO", width_total)
            for k, v in self.scheduler_config.items():
                self._print_kv(k, v)

            self._print_section("DEVICE INFO", width_total)
            for k, v in self.get_device_info().items():
                self._print_kv(k, v)

        else:
            total_train = sum(d["train"] for dl in self.durations.values() for d in dl)
            total_val = sum(d["val"] for dl in self.durations.values() for d in dl)
            total_overall = total_train + total_val

            self._print_section("EXPERIMENT DURATION (HH:MM:SS)", width_total)
            print(f"Train:      {self._fmt_duration(total_train)}")
            print(f"Validation: {self._fmt_duration(total_val)}")
            print(f"Overall:    {self._fmt_duration(total_overall)}")
            print("-" * width_total)
            self._print_section(f"VALIDATION SUMMARY (Bootstraps {self.num_bootstraps})", width_total)
            headers = ["Metric", "Mean ± Std"]
            rows = [[metric, vals["formatted"]] for metric, vals in validation_summary.items()]
            col_widths = [50, 65]
            self._print_table(headers, rows, col_widths, width=width_total)
            print("=" * width_total + "\n\n")

    def training_report(self, validation_scores: dict = None):
        num_folds = self.dataset.num_folds
        fold = self.current_fold + 1
        num_samples_train = len(self.dataloaders["train"].dataset) if self.dataloaders.get("train") else 0
        num_samples_val = len(self.dataloaders["val"].dataset) if self.dataloaders.get("val") else 0
        now = datetime.now()
        width_total = self.WIDTH_FOLD

        if validation_scores is None:
            # ===== START OF FOLD =====
            self.fold_start_time = now

            print("\n" + "#" * width_total)
            print(f"{f'FOLD {fold}/{num_folds}':^{width_total}}")
            s = f"Date: {self.fold_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            print(f"{s:^{width_total}}")
            print("#" * width_total)

            self._print_section("DATA", width_total)

            headers = ["Info", "Train", "Val"]
            rows = [
                ["WSIs", num_samples_train, num_samples_val],
                ["Physical Batches", self.num_phisical_batches_train, self.num_phisical_batches_val],
                ["Batches", self.num_batches_train, self.num_batches_val]
            ]
            col_widths = [20, 8, 8]

            self._print_table(headers, rows, col_widths, width=width_total)

        else:
            # ===== END OF FOLD =====
            self.fold_end_time = now

            total_train_time = sum(d["train"] for d in self.durations[self.current_fold])
            total_val_time = sum(d["val"] for d in self.durations[self.current_fold])
            total_time = total_train_time + total_val_time
            avg_train, avg_val = self._compute_avg_durations()

            print("#" * width_total)
            print(f"{f'TRAINING SUMMARY: FOLD {fold}':^{width_total}}")
            s = f"Date: {self.fold_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            print(f"{s:^{width_total}}")
            print("#" * width_total)

            self._print_section("DURATION", width_total)
            # Tabella con le prime due righe
            headers = ["Value", "Train | Valid"]
            rows = [
                ["Total ", f"{self._fmt_duration(total_train_time)} | {self._fmt_duration(total_val_time)}"],
                ["Average ", f"{self._fmt_duration(avg_train)} | {self._fmt_duration(avg_val)}"]
            ]
            col_widths = [20, 20]  # larghezza colonne, adatta al width_total

            self._print_table(headers, rows, col_widths, width=width_total)

            # Righe singole, fuori tabella
            self._print_kv(
                "Total duration (train + valid): ",
                self._fmt_duration(total_time)
            )
            self._print_section("EPOCH SUMMARY", width_total)
            self._print_kv(
                "Training Epochs",
                f"{self.fold_training_epochs} / {self.num_epochs}"
            )
            self._print_kv(
                "Optimization Steps",
                f"{self.global_opt_steps} / {self.num_batches_train * self.fold_training_epochs}"
            )
            self._print_kv(
                "Accumulation Scaling Employed",
                self._monitor_scaling_factors_ga
            )

            score = validation_scores.get(self.target_score)
            if self.target_score and score:
                self._print_kv(
                    f"({self.target_score})",
                    f"{score:.2f}"
                )

            print("#" * width_total)

    @staticmethod
    def _print_table(headers, rows, col_widths, width=120):
        """
        Print a table fully framed, centered in the given width.
        Uses: | for vertical borders, _ for top/bottom, - for inner separators.
        """
        # calcola larghezza tabella corretta
        table_width = sum(col_widths) + len(col_widths) + 1  # +1 per il primo | e +1 per ogni separatore

        # indent per centrare
        indent = max((width - table_width) // 2, 0)
        pad = " " * indent

        # funzione per costruire una riga
        def _row(items):
            cells = [f"{str(item):^{w}}" for item, w in zip(items, col_widths)]
            return pad + "|" + "|".join(cells) + "|"

        # linee superiori / inferiori
        hline = pad + "-" * table_width
        sep_line = pad + "-" * table_width

        # stampa tabella
        print(hline)
        print(_row(headers))
        print(sep_line)
        for r in rows:
            print(_row(r))
        print(hline)

    def _print_training_header(self):
        """Stampa l’intestazione della tabella training, centrata."""
        # intestazione sezione
        self._print_section("TRAINING", width=self.WIDTH_FOLD)

        # definizione larghezze colonne
        col_widths = [self.EPOCH_W, self.PHASE_W, self.DURATION_W]
        headers = ["EPOCH", "PHASE", "DURATION (HH:MM:SS)"]

        # larghezza totale tabella (colonne + pipe + spazi)
        table_width = sum(col_widths) + len(col_widths) + 1
        indent = max((self.WIDTH_FOLD - table_width) // 2, 0)
        pad = " " * indent

        # linee superiori e separatori
        hline = pad + "-" * table_width
        sep_line = pad + "-" * table_width

        # stampa header
        print(hline)
        header_row = pad + "|" + "|".join(f"{h:^{w}}" for h, w in zip(headers, col_widths)) + "|"
        print(header_row)
        print(sep_line)

    def _print_training_row(self, epoch, phase, duration_seconds):
        """Stampa una riga della tabella training, centrata."""
        col_widths = [self.EPOCH_W, self.PHASE_W, self.DURATION_W]
        table_width = sum(col_widths) + len(col_widths) + 1
        indent = max((self.WIDTH_FOLD - table_width) // 2, 0)
        pad = " " * indent

        duration_str = str(timedelta(seconds=int(duration_seconds)))
        row_items = [epoch, phase, duration_str]
        row_str = pad + "|" + "|".join(f"{str(i):^{w}}" for i, w in zip(row_items, col_widths)) + "|"
        print(row_str)