import copy
from torch import nn
import torch
from patho_bench.optim.NLLSurvLoss import NLLSurvLoss
from patho_bench.Pooler import Pooler

"""
This is a wrapper class that allows for finetuning of a slide encoder model on a single multiple-instance learning (MIL) classification task.
This class is used by ExperimentFactory.
"""

class DefaultTrainableSlideEncoder(nn.Module):
    def __init__(self,
                 slide_encoder,
                 post_pooling_dim,
                 task_name,
                 num_classes,
                 loss,
                 device,
                 label_dict,
                 **kwargs):
        '''
        Initializes a trainable classifier using a preloaded slide encoder.

        Args:
            slide_encoder (nn.Module, optional): The image pooling module used to process input features. Input shape: batch_size x num_patches x feature_dim. Output shape: batch_size x post_pooling_dim.
            post_pooling_dim (int, optional): Dimension of features after pooling.
            task_name (str): Name of the task.
            num_classes (int): The number of classes.
            loss (nn.Module or dict): Loss function to use for training. If a dictionary is provided, it should map task names to loss functions.
            device (str or torch.device, optional): Device on which to run the model.
        '''
        super().__init__()
        self.slide_encoder = copy.deepcopy(slide_encoder)
        self.post_pooling_dim = post_pooling_dim
        self.task_name = task_name
        self.num_classes = num_classes
        self.loss = loss
        self.device = device
        self.label_dict = label_dict

        # Create classification head
        # Input shape: batch_size x feature_dim
        # Output shape: batch_size x num_classes
        self.classification_head = nn.Linear(self.post_pooling_dim, self.num_classes)

        # Move to device
        self.to(device)
        if isinstance(self.loss, dict): # If balanced loss is used
            for iter_idx, loss in self.loss.items():
                self.loss[iter_idx].to(device)
        else:
            self.loss.to(device)

    def forward(self, batch, output = 'loss'):
        '''        
        Args:
            batch (dict): Input batch containing 'slide' and 'labels' keys.
            output (str): 'loss', 'features', or 'logits'
        Returns:
            Logits (shape: batch_size x n_categories) if return_loss is False, otherwise loss and accuracy
        '''
        # Slide encoding
        slide_encoder_input = Pooler.prepare_slide_encoder_input_batch(batch['slide'])
        slide_features = Pooler.pool(self.slide_encoder, slide_encoder_input, self.device)
        
        if output == 'features':
            return slide_features
        
        # Classification heads
        logits = self.classification_head(slide_features)
        if output == 'logits':
            return logits
        
        # Compute survival loss
        if isinstance(self.loss, NLLSurvLoss):
            # Note that survival task labels follow a particular format. If the expected format is unclear from the examples, please raise a GitHub issue.
            y_bins = batch['labels'][self.task_name].to(self.device)
            y_bins = y_bins % 4 # Convert y_bins from 8 to 4 bins (survival quartiles)
            y_event = batch['labels']['extra_attrs'][f'{self.task_name}_event'].to(self.device)
            loss = self.loss(logits, y_bins.unsqueeze(0), y_event.unsqueeze(0))
        # Compute balanced loss
        elif isinstance(self.loss, dict):
            assert batch.get('current_iter') is not None, "Current iter must be provided for weighted loss, but got None from batch. Please check the dataloader."
            loss = self.loss[batch['current_iter']](logits.squeeze(), batch['labels'][self.task_name].to(self.device).squeeze())
        # Compute standard loss
        else:
            loss = self.loss(logits.squeeze(), batch['labels'][self.task_name].to(self.device).squeeze())

        if output == 'loss':
            info = [{}]
            return loss, info
        
        raise ValueError(f"Invalid output type {output}")
    
class im4MECTrainableSlideClassifier(nn.Module):
    """
    Classificatore per slide multibranch (CLAM-style) compatibile con TrainableSlideEncoder.
    Output: B x n_classes
    """

    def __init__(
            self, 
            slide_encoder, 
            post_pooling_dim, 
            task_name,
            num_classes,
            loss, 
            device, 
            label_dict,
            **kwargs
        ):
        super().__init__()
        self.slide_encoder = copy.deepcopy(slide_encoder)
        self.post_pooling_dim = post_pooling_dim
        self.task_name = task_name
        self.num_classes = num_classes
        self.loss = loss
        self.device = device
        self.label_dict = label_dict

        # Classifier indipendente per ciascuna classe
        self.classifiers = nn.ModuleList([nn.Linear(post_pooling_dim,1) for _ in range(num_classes)])

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()

        self.to(device)
        if isinstance(self.loss, dict): # If balanced loss is used
            for iter_idx, loss in self.loss.items():
                self.loss[iter_idx].to(device)
        else:
            self.loss.to(device)

    def forward(self, batch, output='loss'):
        """
        Args:
            batch (dict): contiene almeno 'features' e 'labels'
            output (str): 'loss', 'logits', 'features'
        Returns:
            loss, info se output='loss'
        """
        # --- Slide encoding (dal nostro pooler multibranch) ---
        slide_encoder_input = Pooler.prepare_slide_encoder_input_batch(batch['slide'])
        slide_features = Pooler.pool(self.slide_encoder, slide_encoder_input, self.device)

        B, n_branches, hidden_dim = slide_features.shape
        assert n_branches == self.num_classes, f"{n_branches = }, {self.num_classes = }"
        assert hidden_dim == self.post_pooling_dim, f"{B = }, {n_branches = }, {hidden_dim = }"

        # --- Logits branch-wise ---
        logits = torch.empty(B, self.num_classes, device=self.device)
        for c in range(self.num_classes):
            logits[:, c] = self.classifiers[c](slide_features[:, c, :]).squeeze(-1)

        if output == 'loss':
            labels = batch['labels'][self.task_name].to(self.device)
            if isinstance(self.loss, dict):
                # balanced loss: richiede current_iter nel batch
                assert batch.get('current_iter') is not None, "current_iter deve essere presente per loss bilanciata"
                loss_val = self.loss[batch['current_iter']](logits.squeeze(), labels.squeeze())
            else:
                loss_val = self.loss(logits.squeeze(), labels.squeeze())
            info = [{}]  # eventuali metriche opzionali
            return loss_val, info

        elif output == 'logits':
            return logits

        elif output == 'features':
            return slide_features

        else:
            raise NotImplementedError(f"Output mode {output} non implementato")

class ABMIL_MILLAB_TrainableSlideClassifier(nn.Module):
    """
    Wrapper per modelli MIL-Lab (es. ABMIL) compatibile con FinetuningExperiment.
    Supporta batch di training e batch pathobench.
    """
    
    def _unfold_pathobench_batch(self, patho_bench_input_batch):
        return {
            'features': patho_bench_input_batch['features'],
            'coords': patho_bench_input_batch.get('coords', None),
            'mask': patho_bench_input_batch.get('mask', None),
            'attributes': patho_bench_input_batch.get('attributes', None)
        }
    
    def _prepare_batch(self, batch: dict):
        """
        Decide il tipo di batch e ritorna:
        - features (tensor)
        - attn_mask (tensor o None)
        - labels (tensor o None)
        - current_iter (int o None)
        """
        # training through pathobench
        if "slide" in batch:
            slide_input = self._unfold_pathobench_batch(batch["slide"])
            features = slide_input["features"].to(self.device)
            attn_mask = slide_input.get("mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.device, dtype=int)
            labels = batch["labels"][self.task_name].to(self.device)
            current_iter = batch.get("current_iter", None)
        else:  # inference
            features = batch["features"].to(self.device)
            attn_mask = batch.get("mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.device, dtype=int)
            labels = None
            current_iter = None

        return features, attn_mask, labels, current_iter

    def __init__(
        self,
        slide_classifier: nn.Module,
        task_name: str,
        num_classes: int,
        loss,
        label_dict: dict,
        device,
        dropout: float,
        freeze_backbone: bool = False,
        debug: bool = False,
        post_pooling_dim: int = -1,
        **kwargs
    ):
        super().__init__()
        self.slide_classifier = copy.deepcopy(slide_classifier)
        set_dropout(self.slide_classifier, dropout)

        self.task_name = task_name
        self.num_classes = num_classes
        self.loss = loss
        self.label_dict = label_dict
        self.device = device
        self.freeze_backbone = freeze_backbone
        self.debug = debug
        self.post_pooling_dim = post_pooling_dim

        if self.freeze_backbone:
            for p in self.slide_classifier.parameters():
                p.requires_grad = False
            if hasattr(self.slide_classifier, "model") and hasattr(self.slide_classifier.model, "classifier"):
                for p in self.slide_classifier.model.classifier.parameters():
                    p.requires_grad = True
            else:
                raise AttributeError("slide_classifier non espone model.classifier")

        self.to(device)

        if isinstance(self.loss, dict):
            for _, l in self.loss.items():
                l.to(device)
        elif self.loss is not None:
            self.loss.to(device)

    def forward(self, batch: dict, output: str = "loss"):
        features, attn_mask, labels, current_iter = self._prepare_batch(batch)
        loss_val = None

        # --- Forward MIL-Lab ---
        results_dict, log_dict = self.slide_classifier(
            features,
            attn_mask=attn_mask,
            return_attention=True if self.debug else (output == "features"),
            return_slide_feats=True if self.debug else (output == "features"),
        )

        logits = results_dict.get("logits", None)
        slide_feats = log_dict.get("slide_feats", None)
        attention = log_dict.get("attention", None)

        # --- Loss computation ---
        if output == "loss" and labels is not None:
            if isinstance(self.loss, dict):
                assert current_iter is not None, "current_iter richiesto per loss bilanciata"
                loss_fn = self.loss[current_iter]
                loss_val = loss_fn(logits, labels)
            else:
                loss_val = self.loss(logits, labels)

        info = [{
            "slide_features": slide_feats,
            "attention": attention,
            "logits": logits,
            "loss": loss_val
        }]

        return loss_val, info

class CLAMTrainableSlideClassifier(nn.Module):
    """
    Wrapper per CLAMSB compatibile con FinetuningExperiment.
    Gestisce batch come dizionario, linear probing e vari output types.
    """

    def __init__(
        self,
        slide_classifier: nn.Module,
        post_pooling_dim: int,
        task_name: str,
        num_classes: int,
        loss,
        label_dict: dict,
        device,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.slide_classifier = copy.deepcopy(slide_classifier)
        self.post_pooling_dim = post_pooling_dim
        self.task_name = task_name
        self.num_classes = num_classes
        self.loss = loss
        self.label_dict = label_dict
        self.device = device
        self.freeze_backbone = freeze_backbone

        # --- Linear probing: freeze backbone se richiesto ---
        if self.freeze_backbone:
            for p in self.slide_classifier.parameters():
                p.requires_grad = False

            # Riabilita il classifier bag-level
            if hasattr(self.slide_classifier.model, "classifier"):
                for p in self.slide_classifier.model.classifier.parameters():
                    p.requires_grad = True

        # --- Move to device ---
        self.to(device)

        # --- Move loss to device ---
        if isinstance(self.loss, dict):
            for _, l in self.loss.items():
                l.to(device)
        else:
            self.loss.to(device)

    def forward(self, X, output: str = "loss"):
        """
        Args:
            batch (dict): deve contenere 'slide' con features, mask, coords, attributes.
            output (str): 'loss' | 'logits' | 'features'

        Returns:
            logits se output='logits', oppure (loss, info) se output='loss',
            oppure (features, info) se output='features'
        """

        # --- Prepare input per CLAM ---
        slide_input = Pooler.prepare_slide_encoder_input_batch(X["slide"])
        h = slide_input["features"].to(self.device)
        attn_mask = slide_input.get("mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        # --- Forward pass ---
        label = X["labels"][self.task_name].to(self.device) if "labels" in X else None
        results_dict, log_dict = self.slide_classifier(
            h=h,
            label=label,
            loss_fn=self.loss if output=="loss" else None,
            attn_mask=attn_mask,
            return_attention=True,
            return_slide_feats=True
        )

        # --- Gestione output ---
        if output == "logits":
            return results_dict["logits"]

        elif output == "loss":
            return results_dict["loss"], [log_dict]

        elif output == "features":
            info = {
                "attention": log_dict.get("attention"),
                "slide_feats": log_dict.get("slide_feats")
            }
            return log_dict.get("slide_feats"), [info]

        else:
            raise ValueError(f"Invalid output type: {output}")

def set_dropout(model: nn.Module, p: float):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p