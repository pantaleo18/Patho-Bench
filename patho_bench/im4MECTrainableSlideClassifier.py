import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from patho_bench.Pooler import Pooler

class im4MECTrainableSlideClassifier(nn.Module):
    """
    Classificatore per slide multibranch (CLAM-style) compatibile con TrainableSlideEncoder.
    Output: B x n_classes
    """

    def __init__(self, slide_encoder, post_pooling_dim, task_name, num_classes, loss, device, label_dict):
        """
        Args:
            slide_encoder (nn.Module): il pooler multibranch
            post_pooling_dim (int): hidden_dim dei branch
            task_name (str): nome del task
            num_classes (int): numero di classi target
            loss (nn.Module o dict): CrossEntropyLoss (bilanciata o no)
            device (str o torch.device)
            label_dict (dict): opzionale, come in TrainableSlideEncoder
        """
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
