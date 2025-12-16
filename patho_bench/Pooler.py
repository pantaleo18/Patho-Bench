import os
import h5py
import torch
from tqdm import tqdm

class Pooler:
    """
    Pool patch features using a pretrained slide encoder and save pooled features to disk.
    Assumes batches come direttamente dal DataLoader, già collati e pronti.
    """

    def __init__(self, patch_embeddings_dataset, model_name, model_kwargs, save_path, device):
        self.dataset = patch_embeddings_dataset
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.device = device
        self.model = None

    def _load_model(self):
        """Load frozen slide encoder."""
        from trident.slide_encoder_models.load import encoder_factory
        self.model = encoder_factory(self.model_name, freeze=True, **self.model_kwargs)
        self.model.eval()
        self.model.to(self.device)

        # Save model info
        with open(os.path.join(self.save_path, '_model.txt'), 'w') as f:
            f.write(repr(self.model))
            f.write(f'\nTotal parameters: {sum(p.numel() for p in self.model.parameters())}')
            f.write(f'\nTrainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

    @torch.inference_mode()
    def run(self):
        """Loop through the dataset and pool features."""
        if self.model is None:
            self._load_model()

        loop = tqdm(self.dataset.ids, desc="Pooling features")
        for sample_id in loop:

            save_file = os.path.join(self.save_path, f"{sample_id}.h5")
            if os.path.exists(save_file):
                loop.set_postfix_str(f"{sample_id} already pooled, skipping")
                continue

            # Load batch
            batch = self.dataset[sample_id]
            if batch['id'] is None:
                continue

            try:
                cleaned_batch = self.prepare_slide_encoder_input_batch(batch)

                # Forward pass
                pooled_features = self.pool(self.model, cleaned_batch, self.device)

                # Save
                with h5py.File(save_file, 'w') as f:
                    f.create_dataset('features', data=pooled_features.float().cpu().numpy())

            except Exception as e:
                print(f"\033[31mError processing {sample_id}: {e}\033[0m")

    @staticmethod
    def prepare_slide_encoder_input_batch(batch):
        """
        Keep batch ready for ABMIL / Trident encoders.
        Assumes batch is already collated: features shape (B, N, D), mask (B, N)
        """
        return {
            'features': batch['features'],
            'coords': batch['coords'],
            'mask': batch['mask'],
            'attributes': batch['attributes']
        }

    @staticmethod
    def pool(model, cleaned_batch, device):
        """
        Forward pass for ABMIL / slide encoders.
        Supports full batch in parallel.
        """
        return model.forward(cleaned_batch, device=device)
