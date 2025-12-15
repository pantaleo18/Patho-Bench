from patho_bench.datasets.BaseDataset import BaseDataset
import torch
import os
from IPython.display import display

"""
PatchEmbeddingsDataset loads patch-level features for a sample.
A sample may be a slide or a collection of slides (e.g. a patient).
"""

class PatchEmbeddingsDataset(BaseDataset):
    def __init__(self,
                 split,
                 load_from,
                 preprocessor = None,
                 bag_size = None,
                 pad = False,
                 shuffle = False,
                 combine_slides_per_patient = True):
        '''
        Loads patch-level embeddings for a sample.
        A sample may be a slide or a collection of slides (e.g. a patient).
        
        Args:
            split (BaseSplit): Split object
            load_from (str or list): Path to directory containing h5 files or list of paths to h5 files
            preprocessor (dict): Dict of preprocessor callables to apply to each asset
            bag_size (int): Number of features to randomly sample from each sample. None to use all features.
            pad (bool): Whether to pad bags to the same size
            shuffle (bool): Whether to shuffle the dataset (only used if bag_size is None)
            combine_slides_per_patient (bool): Whether to combine patches from multiple slides when sampling bag. If False, will sample from each slide independently and return a list of feature tensors for each sample.
        '''
        
        super().__init__(split)
        self.load_from = load_from
        self.preprocessor = preprocessor
        self.bag_size = bag_size
        self.pad = pad
        self.shuffle = shuffle
        self.combine_slides_per_patient = combine_slides_per_patient

        if isinstance(self.load_from, str):
            self.load_from = [self.load_from]

        self.available_slide_paths = {}
        for path in self.load_from:
            if not os.path.exists(path):
                print(f"WARNING: Dataset source path {path} does not exist. Skipping.")
                continue
            for file in os.listdir(path):
                if file.endswith('.h5'):
                    slide_id = os.path.splitext(file)[0]
                    self.available_slide_paths[slide_id] = os.path.join(path, file)
                    
    def _apply_preprocessor(self, assets):
        '''
        Apply preprocessor functions to each item in the provided asset.
        
        Args:
          assets (dict): Dictionary of assets to preprocess. Each key should correspond to a key in the self.preprocessor dict.
        '''
        if self.preprocessor:
            for key, preprocessor in self.preprocessor.items():
                if preprocessor is not None:
                    assets[key] = preprocessor(assets[key])
        return assets
                    
    def _collate_slides(self, assets, method):
        '''
        Collates list of dicts into a dict of collated slide assets.
        
        Args:
          assets (list[dict]): List of assets to concatenate. Each asset should be a dictionary with keys corresponding to ['features', 'coords'].
          method (str): Method to use for collation. Options are 'concat' or 'list'.
        
        Returns:
          collated_assets (dict): Dictionary of collated assets.
        '''
        collated_assets = {}
        # Cicla su tutte le chiavi dei singoli asset, così non perdi la mask
        for asset_key in assets[0].keys():
            if method == 'concat' and asset_key != 'mask':
                collated_assets[asset_key] = torch.cat([asset[asset_key] for asset in assets], axis=0)
            elif method == 'list':
                collated_assets[asset_key] = [asset[asset_key] for asset in assets]
        return collated_assets
    
    def _sample_dict_of_lists(self, assets):
        '''
        Sample from a dictionary of lists using the same indices for each list.
        
        Args:
        - assets (dict): Dictionary of assets to sample from.
        
        Returns:
        - final_assets (dict): Dictionary of sampled assets.
        '''
        
        # Sample bag of assets
        sampled_assets = {}
        sample_indices = None
        for key, val in assets.items():
            # The first loop sets sample_indices to a random list of indices, which is applied to all subsequent keys
            sampled_assets[key], mask, sample_indices = self._sample(val, self.bag_size, self.pad, sample_indices)
        sampled_assets['mask'] = mask
        
        return sampled_assets
    
    def get_dataloader(self, current_iter, fold, batch_size=None, num_workers=16):
        subset_dataset = self.get_subset(current_iter, fold)
        if subset_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=len(subset_dataset) if batch_size is None else batch_size,
            sampler=subset_dataset.get_datasampler('random'),
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=subset_dataset.collate_fn
        )