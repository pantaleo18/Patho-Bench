from patho_bench.datasets.BaseDataset import BaseDataset
import torch
import os
from IPython.display import display
from patho_bench.debug.TimingTracker import TimingTracker

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
    
    def __getitem__(self, idx):
        # -------------------------
        # ID + slide lookup
        # -------------------------
        sample_id = self.ids[idx] if isinstance(idx, int) else idx
        slide_ids = self.data[sample_id]['slide_id']

        if len(slide_ids) != 1:
            raise RuntimeError(
                f"Expected exactly 1 slide per sample, got {len(slide_ids)} "
                f"for sample {sample_id}. PORCODIO NOOOOO!"
            )

        slide_id = slide_ids[0]

        if slide_id not in self.available_slide_paths:
            raise ValueError(f"Slide {slide_id} not found in {self.load_from}")

        # -------------------------
        # H5 loading
        # -------------------------
        data, attrs = self.load_h5(
            self.available_slide_paths[slide_id],
            keys=['features', 'coords']
        )

        attributes = {
            'patch_size_level0': (
                attrs['coords'].get('patch_size_level0', None)
                if 'coords' in attrs else None
            )
        }

        # -------------------------
        # NumPy → Torch
        # -------------------------
        asset = {k: torch.from_numpy(v) for k, v in data.items()}

        # -------------------------
        # Preprocessing
        # -------------------------
        asset = self._apply_preprocessor(asset)

        # -------------------------
        # Sampling (bag_size, shuffle)
        # -------------------------
        if self.bag_size is not None or self.shuffle:
            asset = self._sample_dict_of_lists(asset)

        # -------------------------
        # Metadata
        # -------------------------
        asset.update({
            'id': sample_id,
            'paths': [self.available_slide_paths[slide_id]],
            'attributes': attributes
        })

        return asset

    ##########DEBUG########################à
    # def __getitem__(self, idx):
    #     TT = TimingTracker()

    #     t_total = TT.tic("getitem_total")

    #     # -------------------------
    #     # ID + slide lookup
    #     # -------------------------
    #     t = TT.tic("id_resolution")
    #     sample_id = self.ids[idx] if isinstance(idx, int) else idx
    #     slide_ids = self.data[sample_id]['slide_id']
    #     TT.toc(t)

    #     if len(slide_ids) == 0:
    #         display(self.data)
    #         raise ValueError(
    #             f'No slides found for case ID {sample_id} '
    #             f'in split with {len(self.data)} samples'
    #         )

    #     # -------------------------
    #     # H5 loading
    #     # -------------------------
    #     t = TT.tic("h5_load_total")
    #     assets = []
    #     attributes = []

    #     for slide_id in slide_ids:
    #         if slide_id not in self.available_slide_paths:
    #             raise ValueError(
    #                 f"Slide {slide_id} not found in {self.load_from}."
    #             )

    #         t_slide = TT.tic("h5_load_single")
    #         data, attrs = self.load_h5(
    #             self.available_slide_paths[slide_id],
    #             keys=['features', 'coords']
    #         )
    #         TT.toc(t_slide)

    #         assets.append(data)
    #         attributes.append({
    #             'patch_size_level0': (
    #                 attrs['coords']['patch_size_level0']
    #                 if 'patch_size_level0' in attrs['coords']
    #                 else None
    #             )
    #         })

    #     TT.toc(t)

    #     # -------------------------
    #     # Attribute consistency
    #     # -------------------------
    #     t = TT.tic("attribute_check")
    #     if self.combine_slides_per_patient:
    #         assert all(
    #             attr == attributes[0] for attr in attributes
    #         ), (
    #             f'Tried to combine slides for each patient, but attributes '
    #             f'of slides {slide_ids} do not match: {attributes}'
    #         )
    #         attributes = attributes[0]
    #     TT.toc(t)

    #     # -------------------------
    #     # NumPy → Torch
    #     # -------------------------
    #     t = TT.tic("numpy_to_tensor")
    #     assets = [
    #         {key: torch.from_numpy(val) for key, val in asset.items()}
    #         for asset in assets
    #     ]
    #     TT.toc(t)

    #     # -------------------------
    #     # Preprocessing
    #     # -------------------------
    #     t = TT.tic("preprocessing")
    #     assets = [self._apply_preprocessor(asset) for asset in assets]
    #     TT.toc(t)

    #     # -------------------------
    #     # Collation + sampling
    #     # -------------------------
    #     if self.combine_slides_per_patient:
    #         t = TT.tic("slide_collate_concat")
    #         assets = self._collate_slides(assets, method='concat')
    #         TT.toc(t)

    #         if self.bag_size is not None or self.shuffle:
    #             t = TT.tic("sampling_concat")
    #             assets = self._sample_dict_of_lists(assets)
    #             TT.toc(t)
    #     else:
    #         t = TT.tic("sampling_per_slide")
    #         for i, slide_assets in enumerate(assets):
    #             if self.bag_size is not None or self.shuffle:
    #                 assets[i] = self._sample_dict_of_lists(slide_assets)
    #         TT.toc(t)

    #         t = TT.tic("slide_collate_list")
    #         assets = self._collate_slides(assets, method='list')
    #         TT.toc(t)

    #     # -------------------------
    #     # Metadata
    #     # -------------------------
    #     t = TT.tic("metadata_attach")
    #     assets.update({
    #         'id': sample_id,
    #         'paths': [
    #             self.available_slide_paths[slide_id]
    #             for slide_id in slide_ids
    #         ],
    #         'attributes': attributes
    #     })
    #     TT.toc(t)

    #     TT.toc(t_total)
    #     TT.report("__gettitem__")
    #     return assets
