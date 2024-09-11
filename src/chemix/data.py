import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import pickle

import numpy as np
from typing import Callable, List, Union

class DreamDataset(Dataset):
    def __init__(self, inputs, labels):
        super().__init__()
        self.inputs = inputs
        self.labels = labels

        self.representation_dim = self.inputs.shape[2]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        labels = self.labels[idx]
        return input_data, labels


def load_pickled_dataset(data_path, batch_size, num_workers, shuffle=False):
    with open(os.path.join(data_path, "x1.pkl"), "rb") as f:
        X_1 = pickle.load(f)
    with open(os.path.join(data_path, "x2.pkl"), "rb") as f:
        X_2 = pickle.load(f)
    with open(os.path.join(data_path, "y.pkl"), "rb") as f:
        y = pickle.load(f)

    X = torch.stack((torch.Tensor(X_1), torch.Tensor(X_2)), dim=-1)
    y = torch.Tensor(y)

    dataset = DreamDataset(inputs=X, labels=y)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return dataset, dataloader

def map_nested_list(nested_list, mapping_dict):
    def map_element(x):
        return mapping_dict.get(x, x)
    
    def recursive_map(item):
        if isinstance(item, list):
            return [recursive_map(sub_item) for sub_item in item]
        else:
            return map_element(item)
    
    return recursive_map(nested_list)

def get_mixture_smiles(
        mixtures: np.ndarray, 
        from_smiles: Callable
    ) -> Union[List[Data], np.ndarray]:
    # this function will return the graphs for all smiles
    # present in a mixture, returning them as a list
    flat_mix = mixtures.flatten()
    smiles_list = list(set([x for xs in flat_mix for x in xs]))
    smiles_list.sort()

    pad_len = max([len(x) for x in flat_mix])
    mixtures_processed = []
    for mix in mixtures.transpose():
        smiles_processed = []
        for smi_arr in mix:
            smiles_processed.append(np.pad(smi_arr, (0, pad_len - len(smi_arr)), constant_values=''))
        mixtures_processed.append(smiles_processed)
    mixtures_processed = np.array(mixtures_processed).transpose((1,2,0)).tolist()

    # map it to the feature
    feature_map = {smi: i-1 for i, smi in enumerate([''] + smiles_list)}
    mix_inds = map_nested_list(mixtures_processed, feature_map)

    return [from_smiles(smi) for smi in smiles_list], np.array(mix_inds) 
    
