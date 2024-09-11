import torch
from torch.utils.data import Dataset

from typing import List, Iterable, Union, Optional, Callable

import numpy as np
import rdkit.Chem.AllChem as Chem
import torch_geometric as pyg


class GraphDataset(Dataset):
    """
    Wrapper for PyTorch dataset for a list of graph objects
    """
    def __init__(self, x: List[pyg.data.Data],
                 y: Iterable[Union[int, float]]):
        self.x = x
        self.y = y
        self.node_dim = x[0].x.shape[-1]
        self.edge_dim = x[0].edge_attr.shape[-1]
        self.global_dim = x[0].u.shape[-1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], torch.tensor(self.y[idx], dtype=torch.float32)
    

