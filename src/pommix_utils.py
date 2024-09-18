from typing import Optional, Iterable

from dataloader.representations import graph_utils
from pom.gnn import GraphNets
from pom.data import GraphDataset

from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch_geometric.loader import DataLoader as pygdl


def get_embeddings_from_smiles(smi_list: list[str], file_path: str, gnn: Optional[torch.nn.Module] = None):
    # generate a matrix of embeddings
    # Size: [N, embedding_size]
    # enter a model if you want to load a different model, otherwise defaulting
    graphs = [graph_utils.from_smiles(smi, init_globals=True) for smi in smi_list]
    ds = GraphDataset(graphs, np.zeros((len(graphs), 1)))
    loader = pygdl(ds, batch_size=64, shuffle=False)

    if gnn is None:
        gnn = GraphNets.from_json(node_dim=ds.node_dim, edge_dim=ds.edge_dim, json_path=f'{file_path}/hparams.json')
        state_dict = torch.load(f'{file_path}/gnn_embedder.pt', map_location=torch.device('cpu'))
        gnn.load_state_dict(state_dict)

    embeddings = []
    with torch.no_grad():
        gnn.eval()
        for batch in loader:
            data, _ = batch
            embed = gnn(data)
            embeddings.append(embed)
        embeddings = torch.concat(embeddings, dim=0)

    return embeddings


def create_tvt_split_indices(
        n: int, 
        stratify: Optional[Iterable] = None, 
        train_size: float = 0.7, 
        valid_size: float = 0.1, 
        test_size: float = 0.2,
        seed: int = 0
    ):
    indices = list(range(n))

    if stratify is None:
        stratify = np.ones((n, 1))

    # split trainval/test (test)
    train_ind, train_strat, test_ind, _ = train_test_split(indices, stratify, test_size = test_size, stratify = stratify, random_state = seed)

    # now split val from trainval
    train_ind, val_ind = train_test_split(train_ind, train_size = train_size / (train_size + valid_size), stratify = train_strat, random_state = seed)  # split out the val size

    return train_ind, val_ind, test_ind


def augment_mixture_pairs(features, labels):
    """
    Augments the given mixture pairs by creating a new feature list and concatenating it with the original features, but permuted.
    Assumes that the last dimension is the dimension of mixtures pairs
    
    Args:
        features (ndarray): The original feature list.
        labels (ndarray): The original label list.
        
    Returns:
        tuple: A tuple containing the augmented features and labels.
    """
    feature_list_augment = np.array([np.stack([x[..., 1], x[..., 0]], axis=-1) for x in features])
    features = np.vstack((features, feature_list_augment))
    labels = np.concatenate([labels, labels])
    return features, labels


