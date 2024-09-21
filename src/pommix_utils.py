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


def pna(feat):
    """
    Input tensor of shape (n, m, d), where n is the number of samples, m is the number of features, and d is the dimesnion of features
    will produce a tensor of shape (n, 4*d) where each feature is replaced by its mean, variance, max, and min
    Collapse along the second dimension.
    Padding is designated -999
    """
    if torch.is_tensor(feat):
        feat[feat == -999] = torch.nan      # remove any padding
        new_feat = torch.zeros((len(feat), feat.shape[-1]*4))
        for i, x in enumerate(feat):
            x = x[~torch.isnan(x).any(dim=1)]
            var = torch.zeros(x.shape[1]) if len(x)==1 else x.var(0)
            new_feat[i,:] = torch.cat([x.mean(0), var, x.max(0)[0], x.min(0)[0]])
        return new_feat
    else:
        feat[feat == -999] = np.nan         # remove any padding
        new_feat = np.zeros((len(feat), feat.shape[-1]*4))
        for i, x in enumerate(feat):
            x = x[~np.isnan(x).any(axis=1)]
            var = np.zeros(x.shape[1]) if len(x) == 1 else x.var(axis=0)
            new_feat[i, :] = np.concatenate([x.mean(axis=0), var, x.max(axis=0), x.min(axis=0)])
        return new_feat


def cast_float(x):
    return x if isinstance(x, float) else x.item()

def bootstrap_ci(true_values, predictions, metric_fn, num_samples=1000, alpha=0.05):
    """
    Calculates a bootstrap confidence interval for a given metric.

    Args:
        true_values: True values of the target variable.
        predictions: Predicted values.
        metric: A function that takes true_values and predictions as input and returns a scalar metric.
        num_samples: Number of bootstrap samples to generate.
        alpha: Significance level for the confidence interval.

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
    """

    n = len(true_values)
    values = []
    for _ in range(num_samples):
        indices = np.random.randint(0, n, n)
        bootstrap_true = true_values[indices]
        bootstrap_pred = predictions[indices]
        value = metric_fn(bootstrap_true, bootstrap_pred)
        values.append(cast_float(value))
    lower_bound = np.percentile(values, alpha / 2 * 100)
    upper_bound = np.percentile(values, (1 - alpha / 2) * 100)

    return lower_bound, upper_bound, values


