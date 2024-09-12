import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-1])
sys.path.append( str(base_dir / 'src/') )
import seaborn

from typing import List, Optional

from dataloader import DreamLoader, SplitLoader

from dataloader.representations import graph_utils
from pom.data import GraphDataset
from pom.gnn.graphnets import GraphNets

import numpy as np
import torch
from torch_geometric.loader import DataLoader as pygdl


def get_embeddings_from_smiles(smi_list: List[str], file_path: str, gnn: Optional[torch.nn.Module] = None):
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


if __name__ == '__main__':

    dl = DreamLoader()
    dl.load_benchmark('mixtures')
    dl.featurize('smiles')

    sl = SplitLoader()
    for id, training, validation, testing in sl.load_splits(dl.features, dl.labels):
        train_features, train_labels = training
        val_features, val_labels = validation
        test_features, test_labels = testing
        import pdb; pdb.set_trace()
