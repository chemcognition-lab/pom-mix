import sys
sys.path.append('..')  # required to load dreamloader utilities

import numpy as np
import torch
import pickle
import os
from dataloader import DatasetLoader, SplitLoader
from scripts_pom.make_embeddings import get_embeddings_from_smiles
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from dataloader.representations import graph_utils
from pom.data import GraphDataset
from pom.gnn.graphnets import GraphNets
from torch_geometric.loader import DataLoader as pygdl

from typing import List, Optional

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

if __name__ == "__main__":
    TRAIN_DATAPATH = "/u/rajao/pom-mix/datasets/mixtures/pickled_dataset/random_split/train"
    TEST_DATAPATH = "/u/rajao/pom-mix/datasets/mixtures/pickled_dataset/random_split/test"
    POM_FILEPATH = "/u/rajao/pom-mix/scripts_pom/gs-lf_models/pretrained_pom"
    UNK_TOKEN = -999

    # Training set
    dl = DatasetLoader()
    dl.load_dataset('mixtures')
    dl.featurize('smiles')

    df = pd.read_csv("/u/rajao/pom-mix/datasets/mixtures/mixtures_combined.csv")

    train, test = train_test_split(df, test_size=0.2, random_state=0, 
                                stratify=df["Dataset"])

    train_indices = train.index.tolist()
    test_indices = test.index.tolist()

    X = dl.features
    y = dl.labels

    max_pad_len_X1 = max(len(arr) for arr in X[:, 0])
    max_pad_len_X2 = max(len(arr) for arr in X[:, 1])

    max_pad_len = max(max_pad_len_X1, max_pad_len_X2)

    print(X.shape)
    print(y.shape)

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_1_train = []
    for mix in X_train[:, 0]:

        emb = get_embeddings_from_smiles(mix, file_path=POM_FILEPATH)
        mix_size = emb.shape[0]

        mix_mat = np.pad(emb, ((0, max_pad_len - mix_size), (0, 0)))

        X_1_train.append(mix_mat)
    
    X_1_train = torch.Tensor(np.array(X_1_train)).to(torch.float32)

    X_2_train = []
    for mix in X_train[:, 1]:

        emb = get_embeddings_from_smiles(mix, file_path=POM_FILEPATH)
        mix_size = emb.shape[0]

        mix_mat = np.pad(emb, ((0, max_pad_len - mix_size), (0, 0)))

        X_2_train.append(mix_mat)
    
    X_2_train = torch.Tensor(np.array(X_2_train)).to(torch.float32)

    y_train = torch.Tensor(y_train).flatten().to(torch.float32)

    print(X_1_train.shape)
    print(X_2_train.shape)
    print(y_train.shape)

    with open(os.path.join(TRAIN_DATAPATH, "x1.pkl"), "wb") as f:
        pickle.dump(X_1_train, f)
    with open(os.path.join(TRAIN_DATAPATH, "x2.pkl"), "wb") as f:
        pickle.dump(X_2_train, f)
    with open(os.path.join(TRAIN_DATAPATH, "y.pkl"), "wb") as f:
        pickle.dump(y_train, f)

    # Same stuff but for test
    X_test = X[test_indices]
    y_test = y[test_indices]

    X_1_test = []
    for mix in X_test[:, 0]:

        emb = get_embeddings_from_smiles(mix, file_path=POM_FILEPATH)
        mix_size = emb.shape[0]

        mix_mat = np.pad(emb, ((0, max_pad_len - mix_size), (0, 0)))

        X_1_test.append(mix_mat)
    
    X_1_test = torch.Tensor(np.array(X_1_test)).to(torch.float32)

    X_2_test = []
    for mix in X_test[:, 1]:

        emb = get_embeddings_from_smiles(mix, file_path=POM_FILEPATH)
        mix_size = emb.shape[0]

        mix_mat = np.pad(emb, ((0, max_pad_len - mix_size), (0, 0)))

        X_2_test.append(mix_mat)
    
    X_2_test = torch.Tensor(np.array(X_2_test)).to(torch.float32)

    y_test = torch.Tensor(y_test).flatten().to(torch.float32)

    print(X_1_test.shape)
    print(X_2_test.shape)
    print(y_test.shape)

    with open(os.path.join(TEST_DATAPATH, "x1.pkl"), "wb") as f:
        pickle.dump(X_1_test, f)
    with open(os.path.join(TEST_DATAPATH, "x2.pkl"), "wb") as f:
        pickle.dump(X_2_test, f)
    with open(os.path.join(TEST_DATAPATH, "y.pkl"), "wb") as f:
        pickle.dump(y_test, f)