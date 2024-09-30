import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-1])
sys.path.append( str(base_dir / 'src/') )

import numpy as np
import seaborn as sns
import pandas as pd

from dataloader import DatasetLoader
from dataloader.representations.graph_utils import EDGE_DIM, NODE_DIM, from_smiles
from pom.gnn.graphnets import GraphNets
from chemix import build_chemix, get_mixture_smiles
from pommix_utils import pna

import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchmetrics
import json
from ml_collections import ConfigDict
from torch_geometric.data import Batch

from umap import UMAP
from sklearn.decomposition import PCA

# from scipy.spatial.distance import cosine
from scipy.stats import linregress
from sklearn.metrics.pairwise import cosine_distances as cosine_measure
from scipy.stats import pearsonr

DATASET_DIR = base_dir / 'datasets'

from pommix_utils import set_visualization_style
set_visualization_style()

if __name__ == '__main__':

    # get the mixtures and their embeddings
    dl = DatasetLoader()
    dl.load_dataset('mixtures')
    dl.featurize('mix_smiles')

    # keep only snitz
    indices = np.where(np.char.find(dl.dataset_id.astype(str), 'Snitz 1') != -1)[0]
    dl.features = dl.features[indices]
    dl.labels = dl.labels[indices]

    geom_mean = np.array([ np.sqrt(len(mixes[0]) * len(mixes[1])) for mixes in dl.features])
    # geom_mean = np.round(geom_mean * 2) / 2

    # load chemix model and get embedding=
    pommix_path = base_dir / "scripts_pommix/results/pretrained_model/model"
    hp_gnn = ConfigDict(json.load(open(pommix_path / 'hparams_graphnets.json', 'r')))
    embedder = GraphNets(node_dim=NODE_DIM, edge_dim=EDGE_DIM, **hp_gnn)
    embedder.load_state_dict(torch.load(pommix_path / 'gnn_embedder.pt'))
    embedder.eval()
    hp_mix = ConfigDict(json.load(open(pommix_path / f'hparams_chemix.json', 'r')))
    chemix = build_chemix(config=hp_mix.chemix)
    chemix.load_state_dict(torch.load(pommix_path / 'chemix.pt'))
    chemix.eval()

    # the smiles
    graph_list, indices = get_mixture_smiles(dl.features, from_smiles)
    train_gr = Batch.from_data_list(graph_list)
    out = embedder.graphs_to_mixtures(train_gr, indices)
    pommix_embeds = chemix.embed(out).detach().numpy().squeeze()

    dist = np.diagonal(cosine_measure(pommix_embeds[...,0], pommix_embeds[...,1]))
    
    # remove single mixtures
    geom_mean = geom_mean[dist != 0 ]
    dist = dl.labels.squeeze()[dist != 0 ]

    m, b, rho, p, _ = linregress(geom_mean, dist)
    x_fit = np.linspace(0, 43)
    y_fit = m*x_fit + b

    df = pd.DataFrame({r"$\sqrt{n_1 \cdot n_2}$": geom_mean, "Cosine distance": dist})
    lp = sns.lineplot(data=df, x=r"$\sqrt{n_1 \cdot n_2}$", y="Cosine distance", linestyle='', err_style='bars', marker='o')
    ax = lp.axes
    ax.plot(x_fit, y_fit)
    ax.text(0.65, 0.85, f'Pearson $\\rho$: {rho:.2f}',
            transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))      
    plt.savefig("pommix_white_noise_snitz_exp.png", bbox_inches="tight")
    # plt.savefig("pommix_white_noise.svg", bbox_inches="tight")

