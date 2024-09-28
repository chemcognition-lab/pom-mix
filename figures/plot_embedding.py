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

sns.set_context('talk', font_scale=1.3)
DATASET_DIR = base_dir / 'datasets'

if __name__ == '__main__':

    # get the mixtures and their embeddings
    df = pd.read_csv(DATASET_DIR / 'mixtures/mixture_smi_definitions_clean.csv')
    dataset_id = df['Dataset'].values

    # pom embeddings 
    pom_embeds = np.load(DATASET_DIR / "mixtures/mixture_pom_embeddings.npz")['features']
    pna_pom_embeds = pna(pom_embeds.copy())

    # load chemix model and get embedding
    chemix_path = base_dir / "scripts_chemix/results/pretrained_model"
    hp_mix = ConfigDict(json.load(open(chemix_path / f'hparams_chemix.json', 'r')))
    chemix = build_chemix(config=hp_mix.chemix)
    chemix.load_state_dict(torch.load(chemix_path / 'best_model_dict.pt'))
    chemix.eval()
    chemix_embeds = chemix.embed(torch.from_numpy(pom_embeds).float().unsqueeze(-1)).detach().numpy().squeeze()

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

    # the smiels
    mixtures = np.array([mix[mix != ''] for mix in  df.fillna('')[[f'smi_{i}' for i in range(len(df.columns)-2)]].values], dtype=object).reshape(-1, 1)
    graph_list, indices = get_mixture_smiles(mixtures, from_smiles)
    train_gr = Batch.from_data_list(graph_list)
    out = embedder.graphs_to_mixtures(train_gr, indices)
    pommix_embeds = chemix.embed(out).detach().numpy().squeeze()


    for feat_name, embed in zip(['POM', 'Chemix', 'POM-Mix'], [pna_pom_embeds, chemix_embeds, pommix_embeds]):
        df_embed = pd.DataFrame({'Dataset': dataset_id.tolist()})

        # UMAP
        reducer = UMAP(metric='cosine')
        red_dim = reducer.fit_transform(embed)
        df_embed['UMAP 1'] = list(red_dim[:,0])
        df_embed['UMAP 2'] = list(red_dim[:,1])

        # PCA
        reducer = PCA()
        red_dim = reducer.fit_transform(embed)
        df_embed[f'PCA 1 ({reducer.explained_variance_ratio_[0]*100:.2f}%)'] = list(red_dim[:,0])
        df_embed[f'PCA 2 ({reducer.explained_variance_ratio_[1]*100:.2f}%)'] = list(red_dim[:,1])

        df_embed['Dataset'] = df_embed['Dataset'].str.replace(' 1', '').str.replace(' 2', '').str.replace(' 3', '').str.replace(' 4', '')


        fig, axes = plt.subplots(2, 1, figsize=(13, 26))
        for ax, decomp in zip(axes.flatten(), ['UMAP', 'PCA']):
            names = [n for n in df_embed.columns if decomp in n]
            sns.scatterplot(df_embed, x=names[0], y=names[1],  hue='Dataset', palette=sns.color_palette('colorblind'),
                            alpha=0.7, ax=ax)
        plt.savefig(f'embedding_space_{feat_name}.png', bbox_inches='tight')
        plt.savefig(f'embedding_space_{feat_name}.svg', bbox_inches='tight', format='svg')
        import pdb; pdb.set_trace()

