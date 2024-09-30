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

from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

DATASET_DIR = base_dir / 'datasets'

from pommix_utils import set_visualization_style
set_visualization_style()

if __name__ == '__main__':

    # get the mixtures and their embeddings
    df = pd.read_csv(DATASET_DIR / 'mixtures/mixture_smi_definitions_clean.csv')
    dataset_id = df['Dataset'].values

    # pom embeddings 
    pom_embeds = np.load(DATASET_DIR / "mixtures/mixture_pom_embeddings.npz")['features']
    pna_pom_embeds = pna(pom_embeds.copy())

    # load chemix model and get embedding
    chemix_path = base_dir / "scripts_chemix/results/random_train_val/model_augmented"
    hp_mix = ConfigDict(json.load(open(chemix_path / f'hparams_chemix.json', 'r')))
    chemix = build_chemix(config=hp_mix.chemix)
    chemix.load_state_dict(torch.load(chemix_path / 'random_train_val_chemix.pt'))
    chemix.eval()
    chemix_embeds = chemix.embed(torch.from_numpy(pom_embeds).float().unsqueeze(-1)).detach().numpy().squeeze()

    # load chemix model and get embedding=
    pommix_path = base_dir / "scripts_pommix/results/random_train_val/model_augmented"
    hp_gnn = ConfigDict(json.load(open(pommix_path / 'hparams_graphnets.json', 'r')))
    embedder = GraphNets(node_dim=NODE_DIM, edge_dim=EDGE_DIM, **hp_gnn)
    embedder.load_state_dict(torch.load(pommix_path / 'random_train_val_gnn_embedder.pt'))
    embedder.eval()
    hp_mix = ConfigDict(json.load(open(pommix_path / f'hparams_chemix.json', 'r')))
    chemix = build_chemix(config=hp_mix.chemix)
    chemix.load_state_dict(torch.load(pommix_path / 'random_train_val_chemix.pt'))
    chemix.eval()

    # the smiles
    mixtures = np.array([mix[mix != ''] for mix in  df.fillna('')[[f'smi_{i}' for i in range(len(df.columns)-2)]].values], dtype=object).reshape(-1, 1)
    graph_list, indices = get_mixture_smiles(mixtures, from_smiles)
    train_gr = Batch.from_data_list(graph_list)
    out = embedder.graphs_to_mixtures(train_gr, indices)
    pommix_embeds = chemix.embed(out).detach().numpy().squeeze()

    df['pom_embed'] = list(pna_pom_embeds)
    df['chemix_embed'] = list(chemix_embeds)
    df['pommix_embed'] = list(pommix_embeds)
    label_df = pd.read_csv(DATASET_DIR / 'mixtures/mixtures_combined.csv')

    # def get_embedding(df, dataset, mixture_id, embedding_type):
    #     return df[(df['Dataset'] == dataset) & (df['Mixture Label'] == mixture_id)][embedding_type].values[0]
    
    # def embedding_cosine_distance(df, dataset, mixture_1, mixture_2, embedding_type):
    #     return cosine(get_embedding(df, dataset, mixture_1, embedding_type), get_embedding(df, dataset, mixture_2, embedding_type))

    # for i, row in label_df.iterrows():
    #     label_df.loc[i, 'POM Cosine Distance'] = embedding_cosine_distance(df, row['Dataset'], row['Mixture 1'], row['Mixture 2'], 'pom_embed')
    #     label_df.loc[i, 'CheMix Cosine Distance'] = embedding_cosine_distance(df, row['Dataset'], row['Mixture 1'], row['Mixture 2'], 'chemix_embed')
    #     label_df.loc[i, 'POMMix Cosine Distance'] = embedding_cosine_distance(df, row['Dataset'], row['Mixture 1'], row['Mixture 2'], 'pommix_embed')


    ##### PLOT INDIVIDUAL EMBEDDINGS #####
    for feat_name, embed in zip(['CheMix augmented', 'POMMix augmented'], [chemix_embeds, pommix_embeds]):
        df_embed = pd.DataFrame({'Dataset': dataset_id.tolist()})

        # UMAP
        reducer = UMAP(metric='cosine', n_neighbors=100, min_dist=1.0)
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
            sns.scatterplot(df_embed, x=names[0], y=names[1],  hue='Dataset', alpha=0.7, ax=ax)
        plt.savefig(f'embedding_space_{feat_name}.png', bbox_inches='tight')
        plt.savefig(f'embedding_space_{feat_name}.svg', bbox_inches='tight', format='svg')
        
    import pdb; pdb.set_trace()

    # plot the cosine distance
    label_df['Dataset'] = label_df['Dataset'].str.replace(' 1', '').str.replace(' 2', '').str.replace(' 3', '').str.replace(' 4', '')
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), dpi=180, sharey=True)
    for i, (ax, feat_name) in enumerate(zip(axes.flatten(), ['POM', 'CheMix', 'POMMix'])):
        
        if i == len(axes)-1:
            sns.scatterplot(label_df, x=f'{feat_name} Cosine Distance', y='Experimental Values', hue='Dataset', ax=ax)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        else:
            sns.scatterplot(label_df, x=f'{feat_name} Cosine Distance', y='Experimental Values', hue='Dataset', ax=ax, legend=False)
        
        r, _ = pearsonr(label_df[f'{feat_name} Cosine Distance'], label_df['Experimental Values'])
        snitz_r, _ = pearsonr(label_df[label_df['Dataset'] == 'Snitz'][f'{feat_name} Cosine Distance'], label_df[label_df['Dataset'] == 'Snitz']['Experimental Values'])
        bushdid_r, _ = pearsonr(label_df[label_df['Dataset'] == 'Bushdid'][f'{feat_name} Cosine Distance'], label_df[label_df['Dataset'] == 'Bushdid']['Experimental Values'])
        ravia_r, _ = pearsonr(label_df[label_df['Dataset'] == 'Ravia'][f'{feat_name} Cosine Distance'], label_df[label_df['Dataset'] == 'Ravia']['Experimental Values'])
        ax.text(0.7, 0.05, f'Overall $\\rho$: {r:.2f}\n' 
                            f'Snitz $\\rho$: {snitz_r:.2f}\n'
                            f'Bushdid $\\rho$: {bushdid_r:.2f}\n'
                            f'Ravia $\\rho$: {ravia_r:.2f}', transform=ax.transAxes, fontsize=14,
                            verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))           
        ax.set_ylim([0,1])   
        ax.set_xlim([0,2])     
        # ax.set_title(feat_name)
        ax.set_xlabel(f'{feat_name} Cosine Distance')
        ax.set_ylabel('Perceptual Similarity')

    plt.savefig(f'embedding_cosine_distance.png', bbox_inches='tight')
    plt.savefig(f'embedding_cosine_distance.svg', bbox_inches='tight', format='svg')
