import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-2])
sys.path.append(str(base_dir / "src/"))

import seaborn as sns
import torch
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from rdkit import Chem
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
# import umap
from umap import UMAP

from pommix_utils import set_visualization_style, get_embeddings_from_smiles
from dataloader.representations.graph_utils import EDGE_DIM, NODE_DIM


def cluster_labels_by_jaccard(
    gslf,
    smiles_list,
):
    # Jaccard distance of labels
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    descriptors = [gslf["descriptors"].loc[gslf["IsomericSMILES"] == smiles].tolist()[0].replace(";", "\n") if smiles in list(gslf["IsomericSMILES"]) else "none" for smiles in smiles_list]
    labels = [set(i.split("\n")) for i in descriptors]

    # Create a matrix of Jaccard distances
    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(labels)
    n = len(labels)
    jaccard_matrix = np.zeros((n, n))

    # Compute pairwise Jaccard distance
    for i in range(n):
        for j in range(n):
            jaccard_matrix[i, j] = 1 - jaccard_score(binary_labels[i], binary_labels[j])

    # Perform hierarchical clustering on the Jaccard distance matrix
    # Use the 'ward' method (which is often used for minimizing variance within clusters)
    Z = linkage(squareform(jaccard_matrix), method='ward')

    # Group items by clusters
    cluster_labels = fcluster(Z, t=1.75, criterion='distance')

    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    return clusters

if __name__ == '__main__':
    set_visualization_style()

    np.random.seed(0)

    df = pd.read_csv(base_dir / "figures" / "attention_interpretability" / "min_max_keys_across_rows.csv")
    gslf = pd.read_csv(base_dir / "datasets/gs-lf/gs-lf_combined.csv")

    final_input_emb = torch.load(base_dir / "figures/attention_interpretability/final_input_emb.pt", map_location=torch.device('cpu'))

    # Only select query molecules which are considered as "interacting" with key molecules
    df = df.loc[df["max_weight_val"] > 0.5]

    # Key molecules only associated to maximizing attention weight
    smiles_list_max = list(set(df["max_key_mol"].value_counts().keys()) - set(df["min_key_mol"].value_counts().keys()))

    # Key molecules only associated to minimizing attention weight
    smiles_list_min = list(set(df["min_key_mol"].value_counts().keys()) - set(df["max_key_mol"].value_counts().keys()))

    # Averaging attention weight values for each key molecule
    max_key_avg_attn = []
    for smi in smiles_list_max:
        interactions = df.loc[df["max_key_mol"] == smi]
        avg_attn_weight = interactions["max_weight_val"].mean()
        max_key_avg_attn.append(avg_attn_weight)

    max_key_embs = []

    for smi in smiles_list_max:
        interactions = df.loc[df["max_key_mol"] == smi]

        all_emb = []
        for i, row in interactions.iterrows():
            emb = final_input_emb[row["matrix_index"], row["max_key_idx"], :, row["mixture_num"]]
            all_emb.append(emb)
        
        x = torch.stack(all_emb)
        all_equal = all(np.allclose(x[0], x[i], atol=1e-5) for i in range(1, x.shape[0]))
        assert all_equal

        max_key_embs.append(x[0])

    max_key_embs = torch.stack(max_key_embs)

    # Averaging attention weight values for each key molecule
    min_key_avg_attn = []
    for smi in smiles_list_min:
        interactions = df.loc[df["min_key_mol"] == smi]
        avg_attn_weight = interactions["min_weight_val"].mean()
        min_key_avg_attn.append(avg_attn_weight)

    min_key_embs = []

    for smi in smiles_list_min:
        interactions = df.loc[df["min_key_mol"] == smi]

        all_emb = []
        for i, row in interactions.iterrows():
            emb = final_input_emb[row["matrix_index"], row["min_key_idx"], :, row["mixture_num"]]
            all_emb.append(emb)
        
        x = torch.stack(all_emb)
        all_equal = all(np.allclose(x[0], x[i], atol=1e-5) for i in range(1, x.shape[0]))
        assert all_equal

        min_key_embs.append(x[0])

    min_key_embs = torch.stack(min_key_embs)

    new_df_max = pd.DataFrame()
    new_df_max["smiles"] = smiles_list_max
    new_df_max["strong_interaction"] = [1 for i in range(len(smiles_list_max))]
    new_df_max["attn_value"] = max_key_avg_attn

    new_df_min = pd.DataFrame()
    new_df_min["smiles"] = smiles_list_min
    new_df_min["strong_interaction"] = [0 for i in range(len(smiles_list_min))]
    new_df_min["attn_value"] = min_key_avg_attn

    new_df = pd.concat([new_df_max, new_df_min])

    emb = torch.cat([max_key_embs, min_key_embs])

    # Jaccard distance label clustering
    all_smiles = list(new_df["smiles"])
    clusters = cluster_labels_by_jaccard(gslf, all_smiles)

    for cluster_id, cluster_points in clusters.items():
        print(f"Cluster {cluster_id}: {cluster_points}")

    cluster_idx = []
    for idx, smi in enumerate(all_smiles):
        for key, val in clusters.items():
            if idx in val:
                cluster_idx.append(key - 1)
    
    new_df["cluster_id"] = cluster_idx

    new_df.to_csv('smiles_interactions_cc.csv', index=False)

    umap = UMAP(random_state=0)
    reduced_emb = umap.fit_transform(emb)

    new_df['UMAP 1'] = reduced_emb[:,0]
    new_df['UMAP 2'] = reduced_emb[:,1]

    N = len(new_df["cluster_id"].unique())
    N = N - 1

    cmap = plt.get_cmap('tab10', N)

    df_ni = new_df[new_df['strong_interaction'] == 0]
    df_ni = df_ni[df_ni['cluster_id'] != 0]
    plt.scatter(df_ni['UMAP 1'], df_ni['UMAP 2'], c=df_ni['cluster_id'], cmap=cmap, alpha=0.9, marker='P', label='Low', edgecolors='black')
    plt.clim(1,N)
    
    df_i = new_df[new_df['strong_interaction'] == 1]
    df_i = df_i[df_i['cluster_id'] != 0]
    plt.scatter(df_i['UMAP 1'], df_i['UMAP 2'], c=df_i['cluster_id'], cmap=cmap, alpha=0.9, marker='o', label='High', edgecolors='black')
    plt.clim(1,N)

    plt.legend(title='Interaction')

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    cbar = plt.colorbar()
    tick_locs = (np.arange(1,N+1) + 0.5)*N/(N+1)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(np.arange(1, N+1))

    plt.savefig('embedding_space_new.svg', bbox_inches='tight')