import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-2])
sys.path.append(str(base_dir / "src/"))

import seaborn as sns
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# import umap
from umap import UMAP

from pommix_utils import set_visualization_style, get_embeddings_from_smiles
from dataloader.representations.graph_utils import EDGE_DIM, NODE_DIM

if __name__ == '__main__':
    set_visualization_style()

    np.random.seed(0)

    df = pd.read_csv(base_dir / "figures" / "attention_interpretability" / "min_max_keys_across_rows.csv")
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
    new_df.to_csv('smiles_interactions.csv', index=False)


    # smi = new_df['smiles']

    # # create POM embeddings
    # MODEL_PATH = Path(
    #     base_dir / "scripts/pom/gs-lf_models/pretrained_pom/"
    # )  # where the pretrained model is saved
    # emb = get_embeddings_from_smiles(smi, file_path=MODEL_PATH).numpy()

    emb = torch.cat([max_key_embs, min_key_embs]).numpy()

    umap = UMAP(random_state=0)
    reduced_emb = umap.fit_transform(emb)

    new_df['UMAP 1'] = reduced_emb[:,0]
    new_df['UMAP 2'] = reduced_emb[:,1]

    green_color = sns.color_palette("Dark2")[4]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_to_green", [(1, 1, 1), green_color]
    )

    df_ni = new_df[new_df['strong_interaction'] == 0]
    plt.scatter(df_ni['UMAP 1'], df_ni['UMAP 2'], c=df_ni['attn_value'], cmap=cmap, alpha=1, marker='P', label='Low', edgecolors='black')
    plt.clim(0,1)
    
    df_i = new_df[new_df['strong_interaction'] == 1]
    plt.scatter(df_i['UMAP 1'], df_i['UMAP 2'], c=df_i['attn_value'], cmap=cmap, alpha=1, marker='o', label='High', edgecolors='black')
    plt.legend(title='Interaction')
    plt.clim(0,1)

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.colorbar()
    plt.savefig('embedding_space_attention.png', bbox_inches='tight')