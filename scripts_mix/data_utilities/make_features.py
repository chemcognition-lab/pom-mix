from typing import Tuple, Optional

import os, sys
import random
import seaborn as sns
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

from pathlib import Path
script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-2])
sys.path.append( str(base_dir / 'src/') )

import rdkit.Chem as Chem

from pommix_utils import get_embeddings_from_smiles, pna
from dataloader import DatasetLoader
from dataloader.representations.features import rdkit2d_normalized_features


if __name__ == '__main__':
    DATASET_DIR = Path(base_dir / 'datasets/mixtures')

    df = pd.read_csv(DATASET_DIR / 'mixture_smi_definitions_clean.csv')
    df = df.set_index(['Dataset', 'Mixture Label'])


    # create PNA rdkit features
    features = []
    for i, row in df.iterrows():
        row = row.dropna()
        feat = pna(np.expand_dims(np.array(rdkit2d_normalized_features(row.tolist())), 0))
        features.append(feat)
    features = np.concatenate(features)

    # Create a new dataframe with the same indices as the original df, but only with the pna features
    feature_columns = [f'{i}' for i in range(features.shape[1])]
    features_df = pd.DataFrame(features, columns=feature_columns, index=df.index).reset_index()
    features_df.to_csv(DATASET_DIR / 'mixture_rdkit_definitions_clean.csv', index=False)


    # create POM embeddings
    MODEL_PATH = Path(base_dir / 'scripts_pom/gs-lf_models/pretrained_pom/')        # where the pretrained model is saved
    max_pad_len = df.columns.size        # mixtures are padded
    features = []
    for i, row in df.iterrows():
        row = row.dropna()
        emb = get_embeddings_from_smiles(row.tolist(), file_path=MODEL_PATH)
        mix_mat = np.pad(emb, ((0, max_pad_len - emb.shape[0]), (0, 0)), constant_values=-999)
        features.append(mix_mat)
    features = np.stack(features, axis=0)

    # note that the ordering of the pom embeddings are preserved
    # so that we can look up the corresponding dataset/mixture label 
    # based on the mixture_smi_definitions_clean.csv file
    np.savez(DATASET_DIR / 'mixture_pom_embeddings.npz', features=features)



