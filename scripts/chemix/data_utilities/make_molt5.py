import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-3]) # this is for training
sys.path.append(str(base_dir / "src/"))

import numpy as np
import json

from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer

import pandas as pd


transformer = PretrainedHFTransformer(kind='MolT5', notation='smiles', dtype=float)

def make_embedding(arr):
    n = len(arr)
    valids = arr.dropna().tolist()
    embeds = transformer(valids)
    embeds = np.pad(embeds, pad_width=((0, n-len(valids)), (0, 0)), constant_values=-999)
    return embeds

if __name__ == '__main__':    
    df = pd.read_csv(base_dir / 'datasets/mixtures/mixture_smi_definitions_clean.csv')
    smiles = df.drop(['Dataset', 'Mixture Label', 'length', 'Duplicate','Duplicate Of'], axis=1)

    embeddings = []
    for i, arr in smiles.iterrows():
        embeddings.append(make_embedding(arr))

    embeddings = np.stack(embeddings, axis=0)
    np.savez('mixture_molt5_embeddings.npz', features=embeddings)
    # import pdb; pdb.set_trace()

