import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-3])
sys.path.append(str(base_dir / "src/"))

import seaborn
import pandas as pd
import numpy as np
from pommix_utils import get_embeddings_from_smiles, pna
from dataloader.representations.features import rdkit2d_normalized_features


if __name__ == "__main__":
    DATASET_DIR = Path(base_dir / "datasets/mixtures")

    df = pd.read_csv(DATASET_DIR / "mixture_smi_definitions_clean.csv")
    df = df.set_index(["Dataset", "Mixture Label"])
    df = df.drop(['length','Duplicate','Duplicate Of'], axis=1)

    # create PNA rdkit features
    features, pna_features, mean_features = [], [], []
    for i, row in df.iterrows():
        n = len(row)
        row = row.dropna()
        rdkit_feat = np.expand_dims(
            np.array(rdkit2d_normalized_features(row.tolist())), 0
        )
        features.append(np.pad(rdkit_feat, pad_width=((0,0), (0, n-len(row)), (0,0)), constant_values=-999))
        pna_features.append(pna(rdkit_feat))
        mean_features.append(rdkit_feat.mean(1))
    features = np.concatenate(features)
    pna_features = np.concatenate(pna_features)
    mean_features = np.concatenate(mean_features)

    # note that the ordering of the rdkit features are preserved
    # so that we can look up the corresponding dataset/mixture label
    # based on the mixture_smi_definitions_clean.csv file
    np.savez(DATASET_DIR / "mixture_rdkit2d.npz", features=features)

    # Create a new dataframe with the same indices as the original df, but only with the pna/mean features
    pd.DataFrame(
        pna_features, columns=[f"{i}" for i in range(pna_features.shape[1])], index=df.index
    ).reset_index().to_csv(
        DATASET_DIR / "mixture_rdkit_pna_definitions_clean.csv", index=False
    )

    pd.DataFrame(
        mean_features,
        columns=[f"{i}" for i in range(mean_features.shape[1])],
        index=df.index,
    ).reset_index().to_csv(
        DATASET_DIR / "mixture_rdkit_mean_definitions_clean.csv", index=False
    )

    # create POM embeddings
    MODEL_PATH = Path(
        base_dir / "scripts_pom/gs-lf_models/pretrained_pom/"
    )  # where the pretrained model is saved
    max_pad_len = df.columns.size  # mixtures are padded
    features = []
    for i, row in df.iterrows():
        row = row.dropna()
        emb = get_embeddings_from_smiles(row.tolist(), file_path=MODEL_PATH)
        mix_mat = np.pad(
            emb, ((0, max_pad_len - emb.shape[0]), (0, 0)), constant_values=-999
        )
        features.append(mix_mat)
    features = np.stack(features, axis=0)

    # note that the ordering of the pom embeddings are preserved
    # so that we can look up the corresponding dataset/mixture label
    # based on the mixture_smi_definitions_clean.csv file
    np.savez(DATASET_DIR / "mixture_pom_embeddings.npz", features=features)
