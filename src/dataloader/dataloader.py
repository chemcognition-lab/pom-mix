import ast
import sys

sys.path.append("..")

from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles
import itertools
from scipy.spatial.distance import pdist, squareform

import pommix_utils

# Inspired by gauche DataLoader
# https://github.com/leojklarner/gauche

current_dir = Path(__file__).parent
DATASET_DIR = current_dir.parents[1] / "datasets"

print(DATASET_DIR)


class DatasetLoader:
    """
    Loads and cleans up your data
    """

    def __init__(self):
        self.features = None
        self.labels = None

        # Create the dictionary of dictionaries
        self.datasets = {}

        self.datasets.update(
            {
                "gs-lf": {
                    "features": ["IsomericSMILES"],
                    "labels": ['alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal', 'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 
                               'beefy', 'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt', 'buttery', 'cabbage', 'camphoreous', 'caramellic', 
                               'cedar', 'celery', 'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean', 'clove', 'cocoa', 'coconut', 
                               'coffee', 'cognac', 'cooked', 'cooling', 'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 'ethereal', 
                               'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin', 'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 
                               'grassy', 'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth', 'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 
                               'leafy', 'leathery', 'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic', 'milky', 'mint', 'muguet', 'mushroom', 
                               'musk', 'musty', 'natural', 'nutty', 'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone', 'peach', 'pear', 
                               'phenolic', 'pine', 'pineapple', 'plum', 'popcorn', 'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted', 'rose', 
                               'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy', 'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet', 
                               'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable', 'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'],
                    "n_datapoints": 4814,
                    "task": "multilabel",
                    "task_dim": 138,
                    "validate": True,  
                },

                "mixtures": {
                    "features": ["Dataset", "Mixture 1", "Mixture 2"],
                    "labels": ["Experimental Values"],
                    "validate": False,  # nan values in columns, broken
                },
            }
        )

        self.is_mixture = False

    def get_dataset_names(self, valid_only: Optional[bool] = True) -> List[str]:
        names = []
        if valid_only:
            for k, v in self.datasets.items():
                if v["validate"]:
                    names.append(k)
        else:
            names = list(self.datasets.keys())
        return names

    def get_dataset_specifications(self, name: str) -> dict:
        assert name in self.datasets.keys(), (
            f"The specified dataset choice ({name}) is not a valid option. "
            f"Choose one of {list(self.datasets.keys())}."
        )
        return self.datasets[name]

    def read_csv(
        self,
        path: str,
        smiles_column: List[str],
        label_columns: List[str],
        validate: bool = True,
    ) -> None:
        """
        Loads a csv and stores it as features and labels.
        """
        assert isinstance(
            smiles_column, List
        ), f"smiles_column ({smiles_column}) must be a list of strings"
        assert isinstance(label_columns, list) and all(
            isinstance(item, str) for item in label_columns
        ), "label_columns ({label_columns}) must be a list of strings."

        df = pd.read_csv(path, usecols=smiles_column + label_columns)

        self.features = df[smiles_column].to_numpy()
        if len(smiles_column) == 1:
            self.features = self.features.flatten()
        self.labels = df[label_columns].values
        if validate:
            self.validate()

    def load_dataset(
        self,
        name: str,
        path=None,
        validate: bool = True,
    ) -> None:
        """
        Pulls existing benchmark from datasets.
        """
        assert name in self.datasets.keys(), (
            f"The specified dataset choice ({name}) is not a valid option. "
            f"Choose one of {list(self.datasets.keys())}."
        )

        path = path or DATASET_DIR / name / f"{name}_combined.csv"

        self.read_csv(
            path=path,
            smiles_column=self.datasets[name]["features"],
            label_columns=self.datasets[name]["labels"],
            validate=self.datasets[name]["validate"],
        )

        self.is_mixture = name == "mixtures"
        if self.is_mixture:
            self.dataset_id = self.features[:, 0]  # expose dataset id for splitting
        else:
            self.dataset_id = None

        if not self.datasets[name]["validate"]:
            print(
                f"{name} dataset is known to have invalid entries. Validation is turned off."
            )

    def validate(
        self, drop: Optional[bool] = True, canonicalize: Optional[bool] = True
    ) -> None:
        """
        Utility function to validate a read-in dataset of smiles and labels by
        checking that all SMILES strings can be converted to rdkit molecules
        and that all labels are numeric and not NaNs.
        Optionally drops all invalid entries and makes the
        remaining SMILES strings canonical (default).

        :param drop: whether to drop invalid entries
        :type drop: bool
        :param canonicalize: whether to make the SMILES strings canonical
        :type canonicalize: bool
        """
        invalid_mols = np.array(
            [True if MolFromSmiles(x) is None else False for x in self.features]
        )
        if np.any(invalid_mols):
            print(
                f"Found {invalid_mols.sum()} SMILES strings "
                f"{[x for i, x in enumerate(self.features) if invalid_mols[i]]} "
                f"at indices {np.where(invalid_mols)[0].tolist()}"
            )
            print(
                "To turn validation off, use dataloader.read_csv(..., validate=False)."
            )

        invalid_labels = np.isnan(self.labels).squeeze()
        if np.any(invalid_labels):
            print(
                f"Found {invalid_labels.sum()} invalid labels "
                f"{self.labels[invalid_labels].squeeze()} "
                f"at indices {np.where(invalid_labels)[0].tolist()}"
            )
            print(
                "To turn validation off, use dataloader.read_csv(..., validate=False)."
            )
        if invalid_labels.ndim > 1:
            invalid_idx = np.any(
                np.hstack((invalid_mols.reshape(-1, 1), invalid_labels)), axis=1
            )
        else:
            invalid_idx = np.logical_or(invalid_mols, invalid_labels)

        if drop:
            self.features = [
                x for i, x in enumerate(self.features) if not invalid_idx[i]
            ]
            self.labels = self.labels[~invalid_idx]
            assert len(self.features) == len(self.labels)

        if canonicalize:
            self.features = [
                MolToSmiles(MolFromSmiles(smiles), isomericSmiles=False)
                for smiles in self.features
            ]

    def featurize(self, representation: Union[str, Callable], **kwargs) -> None:
        """Transforms SMILES into the specified molecular representation.

        :param representation: the desired molecular representation.
        :type representation: str or Callable
        :param kwargs: additional keyword arguments for the representation function
        :type kwargs: dict
        """

        assert isinstance(representation, (str, Callable)), (
            f"The specified representation choice {representation} is not "
            f"a valid type. Please choose a string from the list of available "
            f"representations or provide a callable that takes a list of "
            f"SMILES strings as input and returns the desired featurization."
        )

        valid_representations = [
            "molecular_graphs",
            "morgan_fingerprints",
            "rdkit2d",
            "mordred_descriptors",
            "mix_smiles",
            "mix_rdkit2d",
            "mix_pom_embeddings",
            "mix_molt5_embeddings"
        ]

        if isinstance(representation, Callable):
            self.features = representation(self.features, **kwargs)

        elif representation == "pyg_molecular_graphs":
            from .representations.graphs import pyg_molecular_graphs
            self.features = pyg_molecular_graphs(smiles=self.features, **kwargs)

        elif representation == "molecular_graphs":
            from .representations.graphs import molecular_graphs

            self.features = molecular_graphs(smiles=self.features, **kwargs)

        elif representation == "morgan_fingerprints":
            from .representations.features import morgan_fingerprints

            self.features = morgan_fingerprints(self.features, **kwargs)

        elif representation == "rdkit2d":
            from .representations.features import rdkit2d_normalized_features

            self.features = rdkit2d_normalized_features(self.features, **kwargs)

        elif representation == "mordred_descriptors":
            from .representations.features import mordred_descriptors

            self.features = mordred_descriptors(self.features, **kwargs)

        elif representation == "mix_smiles":
            # Features is ["Dataset", "Mixture 1", "Mixture 2"]
            smi_df = pd.read_csv(
                DATASET_DIR / "mixtures/mixture_smi_definitions_clean.csv"
            )
            smi_df = smi_df.drop(['length', 'Duplicate', 'Duplicate Of'], axis=1)

            smi_df = smi_df.set_index(["Dataset", "Mixture Label"])
            feature_list = np.empty(
                (len(self.features), 2), dtype=object
            )  # assumed to be 2 mixtures
            for mixid, feature in enumerate(self.features):
                for mi in range(0, 2):
                    index = (feature[0], feature[mi + 1])
                    smiles_arr = smi_df.loc[index].dropna().to_numpy()
                    feature_list[mixid, mi] = smiles_arr
            self.features = feature_list

        elif representation in ["mix_rdkit2d"]:
            # Features is ["Dataset", "Mixture 1", "Mixture 2"]
            smi_df = pd.read_csv(
                DATASET_DIR / "mixtures/mixture_smi_definitions_clean.csv"
            )
            smi_df = smi_df.drop(['length', 'Duplicate', 'Duplicate Of'], axis=1)
            
            # note that the rows correspond to each other
            rdkit_feat = np.load(DATASET_DIR / f"mixtures/mixture_rdkit2d.npz")[
                "features"
            ]

            feature_list = []
            for feature in self.features:
                mix_1 = smi_df.loc[
                    (smi_df["Dataset"] == feature[0])
                    & (smi_df["Mixture Label"] == feature[1])
                ][smi_df.columns[2:]].index[0]
                mix_2 = smi_df.loc[
                    (smi_df["Dataset"] == feature[0])
                    & (smi_df["Mixture Label"] == feature[2])
                ][smi_df.columns[2:]].index[0]
                feature_list.append(
                    np.stack([rdkit_feat[mix_1], rdkit_feat[mix_2]], axis=-1)
                )
            self.features = np.array(feature_list, dtype=float)

            # further processing
            if "pna" in representation:
                feat_reduced = []
                for i in range(self.features.shape[-1]):
                    feat_reduced.append(pommix_utils.pna(self.features[...,i]))
                self.features = np.stack(feat_reduced, axis=-1)
            elif "mean" in representation:
                feat_reduced = []
                for i in range(self.features.shape[-1]):
                    feat_reduced.append(pommix_utils.mean(self.features[...,i]))
                self.features = np.stack(feat_reduced, axis=-1)


        elif representation in ["mix_pom_embeddings", "mix_molt5_embeddings"]:
            # Features is ["Dataset", "Mixture 1", "Mixture 2"]
            tag = representation.replace("mix_", "")
            smi_df = pd.read_csv(
                DATASET_DIR / "mixtures/mixture_smi_definitions_clean.csv"
            )
            smi_df = smi_df.drop(['length', 'Duplicate', 'Duplicate Of'], axis=1)

            # load the pom embeddings
            # note that the rows correspond to each other
            embeds = np.load(DATASET_DIR / f"mixtures/mixture_{tag}.npz")[
                "features"
            ]

            feature_list = []
            for feature in self.features:
                mix_1 = smi_df.loc[
                    (smi_df["Dataset"] == feature[0])
                    & (smi_df["Mixture Label"] == feature[1])
                ][smi_df.columns[2:]].index[0]
                mix_2 = smi_df.loc[
                    (smi_df["Dataset"] == feature[0])
                    & (smi_df["Mixture Label"] == feature[2])
                ][smi_df.columns[2:]].index[0]
                feature_list.append(
                    np.stack([embeds[mix_1], embeds[mix_2]], axis=-1)
                )

            self.features = np.stack(feature_list, axis=0)

        else:
            raise Exception(
                f"The specified representation choice {representation} is not a valid option."
                f"Choose between {valid_representations}."
            )
        
    def reduce(self, reduce_type:str = "mean"):
        if not self.is_mixture:
            raise Exception("Can only augment mixtures dataset.")
        print(f'Reducing each mixture by: {reduce_type}. Select between ["pna", "mean"]')
        
        # reduce the mixture along the molecular axis
        reduce_method = {
            'pna': pommix_utils.pna,
            'mean': pommix_utils.mean,
        }

        # loop along mixtures dimension
        feat_reduced = []
        for i in range(self.features.shape[-1]):
            feat_reduced.append(reduce_method[reduce_type](self.features[...,i]))
        self.features = np.stack(feat_reduced, axis=-1)
        

    def augment(self, augment_type: str = None):
        if not self.is_mixture:
            raise Exception("Can only augment mixtures dataset.")
        if not augment_type:
            raise Exception(
                "Must specify augment strategy: 'permute_mixture_pairs', 'self_mixture_unity', 'single_molecule_mixture_gslf_jaccards'"
            )
        if augment_type == "permute_mixture_pairs":
            self.features, self.labels = self.permute_mixture_pairs(
                self.features, self.labels
            )
        elif augment_type == "self_mixture_unity":
            self.features, self.labels = self.self_mixture_unity(
                self.features, self.labels
            )
        elif augment_type == "single_molecule_mixture_gslf_jaccards":
            self.features, self.labels = self.single_molecule_mixture_gslf_jaccards(
                self.features, self.labels
            )
        else:
            raise Exception(
                "Augment strategy must be 'permute_mixture_pairs', 'self_mixture_unity', 'single_molecule_mixture_gslf_jaccards'"
            )

    @staticmethod
    def permute_mixture_pairs(features, labels):
        """
        Augments the given mixture pairs by creating a new feature list and concatenating it with the original features, but permuted.
        Assumes that the last dimension is the dimension of mixtures pairs

        Args:
            features (ndarray): The original feature list.
            labels (ndarray): The original label list.

        Returns:
            tuple: A tuple containing the augmented features and labels.
        """
        feature_list_augment = np.array(
            [np.stack([x[..., 1], x[..., 0]], axis=-1) for x in features]
        )
        features = np.vstack((features, feature_list_augment))
        labels = np.concatenate([labels, labels])
        return features, labels

    @staticmethod
    def self_mixture_unity(features, labels):
        """
        Augments the given mixture pairs by enforcing that the existing mixtures' self distance is 1.
        Assumes that the last dimension is the dimension of mixtures pairs

        Args:
            features (ndarray): The original feature list.
            labels (ndarray): The original label list.

        Returns:
            tuple: A tuple containing the augmented features and labels.
        """
        for mixture_dim in [0, 1]:
            if features[0].dtype == "O":  # check if it is a series of smiles strings
                unique_mixtures = np.array([])
                seen = set()
                for sub_array in features[..., mixture_dim]:
                    fset = frozenset(sub_array)
                    if fset not in seen:
                        seen.add(fset)
                        unique_mixtures = np.append(unique_mixtures, sub_array)
            else:
                unique_mixtures = np.unique(features[..., mixture_dim], axis=0)
            feature_list_augment = np.array(
                [np.stack([x, x], axis=-1) for x in unique_mixtures]
            )
            features = np.vstack((features, feature_list_augment))
            labels = np.concatenate([labels, np.zeros((len(unique_mixtures), 1))])
        return features, labels

    @staticmethod
    def single_molecule_mixture_gslf_jaccards(features, labels):
        """
        Augments the mixture dataset by creating single-molecule mixtures and artificially specifying
        GS-LF label Jaccard distances as mixture perceptual distances.
        Assumes that the last dimension is the dimension of mixtures pairs

        Args:
            features (ndarray): The original feature list.
            labels (ndarray): The original label list.

        Returns:
            tuple: A tuple containing the augmented features and labels.
        """
        if features[0].dtype != "O":  # check if it is a series of smiles strings
            raise Exception(
                "Single molecule Jaccard augmentation only works with SMILES strings."
            )
        # Find unique molecules. Features has (n_mixtures, smiles_string, 2)
        # Flatten features and np unique
        unique_molecules = np.unique(np.concatenate(features.flatten(), axis=0))
        # Create all possible pairs using itertools.combination
        gslf_df = pd.read_csv(DATASET_DIR / "gs-lf/gs-lf_combined.csv")
        gslf_df = gslf_df[gslf_df["IsomericSMILES"].isin(unique_molecules)]
        gslf_df.drop(columns=["descriptors"], inplace=True)
        gslf_smiles = gslf_df.iloc[:, 0]
        gslf_labels = gslf_df.iloc[:, 1:]

        # Calculate pairwise Jaccard distances
        jaccard_distances = pdist(gslf_labels, metric="jaccard")

        # Convert distance vector to square matrix
        distance_matrix = squareform(jaccard_distances)

        # Create a new dataframe with pairwise distances
        jaccard_df = pd.DataFrame(
            distance_matrix, index=gslf_smiles, columns=gslf_smiles
        )

        # Create all possible pairs using itertools.combination
        actual_pairs = np.array(list(itertools.combinations(jaccard_df.columns, 2)))
        jaccard_distances = np.array([])
        for pair in actual_pairs:
            jaccard_distances = np.append(
                jaccard_distances, jaccard_df.loc[pair[0], pair[1]]
            )
        jaccard_distances = jaccard_distances.reshape(-1, 1)

        actual_pairs = np.array(
            [np.array([[pair[0]], [pair[1]]], dtype="object") for pair in actual_pairs],
            dtype="object",
        )

        # Stupid workaround for inhomogeneous third dimension of features
        concatenated = np.empty(
            (features.shape[0] + actual_pairs.shape[0], features.shape[1]),
            dtype="object",
        )
        concatenated[: features.shape[0], :] = features
        for i in range(actual_pairs.shape[0]):
            concatenated[features.shape[0] + i, 0] = actual_pairs[i, 0]
            concatenated[features.shape[0] + i, 1] = actual_pairs[i, 1]

        features = concatenated
        labels = np.concatenate([labels, jaccard_distances])

        return features, labels


class SplitLoader:
    def __init__(self, split_set: str = "random_cv"):
        assert split_set in [
            "random_cv",
            "ablate_components",
            "ablate_molecules",
            "lso_molecules",
            "random_train_val",
        ]
        self.split_set = split_set

    def load_splits(self, features, labels):
        """
        create a set of splits to look at
        suggested usage:
        for id, train, val, test in sl.load_splits(dl.features, dl.labels):
            print(id)
            train_features, train_labels = train
            val_features, val_labels = val
            test_features, test_labels = test

        """
        split_dir = DATASET_DIR / "mixtures/splits/"

        idxs = []
        for fname in sorted(split_dir.glob(f"{self.split_set}*.npz")):
            idx = np.load(fname)
            idxs.append(
                (
                    str(idx["identifier"]),
                    (features[idx["training"]], labels[idx["training"]]),
                    (features[idx["validation"]], labels[idx["validation"]]),
                    (features[idx["testing"]], labels[idx["testing"]]),
                )
            )

        return idxs
