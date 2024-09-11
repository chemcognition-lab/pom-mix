import ast
import os
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles

# Inspired by gauche DataLoader
# https://github.com/leojklarner/gauche

current_dir = Path(__file__).parent
DATASET_DIR = current_dir.parents[1] / "datasets"

print(DATASET_DIR)


class DreamLoader:
    """
    Loads and cleans up your data
    """

    def __init__(self):
        self.features = None
        self.labels = None
        dataset_df = pd.read_csv(DATASET_DIR / "file_cleaning_features.csv")
        dataset_df.index = dataset_df["dataset"]
        dataset_df.drop(columns=["unclean", "label_columns", "dataset"], inplace=True)
        dataset_df.rename({"new_label_columns": "labels"}, axis=1, inplace=True)

        def parse_value(value):
            if isinstance(value, str):
                try:
                    return ast.literal_eval(value)
                except Exception:
                    return value
            return value

        # Create the dictionary of dictionaries
        self.datasets = {}
        for dataset, row in dataset_df.iterrows():
            self.datasets[dataset] = {col: parse_value(val) for col, val in row.items()}
        self.datasets.update(
            {
                "mixtures": {
                    "features": ["Dataset", "Mixture 1", "Mixture 2"],
                    "labels": ["Experimental Values"],
                    "validate": False,  # nan values in columns, broken
                },
            }
        )

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

    def load_benchmark(
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
            "graphein_molecular_graphs",
            "pyg_molecular_graphs",
            "molecular_graphs",
            "morgan_fingerprints",
            "rdkit2d_normalized_features",
            "mordred_descriptors",
            "smiles",
            "competition_rdkit2d",
        ]

        if isinstance(representation, Callable):
            self.features = representation(self.features, **kwargs)

        elif representation == "graphein_molecular_graphs":
            from .representations.graphs import graphein_molecular_graphs

            self.features = graphein_molecular_graphs(smiles=self.features, **kwargs)

        elif representation == "pyg_molecular_graphs":
            from .representations.graphs import pyg_molecular_graphs

            self.features = pyg_molecular_graphs(smiles=self.features, **kwargs)

        elif representation == "molecular_graphs":
            from .representations.graphs import molecular_graphs

            self.features = molecular_graphs(smiles=self.features, **kwargs)

        elif representation == "morgan_fingerprints":
            from .representations.features import morgan_fingerprints

            self.features = morgan_fingerprints(self.features, **kwargs)

        elif representation == "rdkit2d_normalized_features":
            from .representations.features import rdkit2d_normalized_features

            self.features = rdkit2d_normalized_features(self.features, **kwargs)

        elif representation == "mordred_descriptors":
            from .representations.features import mordred_descriptors

            self.features = mordred_descriptors(self.features, **kwargs)

        elif representation == 'smiles':
            paths = {
                "smiles": "mixtures/mixture_smi_definitions_clean.csv",
            }
            path = paths[representation]
            # Features is ["Dataset", "Mixture 1", "Mixture 2"]
            smi_df = pd.read_csv(DATASET_DIR / path)
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

        elif representation == "smiles_augment":
            # Features is ["Dataset", "Mixture 1", "Mixture 2"]
            smi_df = pd.read_csv(
                DATASET_DIR / "mixtures/mixture_smi_definitions_clean.csv"
            )
            feature_list = []
            feature_list_augment = []
            for feature in self.features:
                mix_1 = smi_df.loc[
                    (smi_df["Dataset"] == feature[0])
                    & (smi_df["Mixture Label"] == feature[1])
                ][smi_df.columns[2:]]
                mix_1 = mix_1.dropna(axis=1).to_numpy()[0]
                mix_2 = smi_df.loc[
                    (smi_df["Dataset"] == feature[0])
                    & (smi_df["Mixture Label"] == feature[2])
                ][smi_df.columns[2:]]
                mix_2 = mix_2.dropna(axis=1).to_numpy()[0]
                feature_list.append([mix_1, mix_2])
                feature_list_augment.append([mix_2, mix_1])
            feature_list += feature_list_augment

            self.features = np.array(feature_list, dtype=object)
            self.labels = np.concatenate([self.labels, self.labels])

        elif representation == "competition_rdkit2d":
            # Features is ["Dataset", "Mixture 1", "Mixture 2"]
            rdkit_df = pd.read_csv(
                f"{current_dir}/../datasets/competition_train/mixture_rdkit_definitions_clean.csv"
            )
            feature_list = []
            for feature in self.features:
                mix_1 = rdkit_df.loc[
                    (rdkit_df["Dataset"] == feature[0])
                    & (rdkit_df["Mixture Label"] == feature[1])
                ][rdkit_df.columns[2:]]
                mix_1 = mix_1.dropna(axis=1).to_numpy()[0]
                mix_2 = rdkit_df.loc[
                    (rdkit_df["Dataset"] == feature[0])
                    & (rdkit_df["Mixture Label"] == feature[2])
                ][rdkit_df.columns[2:]]
                mix_2 = mix_2.dropna(axis=1).to_numpy()[0]
                feature_list.append([mix_1, mix_2])

            self.features = np.array(feature_list, dtype=object)

        elif representation == "competition_rdkit2d_augment":
            # Features is ["Dataset", "Mixture 1", "Mixture 2"]
            rdkit_df = pd.read_csv(
                f"{current_dir}/../datasets/competition_train/mixture_rdkit_definitions_clean.csv"
            )
            feature_list = []
            feature_list_augment = []
            for feature in self.features:
                mix_1 = rdkit_df.loc[
                    (rdkit_df["Dataset"] == feature[0])
                    & (rdkit_df["Mixture Label"] == feature[1])
                ][rdkit_df.columns[2:]]
                mix_1 = mix_1.dropna(axis=1).to_numpy()[0]
                mix_2 = rdkit_df.loc[
                    (rdkit_df["Dataset"] == feature[0])
                    & (rdkit_df["Mixture Label"] == feature[2])
                ][rdkit_df.columns[2:]]
                mix_2 = mix_2.dropna(axis=1).to_numpy()[0]
                feature_list.append([mix_1, mix_2])
                feature_list_augment.append([mix_2, mix_1])
            feature_list += feature_list_augment
            self.features = np.array(feature_list, dtype=object)
            self.labels = np.concatenate([self.labels, self.labels])

        elif representation == "only_augment":

            feature_list_augment = np.array([[x[1], x[0]] for x in self.features])

            self.features = np.vstack((self.features, feature_list_augment))

        else:
            raise Exception(
                f"The specified representation choice {representation} is not a valid option."
                f"Choose between {valid_representations}."
            )
        

class SplitLoader:
    def __init__(self, split_set: str = "random_cv"):
        assert split_set in ["random_cv", "albate_components", "ablate_molecules"]
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
            idxs.append((
                idx['identifier'],
                (features[idx['training']], labels[idx['training']]), 
                (features[idx['validation']], labels[idx['validation']]), 
                (features[idx['testing']], labels[idx['testing']]), 
            ))

        return idxs

