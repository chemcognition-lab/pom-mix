from typing import Tuple, Optional

import os, sys
import random
import seaborn as sns
import pandas as pd
import numpy as np

from pathlib import Path
script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-2])
sys.path.append( str(base_dir / 'src/') )


from pathlib import Path
from collections import Counter

from pom import utils
from dataloader import DatasetLoader

from sklearn.model_selection import StratifiedKFold, train_test_split

def calculate_inner_lengths(row):
    return [len(inner_arr) for inner_arr in row]

def create_k_molecules_split(features: np.ndarray, k: int, valid_percent: Optional[float] = 0.1) -> Tuple[list[int], list[int], list[int]]:
    """
    splits dataset into mixtures that are <= k and mixtures that have > k molecules (testing)
    further create a validation split from the training set
    """

    # get mixture lengths
    mixture_lengths = np.array([calculate_inner_lengths(row) for row in features])
    mask = np.all(mixture_lengths <= k, axis=1)
    train_indices = np.where(mask)[0]
    test_indices = np.where(~mask)[0]
    
    # get indices for training and validation
    train_indices, valid_indices = train_test_split(train_indices, test_size=valid_percent)
    
    return train_indices, valid_indices, test_indices


def find_closest_index(value, arr):
    arr = np.array(arr)  # Convert to numpy array for convenience
    index = np.abs(arr - value).argmin()
    return index

def create_molecule_identity_splits(
        mixture_smiles: np.ndarray, num_splits: Optional[int] = 5, valid_percent: Optional[float] = 0.1,
    ) -> Tuple[list[int], list[int], list[int]]:
    # Flatten the list of lists and count the frequency of each string
    all_indices = list(range(mixture_smiles.shape[0]))
    all_strings = np.concatenate(mixture_smiles.ravel())
    string_counts = Counter(all_strings.tolist())
    sorted_strings = sorted(string_counts, key=string_counts.get, reverse=False)

    # progressively build the training set based on frequency of molecules
    training_increments = [[]]
    training_counts = []
    for i, smi in enumerate(sorted_strings):
        inds = [j for j, sublist in enumerate(mixture_smiles) if any(item == smi for item in sublist)]
        inds = sorted(list(set(inds + training_increments[i])))
        training_counts.append(len(inds))
        training_increments.append(inds)
    training_increments.pop(0)

    # overlaps in molecular identity will result in unchanged training sets
    # reverse and then get unique, this will pick most inclusive set
    training_counts, uniq_idx = np.unique(training_counts[::-1], return_index=True)
    training_increments = np.array(training_increments[::-1], dtype=object)[uniq_idx].tolist()

    # get the increments that are give training sizes closest to equal spacing for ablation test
    split_space = np.linspace(0, len(mixture_smiles), num_splits+2)[1:-1]
    closest_indices = [find_closest_index(v, training_counts) for v in split_space]
    training_increments = np.array(training_increments, dtype=object)[closest_indices].tolist()
    training_counts = training_counts[closest_indices]

    # generate final validation split
    splits = []
    for i, train_indices in enumerate(training_increments):
        # Split based on exclusion of specific molecules
        test_indices = [j for j in all_indices if j not in train_indices]
        train_indices, valid_indices = train_test_split(train_indices, test_size=valid_percent)
        splits.append((train_indices, valid_indices, test_indices))
    
    return splits

if __name__ == '__main__':
    DATASET_DIR = base_dir / Path(f'datasets/mixtures')
    OUTPUT_DIR = DATASET_DIR / 'splits/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # set seed for split generation
    seed = 0
    utils.set_seed(seed)

    # stratified by dataset, to ensure even distribution of samples from each dataset
    # this gives 80/20 split for cross validation
    df = pd.read_csv(DATASET_DIR / "mixtures_combined.csv")
    dataset_id = df['Dataset'].values
    n_splits = 5
    train_val_frac = 1. - 1./n_splits   # gives 20 for test set
    train_frac = 0.7/train_val_frac     # get 70/10 of total for train/val sets

    splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(splits.split(list(range(len(df))), dataset_id)):
        # do a split for 70/10/20 train/validation/test
        train_idx, valid_idx = train_test_split(train_idx, train_size=train_frac, random_state=seed)
        np.savez(OUTPUT_DIR / f"random_cv{i}.npz", identifier=f'cv{i}', training=train_idx, validation=valid_idx, testing=test_idx)


    # generate < k (num components) ablation data splits
    # then further create a split from the training set to for validation
    dl = DatasetLoader()
    dl.load_dataset('mixtures')
    dl.featurize('mix_smiles')
    for k in [5, 7, 10, 15, 20, 25, 30, 40]:
        train_idx, valid_idx, test_idx = create_k_molecules_split(dl.features, k = k)
        np.savez(OUTPUT_DIR / f'ablate_components{k}.npz', identifier=f'k{k}', training=train_idx, validation=valid_idx, testing=test_idx)

    # generate molecule ablation data splits
    # certain molecules do not show up in the train set
    # these are decided based on analysis of frequency of molecules in each mixture
    features = np.array([np.concatenate(item) for item in dl.features], dtype=object)
    all_splits = create_molecule_identity_splits(features, num_splits=8)
    for i, (train_idx, valid_idx, test_idx) in enumerate(all_splits):
        np.savez(OUTPUT_DIR / f'ablate_molecules{i}.npz', identifier=f'm{i}', training=train_idx, validation=valid_idx, testing=test_idx)


    


    

    

    