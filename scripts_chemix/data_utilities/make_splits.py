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

def create_k_molecules_split(
        features: np.ndarray, k: int, valid_percent: Optional[float] = 0.1
    ) -> Tuple[list[int], list[int], list[int]]:
    """
    Ablation splits of dataset into mixtures that are <= k and mixtures that have > k molecules (testing).

    Parameters:
    features (np.ndarray): Array of features where each row represents a mixture.
    k (int): Maximum number of molecules in a mixture for it to be included in the training set.
    valid_percent (Optional[float]): Percentage of the training data to use for validation. Default is 0.1.

    Returns:
    Tuple[list[int], list[int], list[int]]: A tuple containing lists of indices for the training, validation, and test sets.
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
    """
    Create ablation splits of the dataset based on the exclusion of specific molecules, ensuring that certain molecules do not appear in the training set.

    Parameters:
    mixture_smiles (np.ndarray): Array of molecule mixtures, where each mixture is a list of SMILES strings.
    num_splits (Optional[int]): Number of splits to create. Default is 5.
    valid_percent (Optional[float]): Percentage of the data to use for validation. Default is 0.1.

    Returns:
    Tuple[list[int], list[int], list[int]]: A tuple containing lists of indices for the training, validation, and test sets for each split.
    """
    # Flatten the mixture smiles, since we care only about identity of molecules found in any mixture
    mixture_smiles = np.array([np.concatenate(item) for item in mixture_smiles], dtype=object)

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


def create_lso_molecule_identity_splits(
    mixture_smiles: np.ndarray, num_splits: Optional[int] = 5, valid_percent: Optional[float] = 0.1, tolerance: Optional[float] = 0.005,
    ) -> Tuple[list[int], list[int], list[int]]:
    """
    Create "Leave Some Out" (lso) splits of the dataset based on the exclusion of specific molecules, ensuring that certain molecules do 
    not appear in the training set.

    Parameters:
    mixture_smiles (np.ndarray): Array of molecule mixtures, where each mixture is a list of SMILES strings.
    num_splits (Optional[int]): Number of splits to create. Default is 5.
    valid_percent (Optional[float]): Percentage of the data to use for validation. Default is 0.1.
    tolerance (Optional[float]): Tolerance for the size of the test set. Default is 0.005.

    Returns:
    Tuple[list[int], list[int], list[int]]: A tuple containing lists of indices for the training, validation, and test sets for each split.
    """
    # Flatten the mixture smiles, since we care only about identity of molecules found in any mixture
    mixture_smiles = np.array([np.concatenate(item) for item in mixture_smiles], dtype=object)

    # Determine the size of splits
    test_percent = 1./num_splits
    train_percent = 1.0 - test_percent - valid_percent
    N = len(mixture_smiles)

    # Flatten the list of lists and count the frequency of each string
    all_indices = list(range(mixture_smiles.shape[0]))
    all_strings = np.concatenate(mixture_smiles.ravel())

    # incrementally 
    splits, smiles_removed = [], []
    for i in range(num_splits):
        while True:
            excluded_mixtures, excluded_smiles = [], []
            sample_strings = all_strings.tolist().copy()
            remaining_mixtures = mixture_smiles.tolist().copy()
            while len(excluded_mixtures)/N < test_percent - tolerance:
                # select a smiles for exclusion
                smi = np.random.choice(sample_strings, size=1)[0]
                excluded_smiles.append(smi)
                excluded_mixtures.extend([arr for arr in remaining_mixtures if smi in arr])
                sample_strings.remove(smi)     # remove it so it won't be sampled again
                remaining_mixtures = [arr for arr in remaining_mixtures if not any(np.array_equal(arr, excl) for excl in excluded_mixtures)]  # remove arrays that have been added to excluded_mixtures
                assert len(remaining_mixtures + excluded_mixtures) == N, 'List of excluded and remaining not adding up to full dataset.'

                if len(excluded_mixtures)/N > test_percent + tolerance:
                    print('Too many excluded sets. Reset.')
                    break
                
            if (len(excluded_mixtures) / N >= test_percent - tolerance and 
                len(excluded_mixtures) / N <= test_percent + tolerance and 
                excluded_smiles not in smiles_removed):
                # For each array of strings in remaining_mixtures, get the indices as seen inside of mixture_smiles
                remaining_indices = [all_indices[j] for j, sublist in enumerate(mixture_smiles) if any(np.array_equal(sublist, rem) for rem in remaining_mixtures)]
                test_indices = [all_indices[j] for j, sublist in enumerate(mixture_smiles) if any(np.array_equal(sublist, excl) for excl in excluded_mixtures)]

                # perform a split on the remaining indices, which makes up the train and val set
                train_indices, valid_indices = train_test_split(remaining_indices, test_size=valid_percent/(train_percent + valid_percent))
                splits.append((train_indices, valid_indices, test_indices))
                smiles_removed.append(excluded_smiles)
                print(f'Complete split {i}.')
                break
    
    return splits, smiles_removed

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
    for k in [5, 7, 8, 9, 10, 17, 20, 27, 30, 40]:
        train_idx, valid_idx, test_idx = create_k_molecules_split(dl.features, k = k)
        np.savez(OUTPUT_DIR / f'ablate_components{k}.npz', identifier=f'k{k}', training=train_idx, validation=valid_idx, testing=test_idx)

    # generate molecule ablation data splits
    # certain molecules do not show up in the train set
    # these are decided based on analysis of frequency of molecules in each mixture
    all_splits = create_molecule_identity_splits(dl.features, num_splits=8)
    for i, (train_idx, valid_idx, test_idx) in enumerate(all_splits):
        np.savez(OUTPUT_DIR / f'ablate_molecules{i}.npz', identifier=f'm{i}', training=train_idx, validation=valid_idx, testing=test_idx)

    all_splits, removed_smiles = create_lso_molecule_identity_splits(dl.features, num_splits=5)
    print(f'Here are the smiles that are only found in test set: {removed_smiles}')
    for i, (train_idx, valid_idx, test_idx) in enumerate(all_splits):
        np.savez(OUTPUT_DIR / f'lso_molecules{i}.npz', identifier=f'lso{i}', training=train_idx, validation=valid_idx, testing=test_idx, 
                 removed_smiles=np.array(removed_smiles, dtype=object))




    


    

    

    