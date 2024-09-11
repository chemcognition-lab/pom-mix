import os
import seaborn as sns
import pandas as pd
import numpy as np

from pom import utils

from sklearn.model_selection import StratifiedKFold, train_test_split


if __name__ == '__main__':
    DATASET_DIR = '../datasets/'
    df = pd.read_csv(f'{DATASET_DIR}/mixtures/mixtures_combined.csv')
    dataset_id = df['Dataset'].values
    os.makedirs(f'{DATASET_DIR}/mixtures/splits/', exist_ok=True)

    seed = 0
    n_splits = 5
    train_val_frac = 1. - 1./n_splits   # gives 20 for test set
    train_frac = 0.7/train_val_frac     # get 70/10 of total for train/val sets

    utils.set_seed(seed)

    # stratified by dataset, to ensure even distribution of samples from each dataset
    # this gives 80/20 split for cross validation
    splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for i, (train_idx, test_idx) in enumerate(splits.split(list(range(len(df))), dataset_id)):
        # do a split for 70/10/20 train/validation/test
        train_idx, valid_idx = train_test_split(train_idx, train_size=train_frac, random_state=seed)
        np.savez(f'{DATASET_DIR}/mixtures/splits/random_cv{i}.npz', identifier=f'cv{i}', training=train_idx, validation=valid_idx, testing=test_idx)

    

    