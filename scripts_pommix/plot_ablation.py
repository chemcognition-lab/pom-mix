import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-1])
sys.path.append( str(base_dir / 'src/') )

from pommix_utils import bootstrap_ci
from chemix.utils import TORCH_METRIC_FUNCTIONS
from dataloader import DatasetLoader, SplitLoader

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--split", action="store", type=str, default="ablate_components", choices=["ablate_molecules", "ablate_components"])

FLAGS = parser.parse_args()

if __name__ == '__main__':
    DATA_DIR = Path(f'results/{FLAGS.split}/top1')

    # training set
    dl = DatasetLoader()
    dl.load_dataset('mixtures')
    dl.featurize('mix_smiles')

    sl = SplitLoader(FLAGS.split)
    X, Y, y_upper, y_lower = [], [], [], []
    for id, train, _, _ in sl.load_splits(dl.features, dl.labels):
        train_feat, _  = train
        df = pd.read_csv(DATA_DIR / f"{id}_test_predictions.csv")
        
        x = len(train_feat)
        metric_fn = TORCH_METRIC_FUNCTIONS['pearson']
        y_true = torch.from_numpy(df['Predicted_Experimental_Values'].to_numpy(np.float32))
        y_pred = torch.from_numpy(df['Ground_Truth'].to_numpy(np.float32))
        lower_bound, upper_bound, values = bootstrap_ci(y_true, y_pred, metric_fn)
        X.append(x)
        Y.append(np.mean(values))
        y_upper.append(upper_bound)
        y_lower.append(lower_bound)


    plt.figure(figsize=(10, 6))
    plt.errorbar(X, Y, yerr=[np.array(Y) - np.array(y_lower), np.array(y_upper) - np.array(Y)], fmt='o', ecolor='r', capsize=5, label='Prediction Interval')
    plt.scatter(X, Y, color='b', label='Predictions')
    plt.xlabel('Number of Training Features')
    plt.ylabel('Mean Pearson Correlation')
    plt.title('Ablation Study Results')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{FLAGS.split}_plot.png', bbox_inches='tight')



    
        

        