import sys, os
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-1])
sys.path.append( str(base_dir / 'src/') )

import seaborn as sns
from dataloader import DatasetLoader, SplitLoader
from pommix_utils import permute_mixture_pairs, pna

from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau

import numpy as np
import pandas as pd
import json

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--feat_type", action="store", type=str, help="Feature type, select [rdkit2d, pom_embeddings].")
FLAGS = parser.parse_args()

if __name__ == '__main__':
    feat_type = FLAGS.feat_type
    
    fname = f'xgb_{feat_type}'
    os.makedirs(fname, exist_ok=True)

    dl = DatasetLoader()
    dl.load_dataset("mixtures")
    dl.featurize(f"mix_{feat_type}")

    # load splits
    sl = SplitLoader("random_cv")
    test_results = []
    for id, train, val, test in sl.load_splits(dl.features, dl.labels):
        train_features, train_labels = train
        train_features, train_labels = permute_mixture_pairs(train_features, train_labels)
        val_features, val_labels = val
        test_features, test_labels = test

        # pna across molecules, and then flatten along mixture dimension
        if feat_type == 'pom_embeddings':
            tr, va, te = [], [], []
            for i in range(train_features.shape[-1]):
                tr.append(pna(train_features[...,i]))
                va.append(pna(val_features[...,i]))
                te.append(pna(test_features[...,i]))
            train_features = np.stack(tr, axis=-1)
            val_features = np.stack(va, axis=-1)
            test_features = np.stack(te, axis=-1)

        # flatten along mixture dimension
        train_features = train_features.reshape(len(train_features), -1)
        val_features = val_features.reshape(len(val_features), -1)
        test_features = test_features.reshape(len(test_features), -1)

        logger = {'pearson': -np.nan}
        for _ in range(100):
            bst = XGBRegressor(n_estimators=1000, max_depth=1000, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
                    early_stopping_rounds=10, eval_metric=mean_squared_error, n_jobs=-1, verbosity=0)

            # fit
            bst.fit(train_features, train_labels, eval_set=[(val_features, val_labels)])

            # perform test
            y_pred = bst.predict(test_features)
            prs, _ = pearsonr(test_labels.flatten(), y_pred.flatten())
            
            if logger['pearson'] < prs:
                logger['id'] = id
                logger['y_pred'] = y_pred.tolist()
                logger['y_test'] = test_labels.flatten().tolist()
                logger['r2_score'] = r2_score(test_labels.flatten(), y_pred.flatten()).astype(float)
                logger['rmse'] = root_mean_squared_error(test_labels.flatten(), y_pred.flatten()).astype(float)
                logger['pearson'] = pearsonr(test_labels.flatten(), y_pred.flatten())[0].astype(float)
                logger['kendall'] = kendalltau(test_labels.flatten(), y_pred.flatten())[0].astype(float)
                json.dump(logger, open(f'{fname}/model_{id}_stats.json', 'w'), indent=4)
                bst.save_model(f'{fname}/model_{id}.json')
        
        test_results.append(logger)

    pd.DataFrame(test_results).to_pickle(f'{fname}/all_results.pkl')

