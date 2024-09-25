import sys, os
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-1])
sys.path.append(str(base_dir / "src/"))

import seaborn as sns
from dataloader.dataloader import DatasetLoader, SplitLoader
from similarity_model.model import AngleSimilarityModel

from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score
from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold
from scipy.stats import pearsonr, kendalltau
import torch
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import json

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--feat_type",
    action="store",
    type=str,
    help="Feature type, select [rdkit2d, pom_embeddings].",
)
FLAGS = parser.parse_args()


def angle_similarity(a, b):
    return torch.acos(
        torch.clamp(
            torch.nn.functional.cosine_similarity(a, b, eps=1e-8),
            min=-1 + 1e-8,
            max=1 - 1e-8,
        )
    )


if __name__ == "__main__":
    np.random.seed(0)

    feat_type = FLAGS.feat_type

    fname = f"best_n_features_angle_sim_step2_mix_rdkit2d_mean"
    os.makedirs(fname, exist_ok=True)

    dl = DatasetLoader()
    dl.load_dataset("mixtures")
    dl.featurize(f"mix_rdkit2d_mean")

    # load splits
    sl = SplitLoader("random_cv")
    test_results = []
    for id, train, val, test in sl.load_splits(dl.features, dl.labels):
        best_n_results = {}
        angle_sim_model = AngleSimilarityModel()
        train_features, train_labels = train
        val_features, val_labels = val
        test_features, test_labels = test
        train_features = train_features.astype(float)
        val_features = val_features.astype(float)
        test_features = test_features.astype(float)
        train_labels = train_labels.astype(float)
        val_labels = val_labels.astype(float)
        test_labels = test_labels.astype(float)

        best_rmse = np.inf
        with open(f"./{"random_select_n_features_angle_sim_step1_mix_rdkit2d_mean"}/model_{id}_stats.json") as json_file:
            select_n = json.load(json_file)
        n_features = select_n["n_best"]
        # iterate through features
        for n in range(0, 200):
            best_n_results[n] = {}
            rmse_list = []
            # generate index with n features
            for i in range(2000):
                print(f"{n=}, {i=}")
                idx_features = np.random.choice(
                    train_features.shape[1], n_features-1, replace=False
                )
                if n in idx_features:
                    # add another feature not in the list
                    idx_features = np.append(
                        idx_features, np.random.choice(train_features.shape[1])
                    )
                else:
                    # add the nth feature
                    idx_features = np.append(idx_features, n)
                # similarity model between mixtures
                y_train = angle_similarity(
                    torch.tensor(train_features[:, idx_features, 0]),
                    torch.tensor(train_features[:, idx_features, 1]),
                )
                y_train = y_train.detach().numpy()
                rmse = root_mean_squared_error(
                    train_labels.flatten(), y_train.flatten()
                )
                rmse_list.append(rmse)
            best_n_results[n]["rmse_avg"] = np.mean(rmse_list)
            best_n_results[n]["rmse_std"] = np.std(rmse_list)

        logger = {"pearson": -np.nan}
        y_pred = angle_similarity(
            torch.tensor(test_features[:, :, 0]), torch.tensor(test_features[:, :, 1])
        )
        prs, _ = pearsonr(test_labels.flatten(), y_pred.flatten())
        logger["id"] = id
        logger["y_pred"] = y_pred.tolist()
        logger["y_test"] = test_labels.flatten().tolist()
        logger["r2_score"] = r2_score(test_labels.flatten(), y_pred.flatten()).astype(
            float
        )
        logger["rmse"] = root_mean_squared_error(
            test_labels.flatten(), y_pred.flatten()
        ).astype(float)
        logger["pearson"] = pearsonr(test_labels.flatten(), y_pred.flatten())[0].astype(
            float
        )
        logger["kendall"] = kendalltau(test_labels.flatten(), y_pred.flatten())[
            0
        ].astype(float)
        logger["n_features"] = n_features
        json.dump(logger, open(f"{fname}/model_{id}_stats.json", "w"), indent=4)

        test_results.append(logger)
        best_n_df = pd.DataFrame.from_dict(best_n_results).T
        best_n_df["nth_feature"] = best_n_df.index
        # save results
        best_n_df.to_csv(f"{fname}/best_nth_feature_{id}.csv")

    pd.DataFrame(test_results).to_pickle(f"{fname}/all_results.pkl")
