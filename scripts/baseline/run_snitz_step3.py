import sys
import os
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-1])
sys.path.append(str(base_dir / "src/"))

from dataloader.dataloader import DatasetLoader, SplitLoader

from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau, zscore
import torch

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

    fname = "snitz_step3"
    os.makedirs(fname, exist_ok=True)

    dl = DatasetLoader()
    dl.load_dataset("mixtures")
    dl.featurize("mix_rdkit2d_mean")

    # load splits
    sl = SplitLoader("random_cv")
    test_results = []
    for id, train, val, test in sl.load_splits(dl.features, dl.labels):
        train_features, train_labels = train
        val_features, val_labels = val
        test_features, test_labels = test
        train_features = train_features.astype(float)
        val_features = val_features.astype(float)
        test_features = test_features.astype(float)
        train_labels = train_labels.astype(float)
        val_labels = val_labels.astype(float)
        test_labels = test_labels.astype(float)

        # select n best features
        best_n_features = pd.read_csv(f"snitz_step2/best_nth_feature_{id}.csv")
        best_n_zscore = -zscore(best_n_features["rmse_avg"])
        # set all negative values to 0
        best_n_zscore = [max(0, x) for x in best_n_zscore]
        # get index of positive features
        positive_features = [i for i, x in enumerate(best_n_zscore) if x > 0]
        # num of features to select
        with open(f"./snitz_step1/model_{id}_stats.json") as json_file:
            select_n = json.load(json_file)
        n_features = select_n["n_best"]
        rmse_min = 1e7
        for i in range(1, 4000):
            print(f"i={i}")
            idx_best_features = np.random.choice(
                positive_features, n_features, replace=False
            )
            # similarity model between mixtures
            y_train = angle_similarity(
                torch.tensor(train_features[:, idx_best_features, 0]),
                torch.tensor(train_features[:, idx_best_features, 1]),
            )
            y_train = y_train.detach().numpy()
            rmse = root_mean_squared_error(train_labels.flatten(), y_train.flatten())
            if rmse < rmse_min:
                rmse_min = rmse
                idx_best_features_min = idx_best_features

        logger = {"pearson": -np.nan}
        y_pred = angle_similarity(
            torch.tensor(test_features[:, idx_best_features_min, 0]),
            torch.tensor(test_features[:, idx_best_features_min, 1]),
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
        logger["best_features"] = idx_best_features_min.tolist()
        json.dump(logger, open(f"{fname}/model_{id}_stats.json", "w"), indent=4)

        test_results.append(logger)
    pd.DataFrame(test_results).to_pickle(f"{fname}/all_results.pkl")
