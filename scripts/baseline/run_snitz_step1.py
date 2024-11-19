import sys
import os
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-2])
sys.path.append(str(base_dir / "src/"))

from dataloader.dataloader import DatasetLoader, SplitLoader
from similarity_model.model import AngleSimilarityModel

from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau
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

    fname = "snitz_step1"
    os.makedirs(fname, exist_ok=True)

    dl = DatasetLoader()
    dl.load_dataset("mixtures")
    dl.featurize("mix_rdkit2d")
    dl.reduce('mean')

    # load splits
    sl = SplitLoader("random_cv")
    test_results = []
    for id, train, val, test in sl.load_splits(dl.features, dl.labels):
        select_n_results = {}
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
        logger = {}
        for n in range(2, 200):  # 200
            select_n_results[n] = {}
            rmse_list = []
            for i in range(0, 20000):
                print(f"id={id}, n={n}, i={i}")
                # generate index with n features
                idx_features = np.random.choice(
                    train_features.shape[1], n, replace=False
                )
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
            select_n_results[n]["rmse_avg"] = np.mean(rmse_list)
            select_n_results[n]["rmse_std"] = np.std(rmse_list)
            select_n_results[n]["rmse_avg-std"] = np.mean(rmse_list) - np.std(rmse_list)

        # use best n features for prediction
        n_best = min(
            select_n_results, key=lambda x: select_n_results[x]["rmse_avg-std"]
        )
        idx_features = np.random.choice(train_features.shape[1], n_best, replace=False)
        y_pred = angle_similarity(
            torch.tensor(test_features[:, idx_features, 0]),
            torch.tensor(test_features[:, idx_features, 1]),
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
        logger["n_best"] = n_best
        json.dump(logger, open(f"{fname}/model_{id}_stats.json", "w"), indent=4)
        test_results.append(logger)
        pd.DataFrame(test_results).to_pickle(f"{fname}/all_results.pkl")

        select_n_df = pd.DataFrame.from_dict(select_n_results).T
        select_n_df["n_features"] = select_n_df.index

        # save results
        select_n_df.to_csv(f"{fname}/select_n_features_{id}.csv")
