import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-1])
sys.path.append(str(base_dir / "src/"))

# basic dependencies
from ml_collections import ConfigDict
import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

import torch.nn as nn

from functools import partial

# dataloader
from dataloader import DatasetLoader

# pom
import pom.utils as utils
from pom import EndToEndModule
from pom.data import GraphDataset
from pom.early_stop import EarlyStopping
from pom.gnn.graphnets import GraphNets
from pom.gnn.graph_utils import get_graph_degree_histogram, get_pooling_function

# glm
from prediction_head.data import TaskType, get_activation, get_loss_fn
from prediction_head.GLM import GLM

# optuna for tuning
import optuna

import torch
import torchinfo
from torch_geometric.loader import DataLoader as pygdl


def get_split_file(dataset_name, test_size):
    os.makedirs(f"{dataset_name}_models/", exist_ok=True)
    fname = f"{dataset_name}_models/{dataset_name}_{test_size}.npz"

    if os.path.isfile(fname):
        split = np.load(fname)
        train_ind, test_ind = split["train_ind"], split["test_ind"]
    else:
        dl = DatasetLoader()
        dl.load_dataset(dataset_name)
        num_dat = dl.get_dataset_specifications(dataset_name)["n_datapoints"]
        labels = dl.labels
        train_ind, _, test_ind, _ = iterative_train_test_split(
            np.array(range(num_dat)).reshape(-1, 1), labels, test_size=test_size
        )
        np.savez(fname, train_ind=train_ind, test_ind=test_ind)
    return train_ind, test_ind


def objective(trial, ds, train_loader, test_loader, task, task_dim):
    # hparams settings
    hp = ConfigDict(
        {"gnn": ConfigDict(), "head": ConfigDict(), "training": ConfigDict()}
    )

    # using graphnets
    hp.gnn.global_dim = 196  # trial.suggest_int('global_dim', 64, 320, step=64) # List [64, 128, 192, 256, 320]
    hp.gnn.depth = trial.suggest_int("depth", 2, 4)
    hp.gnn.hidden_dim = trial.suggest_int(
        "hidden_dim", 64, 320, step=64
    )  # List [64, 128, 192, 256, 320]
    # hp.gnn.num_layers = trial.suggest_int('num_layers', 1, 3) # set to 1 for FiLM and Attn and 2 for all other models
    hp.gnn.dropout = trial.suggest_float("dropout", 0, 0.5, step=0.05)
    hp.training.lr = trial.suggest_categorical(
        "learning_rate", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    )  # [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    hp.training.num_epochs = 1000  # early stopping

    seed = 42
    utils.set_seed(seed)
    dname = "gs-lf"

    # Load all models and datasets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn = GraphNets(
        ds.node_dim,
        ds.edge_dim,
        hp.gnn.global_dim,
        hidden_dim=hp.gnn.hidden_dim,
        depth=hp.gnn.depth,
        dropout=hp.gnn.dropout,
    ).to(device)
    # pred = nn.Sequential(nn.Linear(hp.gnn.global_dim, task_dim)).to(device)
    pred = GLM(input_dim=hp.gnn.global_dim, output_dim=task_dim, tasktype=task).to(
        device
    )
    model = EndToEndModule(gnn, pred).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.training.lr)

    # optimization things
    es = EarlyStopping(
        gnn, patience=50, mode="maximize"
    )  # early stopping only GNN weights
    log = {k: [] for k in ["epoch", "train_loss", "val_loss", "val_metric", "dataset"]}
    pbar = tqdm.tqdm(range(hp.training.num_epochs))

    loss_fn = get_loss_fn(task)()
    metric_fn = utils.get_metric_function(task)

    for epoch in pbar:
        # training loop
        log["epoch"].append(epoch)
        log["dataset"].append(dname)
        training_loss = 0
        model.train()
        for batch in train_loader:
            data, y = batch
            data, y = data.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model(data)
            loss = loss_fn(y_hat.squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        training_loss /= len(train_loader)
        log["train_loss"].append(training_loss)

        # validation loop
        y_pred, y_true = [], []
        with torch.no_grad():
            model.eval()
            for batch in test_loader:
                data, y = batch
                data = data.to(device)

                y_hat = model(data)
                y_pred.append(y_hat.detach().cpu())
                y_true.append(y)

            y_pred = torch.concat(y_pred)
            y_true = torch.concat(y_true)
            testing_metric = metric_fn(y_pred.squeeze(), y_true.squeeze()).item()
            testing_loss = loss_fn(y_pred.squeeze(), y_true.squeeze()).item()

        log["val_loss"].append(testing_loss)
        log["val_metric"].append(testing_metric)

        # print some statistics
        pbar.set_description(
            f"Train: {training_loss:.4f} | Test: {testing_loss:.4f} | Test metric: {testing_metric:.4f} | Dataset: {dname}"
        )

        # check early stopping based on losses averaged of datasets
        stop = es.check_criteria(testing_metric, model.gnn_embedder)
        if stop:
            print(f"Early stop reached at {es.best_step} with loss {es.best_value}")
            break
    log = pd.DataFrame(log)
    return es.best_value


if __name__ == "__main__":
    os.makedirs("optuna/", exist_ok=True)

    dname = "gs-lf"
    dl = DatasetLoader()
    dl.load_dataset(dname)
    dl.featurize("molecular_graphs", init_globals=True)
    data_specs = dl.get_dataset_specifications(dname)
    task = data_specs["task"]
    task_dim = data_specs["task_dim"]

    print(f"Running on CUDA: {torch.cuda.is_available()}")

    train_ind, test_ind = get_split_file(dname, test_size=0.2)
    dataset = GraphDataset(dl.features, dl.labels)

    # split the data and get dataloaders
    train_set = torch.utils.data.Subset(dataset, train_ind.flatten())
    test_set = torch.utils.data.Subset(dataset, test_ind.flatten())
    train_loader = pygdl(train_set, batch_size=64, shuffle=True)
    test_loader = pygdl(test_set, batch_size=128, shuffle=False)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        partial(
            objective,
            ds=dataset,
            train_loader=train_loader,
            test_loader=test_loader,
            task=task,
            task_dim=task_dim,
        ),
        n_trials=200,
    )

    print("###########################################")
    print(f"Best achieved score: {study.best_trial.value}")
    print()
    print(study.best_params)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("optuna/optuna_history.png")

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image("optuna/optuna_coordinates.png")

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("optuna/optuna_importance.png")

    fig = optuna.visualization.plot_contour(study)
    fig.write_image("optuna/optuna_contour.png")

    ### Get top trials
    all_opts = pd.DataFrame(
        {t.number: {"params": t.params, "fitness": t.value} for t in study.get_trials()}
    ).transpose()
    all_opts["fitness"] = all_opts["fitness"].astype(float)

    all_opts.to_csv("optuna/trials_all.csv")
    all_opts.nlargest(10, "fitness").to_csv("optuna/trials_top10.csv")
