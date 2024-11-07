import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-2])
sys.path.append(str(base_dir / "src/"))

# basic dependencies
from ml_collections import ConfigDict
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

# dataloader
from dataloader import DatasetLoader

# pom
import pom.utils as utils
from pom import EndToEndModule
from pom.data import GraphDataset
from pom.early_stop import EarlyStopping
from pom.gnn.graphnets import GraphNets

# glm
from prediction_head.data import TaskType, get_loss_fn
from prediction_head.GLM import GLMStructured, TaskSpec

import torch
from torch_geometric.loader import DataLoader as pygdl

from argparse import ArgumentParser


def get_split_file(dataset_name, test_size):
    os.makedirs(f"{dataset_name}_models/", exist_ok=True)
    fname = f"{dataset_name}_models/{dataset_name}_{test_size}.npz"

    if os.path.isfile(fname):
        split = np.load(fname)
        train_ind, test_ind = split["train_ind"], split["test_ind"]
    else:
        dl = DatasetLoader()
        dl.load_dataset(dataset_name)
        labels = dl.labels
        num_dat = len(labels)
        train_ind, _, test_ind, _ = iterative_train_test_split(
            np.array(range(num_dat)).reshape(-1, 1), labels, test_size=test_size
        )
        np.savez(fname, train_ind=train_ind, test_ind=test_ind)
    return train_ind, test_ind


parser = ArgumentParser()
parser.add_argument(
    "--depth", action="store", type=int, default=3, help="Depth of GNN."
)
parser.add_argument(
    "--hidden_dim", action="store", type=int, default=128, help="Hidden dimension."
)
parser.add_argument(
    "--dropout", action="store", type=float, default=0.15, help="Dropout rate."
)
parser.add_argument(
    "--lr", action="store", type=float, default=1e-4, help="Learning rate."
)
parser.add_argument(
    "--tag", action="store", type=str, help="Name of directory to save model."
)
FLAGS = parser.parse_args()


if __name__ == "__main__":
    # create folders for logging
    fname = f"gs-lf_models/{FLAGS.tag}"  # create a file name to log all outputs
    os.makedirs(fname, exist_ok=True)

    # hparams settings
    hp = ConfigDict()

    # using graphnets parameters from hparam opt
    hp.global_dim = 196  # this is also the embedding space

    ######
    hp.depth = FLAGS.depth
    hp.hidden_dim = FLAGS.hidden_dim
    hp.dropout = round(FLAGS.dropout, 2)
    hp.lr = FLAGS.lr
    ######

    hp.num_epochs = 2000
    hp.batch_size = 64
    hp.val_size = 0.2
    with open(f"{fname}/hparams.json", "w") as f:
        f.write(hp.to_json(indent=4))

    # get dataset names
    # seed = 42
    # utils.set_seed(seed)
    # data_names = DatasetLoader().get_dataset_names()
    # data_names.remove("keller_2016")

    # Load all models and datasets
    print(f"Using GPU: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = {}
    data_store = {}
    dname = "gs-lf"

    print(f"Loading {dname}...")
    dl = DatasetLoader()
    dl.load_dataset(dname)
    dl.featurize("molecular_graphs", init_globals=True)
    data_specs = dl.get_dataset_specifications(dname)

    # Dataset specifics
    task = TaskType(data_specs["task"])
    task_dim = data_specs["task_dim"]
    task_spec = TaskSpec(task_dim, task)
    specs[dname] = task_spec

    train_ind, test_ind = get_split_file(dname, test_size=0.2)
    dataset = GraphDataset(dl.features, dl.labels)

    # split the data and get dataloaders
    train_set = torch.utils.data.Subset(dataset, train_ind.flatten())
    test_set = torch.utils.data.Subset(dataset, test_ind.flatten())
    train_loader = pygdl(train_set, batch_size=hp.batch_size, shuffle=True)
    test_loader = pygdl(test_set, batch_size=128, shuffle=False)

    # use the same optimizer
    data_store.update(
        {
            dname: {
                "train_loader": train_loader,
                "test_loader": test_loader,
                "loss_fn": get_loss_fn(task)(),
                "metric_fn": utils.get_metric_function(task),
                "task": task,
            }
        }
    )

    # create model
    gnn = GraphNets(
        dataset.node_dim,
        dataset.edge_dim,
        hp.global_dim,
        hidden_dim=hp.hidden_dim,
        depth=hp.depth,
        dropout=hp.dropout,
    ).to(device)
    pred = GLMStructured(input_dim=hp.global_dim, tasks=specs).to(device)
    model = EndToEndModule(gnn, pred).to(device)
    # add uncertainty weights, which are optimized with the model
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    # log_var_weights = torch.nn.Parameter(torch.zeros(len(data_store), dtype=torch.float32), requires_grad=True)
    # optimizer = torch.optim.Adam(list(model.parameters()) + [log_var_weights], lr=hp.lr)

    # optimization things
    es = EarlyStopping(
        gnn, patience=200, mode="maximize"
    )  # early stopping only GNN weights
    log = {k: [] for k in ["epoch", "train_loss", "val_loss", "val_metric", "dataset"]}

    pbar = tqdm.tqdm(range(hp.num_epochs))
    for epoch in pbar:
        avg_train_loss, avg_test_loss, avg_test_metric = 0, 0, 0
        for di, (dname, store) in enumerate(data_store.items()):
            train_loader = store["train_loader"]
            test_loader = store["test_loader"]
            loss_fn = store["loss_fn"]
            metric_fn = store["metric_fn"]

            # training loop
            log["epoch"].append(epoch)
            log["dataset"].append(dname)
            training_loss = 0
            model.train()
            for batch in train_loader:
                data, y = batch
                data, y = data.to(device), y.to(device)

                optimizer.zero_grad()
                y_hat = model(data, dname)
                loss = loss_fn(y_hat.squeeze(), y.squeeze())
                # scale the loss
                # loss = torch.exp(-log_var_weights[di]) * loss + log_var_weights[di]
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
            training_loss /= len(train_loader)
            avg_train_loss += training_loss
            log["train_loss"].append(training_loss)

            # validation loop
            y_pred, y_true = [], []
            with torch.no_grad():
                model.eval()
                for batch in test_loader:
                    data, y = batch
                    data = data.to(device)

                    y_hat = model(data, dname)
                    y_pred.append(y_hat.detach().cpu())
                    y_true.append(y)

                y_pred = torch.concat(y_pred)
                y_true = torch.concat(y_true)
                testing_metric = metric_fn(y_pred.squeeze(), y_true.squeeze())
                # testing_metric *= torch.exp(-log_var_weights[di])
                testing_metric = testing_metric.item()

                testing_loss = loss_fn(y_pred.squeeze(), y_true.squeeze())
                # testing_loss *= torch.exp(-log_var_weights[di])
                testing_loss = testing_loss.item()

            avg_test_loss += testing_loss
            avg_test_metric += testing_metric
            log["val_loss"].append(testing_loss)
            log["val_metric"].append(testing_metric)

            # print some statistics
            pbar.set_description(
                f"Train: {training_loss:.4f} | Test: {testing_loss:.4f} | Test metric: {testing_metric:.4f} | Dataset: {dname}"
            )

        # check early stopping based on losses averaged of datasets
        avg_train_loss /= len(data_store)
        avg_test_loss /= len(data_store)
        avg_test_metric /= len(data_store)
        log["epoch"].append(epoch)
        log["train_loss"].append(avg_train_loss)
        log["val_loss"].append(avg_test_loss)
        log["val_metric"].append(avg_test_metric)
        log["dataset"].append("average")

        stop = es.check_criteria(avg_test_metric, model.gnn_embedder)
        if stop:
            print(f"Early stop reached at {es.best_step} with loss {es.best_value}")
            break

    log = pd.DataFrame(log)

    # save model weights
    best_model_dict = es.restore_best()
    model.gnn_embedder.load_state_dict(best_model_dict)  # load the best one trained
    torch.save(model.gnn_embedder.state_dict(), f"{fname}/gnn_embedder.pt")

    # also plot and save the training loss (for diagnostics)
    log.to_csv(f"{fname}/training.csv", index=False)
    plt_log = log[["epoch", "val_metric", "dataset"]].melt(
        id_vars=["epoch", "dataset"], var_name="set", value_name="metric"
    )
    ax = sns.lineplot(
        data=plt_log[plt_log["dataset"] != "average"],
        x="epoch",
        y="metric",
        hue="dataset",
        palette="colorblind",
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.plot(
        plt_log[plt_log["dataset"] == "average"]["epoch"],
        plt_log[plt_log["dataset"] == "average"]["metric"],
        "k-",
        linewidth=2,
    )
    plt.ylim([-0.1, 1.0])
    plt.savefig(f"{fname}/metric.png", bbox_inches="tight")
    plt.close()

    plt_log = log[["epoch", "train_loss", "val_loss", "dataset"]].melt(
        id_vars=["epoch", "dataset"], var_name="set", value_name="loss"
    )
    sns.lineplot(
        data=plt_log,
        x="epoch",
        y="loss",
        style="set",
        hue="dataset",
        palette="colorblind",
    )
    plt.savefig(f"{fname}/loss.png", bbox_inches="tight")
    plt.close()
