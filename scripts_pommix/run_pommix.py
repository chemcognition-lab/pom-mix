"""

Example usage:
```
python run_chemix.py --test_run
```
"""
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-1])
sys.path.append( str(base_dir / 'src/') )

import json
from argparse import ArgumentParser
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchinfo
import torchmetrics.functional as F
import tqdm
from ml_collections import ConfigDict
from torch_geometric.data import Batch

from chemix import build_chemix, get_mixture_smiles
from chemix.train import LOSS_MAP
from chemix.utils import TORCH_METRIC_FUNCTIONS
from dataloader import DatasetLoader, SplitLoader
from dataloader.representations.graph_utils import EDGE_DIM, NODE_DIM, from_smiles

from pom.utils import split_into_batches
from pom.early_stop import EarlyStopping
from pom.gnn.graphnets import GraphNets

parser = ArgumentParser()
parser.add_argument("--trial", action="store", type=int, default=1, help="Trial number.")
parser.add_argument("--no-verbose", action="store_true", default=False, help='Toggle the verbosity of training. Default False')
parser.add_argument("--gnn-lr", action="store", type=float, default=1e-5, help='Learning rate for GNN POM embedder. Default 1e-5.')
parser.add_argument("--mix-lr", action="store", type=float, default=5e-4, help='Learning rate for Chemix. Default 5e-4.')
parser.add_argument("--gnn-freeze", action="store_true", default=False, help='Toggle freeze GNN POM weights. Default False')
parser.add_argument("--test_run", action="store_true", default=False, help='Save test results. Default False')

FLAGS = parser.parse_args()
np.set_printoptions(precision=3)

if __name__ == '__main__':
    # create folder for results
    fname = f'results/chemix_pearson/top{FLAGS.trial}'
    os.makedirs(f'{fname}/', exist_ok=True)

    # path where the pretrained models are stored
    embedder_path = f'../scripts_pom/gs-lf_models/pretrained_pom/'
    chemix_path = f'../scripts_chemix/results/chemix_pearson/top1'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on: {device}')

    # extract hyperparameters and save again in the folder
    hp_gnn = ConfigDict(json.load(open(f'{embedder_path}/hparams.json', 'r')))
    hp_gnn.lr = FLAGS.gnn_lr
    hp_gnn.freeze = FLAGS.gnn_freeze
    with open(f'{fname}/hparams_graphnets.json', 'w') as f:
        f.write(hp_gnn.to_json(indent = 4))
    
    # hp_mix = ConfigDict(json.load(open(f'{chemix_path}/hparams_chemix_{FLAGS.trial}.json', 'r')))
    hp_mix = ConfigDict(json.load(open(f'{chemix_path}/hparams_chemix_bias.json', 'r')))
    hp_mix.lr = FLAGS.mix_lr
    with open(f'{fname}/hparams_chemix.json', 'w') as f:
        f.write(hp_mix.to_json(indent = 4))

    # training set
    dl = DatasetLoader()
    dl.load_dataset('mixtures')
    dl.featurize('mix_smiles')

    # perform CV split
    sl = SplitLoader("random_cv")
    test_results = []
    for id, train, val, test in sl.load_splits(dl.features, dl.labels):
        # gather the graphs for mixtures
        train_features, train_labels = train
        graph_list, train_indices = get_mixture_smiles(train_features, from_smiles)
        train_gr = Batch.from_data_list(graph_list).to(device)
        y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)

        val_features, val_labels = val
        graph_list, val_indices = get_mixture_smiles(val_features, from_smiles)
        val_gr = Batch.from_data_list(graph_list).to(device)
        y_val = torch.tensor(val_labels, dtype=torch.float32).to(device)

        test_features, test_labels = test
        graph_list, test_indices = get_mixture_smiles(test_features, from_smiles)
        test_gr = Batch.from_data_list(graph_list).to(device)
        y_test = torch.tensor(test_labels, dtype=torch.float32).to(device)

        print(f'Running split: {id}')
        print(f'Training set: {len(y_train)}')
        print(f'Validation set: {len(y_val)}')
        print(f'Testing set: {len(y_test)}')

        # load models
        # create the pom embedder model and load weights
        embedder = GraphNets(node_dim=NODE_DIM, edge_dim=EDGE_DIM, **hp_gnn)
        embedder.load_state_dict(torch.load(f'{embedder_path}/gnn_embedder.pt'))
        embedder = embedder.to(device)
        if hp_gnn.freeze:                   # freeze pom if specified
            for p in embedder.parameters():
                p.requires_grad = False

        # create the chemix model and load weights
        chemix = build_chemix(config=hp_mix.chemix)
        # chemix.load_state_dict(torch.load(f'{chemix_path}/best_model_dict_{FLAGS.trial}.pt', map_location=device))
        chemix.load_state_dict(torch.load(f'{chemix_path}/best_model_dict_bias.pt', map_location=device))
        chemix = chemix.to(device=device)

        # training params
        loss_fn = LOSS_MAP[hp_mix.loss_type]()
        metric_fn = F.pearson_corrcoef
        optimizer = torch.optim.Adam(
            [
                {'params': embedder.parameters(), 'lr': hp_gnn.lr},
                {'params': chemix.parameters(), 'lr': hp_mix.lr}
            ])
        num_epochs = 5000 if not FLAGS.test_run else 10
        es = EarlyStopping(nn.ModuleList([embedder, chemix]), patience=1000, mode='maximize')

        # start training loop
        log = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'val_metric']}
        pbar = tqdm.tqdm(range(num_epochs), disable=FLAGS.no_verbose)
        for epoch in pbar:
            embedder.train(); chemix.train()
            if hp_gnn.freeze:
                embedder.eval()

            optimizer.zero_grad()
            out = embedder.graphs_to_mixtures(train_gr, train_indices, device=device)
            y_pred = chemix(out)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

            train_loss = loss.detach().cpu().item()

            # X_batches, Y_batches = split_into_batches(out, y_train, q=32)
            # total_loss = 0
            # optimizer.zero_grad()
            # for b, t in zip(X_batches, Y_batches):
            #     y_pred = chemix(b)
            #     loss = loss_fn(y_pred, t)
            #     total_loss = total_loss + loss
            # total_loss /= len(Y_batches)
            # total_loss.backward()
            # optimizer.step()
            # train_loss = total_loss.detach().cpu().item()
            
            # validation + early stopping
            embedder.eval(); chemix.eval()
            with torch.no_grad():
                out = embedder.graphs_to_mixtures(val_gr, val_indices, device=device)
                y_pred = chemix(out)
                loss = loss_fn(y_pred, y_val)
                metric = metric_fn(y_pred.flatten(), y_val.flatten())
                val_loss = loss.detach().cpu().item()
                val_metric = metric.detach().cpu().item()

            log['epoch'].append(epoch)
            log['train_loss'].append(train_loss)
            log['val_loss'].append(val_loss)
            log['val_metric'].append(val_metric)

            pbar.set_description(f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val metric: {val_metric:.4f}")

            stop = es.check_criteria(val_metric, nn.ModuleList([embedder, chemix]))
            if stop:
                print(f'Early stop reached at {es.best_step} with {es.best_value}')
                break
        log = pd.DataFrame(log)

        # save model weights
        best_model_dict = es.restore_best()
        model = nn.ModuleList([embedder, chemix])
        model.load_state_dict(best_model_dict)      # load the best one trained
        if not FLAGS.test_run:
            torch.save(model[0].state_dict(), f'{fname}/{id}_gnn_embedder.pt')
            torch.save(model[1].state_dict(), f'{fname}/{id}_chemix.pt')


        ##### TESTING #####
        # save the results in a file
        embedder.eval(); chemix.eval()
        with torch.no_grad():
            out = embedder.graphs_to_mixtures(test_gr, test_indices, device=device)
            y_pred = chemix(out)
    
        # calculate a bunch of metrics on the results to compare
        leaderboard_metrics = {}
        for name, func in TORCH_METRIC_FUNCTIONS.items():
            leaderboard_metrics[name] = func(y_pred.flatten(), y_test.flatten()).detach().cpu().item()
        print(leaderboard_metrics)
        leaderboard_metrics = pd.DataFrame(leaderboard_metrics, index=['metrics']).transpose()
        if not FLAGS.test_run:
            leaderboard_metrics.to_csv(f'{fname}/{id}_test_metrics.csv')

        y_pred = y_pred.detach().cpu().numpy().flatten()
        y_test = y_test.detach().cpu().numpy().flatten()
        leaderboard_predictions = pd.DataFrame({
            'Predicted_Experimental_Values': y_pred, 
            'Ground_Truth': y_test,
            'MAE': np.abs(y_pred - y_test),
        }, index=range(len(y_pred)))
        if not FLAGS.test_run:
            leaderboard_predictions.to_csv(f'{fname}/{id}_test_predictions.csv')

        # plot the predictions
        ax = sns.scatterplot(data=leaderboard_predictions, x='Ground_Truth', y='Predicted_Experimental_Values')
        ax.plot([0,1], [0,1], 'r--')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.annotate(''.join(f'{k}: {v["metrics"]:.4f}\n' for k, v in leaderboard_metrics.iterrows()).strip(),
                xy=(0.05,0.7), xycoords='axes fraction',
                # textcoords='offset points',
                size=12,
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"))
        if not FLAGS.test_run:
            plt.savefig(f'{fname}/{id}_test_predictions.png', bbox_inches='tight')
        plt.close()




