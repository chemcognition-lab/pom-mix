"""

Example usage:
```
python run_pommix.py
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
import torchmetrics.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Batch
import tqdm
from ml_collections import ConfigDict

from chemix import build_chemix, get_mixture_smiles
from chemix.train import LOSS_MAP
from chemix.utils import TORCH_METRIC_FUNCTIONS
from dataloader import DatasetLoader, SplitLoader
from dataloader.representations.graph_utils import EDGE_DIM, NODE_DIM, from_smiles

# from pom.utils import split_into_batches
from pommix_utils import get_embeddings_from_smiles
from pom.early_stop import EarlyStopping
from pom.gnn.graphnets import GraphNets

parser = ArgumentParser()
parser.add_argument("--run-name", action="store", type=str, default="model", help="Name of run, defaults to `model`.")
parser.add_argument("--split", action="store", type=str, default="random_cv", choices=["random_cv", "ablate_molecules", "ablate_components", "lso_molecules", "random_train_val"])
parser.add_argument("--batch-size", action="store", type=int, default=500, help='Batch size for training.')
parser.add_argument("--augment", action="store_true", default=False, help="Toggle augmenting the training set.")
parser.add_argument("--pom-path", action="store", default=base_dir / "scripts_pom/gs-lf_models/pretrained_pom", 
                    help="Path where POM model parameter and weights are found.")
parser.add_argument("--chemix-path", action="store", default=base_dir / "scripts_chemix/results/pretrained_model", 
                    help="Path where chemix model parameter and weights are found.")
parser.add_argument("--no-verbose", action="store_true", default=False, help='Toggle the verbosity of training. Default False')
parser.add_argument("--no-bias", action="store_true", default=False, help='Turn off the bias in final linear layer. Default False')
parser.add_argument("--mix-lr", action="store", type=float, default=1e-4, help='Learning rate for Chemix.')

FLAGS = parser.parse_args()
np.set_printoptions(precision=3)

if __name__ == '__main__':
    # create folder for results
    fname = Path(f'results/{FLAGS.split}/{FLAGS.run_name}')
    os.makedirs(f'{fname}/', exist_ok=True)

    # store important arguemnts
    pom_path = Path(FLAGS.pom_path)
    chemix_path = Path(FLAGS.chemix_path)
    batch_size = FLAGS.batch_size
    no_bias = FLAGS.no_bias
    augment = FLAGS.augment

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on: {device}')

    # extract hyperparameters and save again in the folder
    hp_mix = ConfigDict(json.load(open(chemix_path / f'hparams_chemix.json', 'r')))
    hp_mix.lr = FLAGS.mix_lr
    with open(fname / 'hparams_chemix.json', 'w') as f:
        f.write(hp_mix.to_json(indent = 4))

    # training set
    dl = DatasetLoader()
    dl.load_dataset('mixtures')
    if not augment:
        dl.featurize('mix_pom_embeddings')
    else:
        dl.featurize('mix_smiles')

    # perform CV split
    sl = SplitLoader(FLAGS.split)
    test_results = []
    for id, train, val, test in sl.load_splits(dl.features, dl.labels):
        # gather the graphs for mixtures
        train_features, train_labels = train
        val_features, val_labels = val
        test_features, test_labels = test

        # perform augmentation
        if augment:
            train_features, train_labels = dl.single_molecule_mixture_gslf_jaccards(train_features, train_labels)
            
            # translate smiles into embeddings on the fly
            with torch.no_grad():
                hp_gnn = ConfigDict(json.load(open(pom_path / 'hparams.json', 'r')))
                embedder = GraphNets(node_dim=NODE_DIM, edge_dim=EDGE_DIM, **hp_gnn)
                embedder.load_state_dict(torch.load(pom_path / 'gnn_embedder.pt'))
                embedder = embedder.to(device)
                embedder.eval()

                graph_list, train_indices = get_mixture_smiles(train_features, from_smiles)
                train_gr = Batch.from_data_list(graph_list).to(device)
                train_features = embedder.graphs_to_mixtures(train_gr, train_indices, device=device).detach().cpu().numpy()

                graph_list, val_indices = get_mixture_smiles(val_features, from_smiles)
                val_gr = Batch.from_data_list(graph_list).to(device)
                val_features = embedder.graphs_to_mixtures(val_gr, val_indices, device=device).detach().cpu().numpy()

                graph_list, test_indices = get_mixture_smiles(test_features, from_smiles)
                test_gr = Batch.from_data_list(graph_list).to(device)
                test_features = embedder.graphs_to_mixtures(test_gr, test_indices, device=device).detach().cpu().numpy()

        train_features = torch.tensor(train_features, dtype=torch.float32).to(device)
        y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)
        train_loader = DataLoader(TensorDataset(train_features, y_train), batch_size=batch_size, shuffle=True)
    
        val_features = torch.tensor(val_features, dtype=torch.float32).to(device)
        y_val = torch.tensor(val_labels, dtype=torch.float32).to(device)

        test_features = torch.tensor(test_features, dtype=torch.float32).to(device)
        y_test = torch.tensor(test_labels, dtype=torch.float32).to(device)

        print(f'Running split: {id}')
        print(f'Training set: {len(y_train)}')
        print(f'Validation set: {len(y_val)}')
        print(f'Testing set: {len(y_test)}')

        # load models
        # create the chemix model
        hp_mix.chemix.regressor.no_bias = no_bias
        chemix = build_chemix(config=hp_mix.chemix)
        chemix = chemix.to(device)

        # training params
        loss_fn = LOSS_MAP[hp_mix.loss_type]()
        metric_fn = F.pearson_corrcoef
        optimizer = torch.optim.Adam([{'params': chemix.parameters(), 'lr': hp_mix.lr}])
        num_epochs = 5000
        es = EarlyStopping(chemix, patience=1000, mode='maximize')

        # start training loop
        log = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'val_metric']}
        pbar = tqdm.tqdm(range(num_epochs), disable=FLAGS.no_verbose)
        for epoch in pbar:
            
            running_loss = 0
            for batch in train_loader:
                x, y = batch
                optimizer.zero_grad()
                y_pred = chemix(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss

            train_loss = running_loss.detach().cpu().item()
        
            
            # validation + early stopping
            chemix.eval()
            with torch.no_grad():
                y_pred = chemix(val_features)
                loss = loss_fn(y_pred, y_val)
                metric = metric_fn(y_pred.flatten(), y_val.flatten())
                val_loss = loss.detach().cpu().item()
                val_metric = metric.detach().cpu().item()

            log['epoch'].append(epoch)
            log['train_loss'].append(train_loss)
            log['val_loss'].append(val_loss)
            log['val_metric'].append(val_metric)

            pbar.set_description(f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val metric: {val_metric:.4f}")

            stop = es.check_criteria(val_metric, chemix)
            if stop:
                print(f'Early stop reached at {es.best_step} with {es.best_value}')
                break
        log = pd.DataFrame(log)

        # save model weights
        best_model_dict = es.restore_best()
        chemix.load_state_dict(best_model_dict)      # load the best one trained
        torch.save(chemix.state_dict(), fname / f'{id}_chemix.pt')


        ##### TESTING #####
        # save the results in a file
        chemix.eval()
        with torch.no_grad():
            y_pred = chemix(test_features)
    
        # calculate a bunch of metrics on the results to compare
        test_metrics = {}
        for name, func in TORCH_METRIC_FUNCTIONS.items():
            test_metrics[name] = func(y_pred.flatten(), y_test.flatten()).detach().cpu().item()
        print(test_metrics)
        test_metrics = pd.DataFrame(test_metrics, index=['metrics']).transpose()
        test_metrics.to_csv(fname / f'{id}_test_metrics.csv', index=False)

        y_pred = y_pred.detach().cpu().numpy().flatten()
        y_test = y_test.detach().cpu().numpy().flatten()
        test_predictions = pd.DataFrame({
            'Predicted_Experimental_Values': y_pred, 
            'Ground_Truth': y_test,
            'MAE': np.abs(y_pred - y_test),
        }, index=range(len(y_pred)))
        test_predictions.to_csv(fname / f'{id}_test_predictions.csv', index=False)

        # plot the predictions
        ax = sns.scatterplot(data=test_predictions, x='Ground_Truth', y='Predicted_Experimental_Values')
        ax.plot([0,1], [0,1], 'r--')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.annotate(''.join(f'{k}: {v["metrics"]:.4f}\n' for k, v in test_metrics.iterrows()).strip(),
                xy=(0.05,0.8), xycoords='axes fraction',
                # textcoords='offset points',
                size=12,
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"))
        plt.savefig(fname / f'{id}_test_predictions.png', bbox_inches='tight')
        plt.close()




