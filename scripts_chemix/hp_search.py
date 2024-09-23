import sys, os
import json
import torch
import wandb
import copy
import numpy as np
import datetime

from chemix.model import build_chemix
from chemix.data import dataset_to_torch
from chemix.train import train
from dataloader import DatasetLoader
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split


import collections.abc


def update_config(config, wandb_config):
    config = copy.deepcopy(config)

    for k, v in wandb_config.items():
        if isinstance(v, collections.abc.Mapping):
            config[k] = update_config(config.get(k, {}), v)
        else:
            config[k] = v
    return config

def main(config):
    run = wandb.init()

    config = update_config(config=config, wandb_config=wandb.config)

    config.seed = int(round(datetime.datetime.now().timestamp())) % 100

    torch.manual_seed(config.seed)
    print(f'Running on seed: {config.seed}')

    device = torch.device(config.device)
    print(f'Running on: {device}')

    if config.device == "cuda":
        torch.cuda.manual_seed(config.seed)

    root_dir = config.root_dir
    os.makedirs(root_dir, exist_ok=True)

    # Data
    dl = DatasetLoader()
    dl.load_dataset("mixtures")

    train_idx, test_idx = train_test_split(np.arange(len(dl.features)), test_size=0.2, random_state=0, stratify=dl.dataset_id)

    dl.featurize("mix_pom_embeddings")

    _, train_loader = dataset_to_torch(
        X=dl.features[train_idx],
        y=dl.labels[train_idx],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )

    _, val_loader = dataset_to_torch(
        X=dl.features[test_idx],
        y=dl.labels[test_idx],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Model
    model = build_chemix(config=config.chemix).to(device=device)

    # Save hyper parameters
    # TODO: include LR in dict
    with open(f'{root_dir}/hparams_chemix_{run.name}.json', 'w') as f:
        f.write(json.dumps(OmegaConf.to_container(config, resolve=True)))

    # Training
    train(
        root_dir=root_dir,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader, 
        loss_type=config.loss_type,
        lr=config.lr,
        device=device,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        experiment_name=run.name,
        patience=config.patience,
        wandb_logger=wandb,
    )


if __name__ == "__main__":
    conf_file = sys.argv[1]
    config = OmegaConf.load(conf_file)
    main(config=config)