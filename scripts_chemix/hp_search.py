import sys, os
import json
import torch
import wandb
import copy
import datetime

from chemix.model import build_chemix
from chemix.data import load_pickled_dataset
from chemix.train import train
from omegaconf import OmegaConf

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
    _, train_loader = load_pickled_dataset(
        os.path.join(config.data.data_path, config.data.train_data_folder),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )

    _, val_loader = load_pickled_dataset(
        os.path.join(config.data.data_path, config.data.val_data_folder),
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