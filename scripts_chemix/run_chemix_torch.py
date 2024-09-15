import sys, os
import copy
import json
import torch
from torchinfo import summary

from chemix.model import build_chemix
from chemix.data import load_pickled_dataset
from chemix.train import train
from torchtune.utils.metric_logging import WandBLogger
from omegaconf import OmegaConf


def main(
    config,
    experiment_name,
    wandb_logger = None,
):

    config = copy.deepcopy(config)

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    print(f'Running on: {device}')

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

    summary(model, input_size=(config.batch_size, 43, config.chemix.pom_input.embed_dim, config.chemix.pom_input.num_mix))

    # Save hyper parameters
    with open(f'{root_dir}/hparams_chemix_{experiment_name}.json', 'w') as f:
        f.write(json.dumps(OmegaConf.to_container(config.chemix, resolve=True)))

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
        patience=config.patience,
        experiment_name=experiment_name,
        wandb_logger=wandb_logger,
    )


if __name__ == "__main__":
    conf_file = sys.argv[1]
    config = OmegaConf.load(conf_file)

    experiment_name = "test"
    wandb_logger = WandBLogger(project="Chemix")

    main(
        config=config,
        experiment_name=experiment_name,
        wandb_logger=wandb_logger,
    )