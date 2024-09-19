import sys, os
import copy
import json
import torch
import numpy as np
import pandas as pd
from torchinfo import summary

sys.path.append('/u/ctser/pom-mix/src')

from chemix.model import build_chemix
from chemix.predict import predict
from chemix.data import dataset_to_torch
from dataloader import DatasetLoader
from omegaconf import OmegaConf


def main(
    config,
    checkpoint_path,
):

    config = copy.deepcopy(config)

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    print(f'Running on: {device}')

    root_dir = config.root_dir
    os.makedirs(root_dir, exist_ok=True)

    # Data
    dl = DatasetLoader()
    dl.load_dataset("mixtures")
    dl.featurize("mix_pom_embeddings")

    print(dl.features.shape)
    # Augmentation
    X_1 = np.stack((dl.features[:,:,:,0], dl.features[:,:,:,0]), axis=-1)
    X_2 = np.stack((dl.features[:,:,:,1], dl.features[:,:,:,1]), axis=-1)

    X = np.concatenate((X_1, X_2))
    print(X.shape)
    y = np.zeros((len(X),1))

    

    _, test_loader = dataset_to_torch(
        X=X,
        y=y,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )

    # Model
    model = build_chemix(config=config.chemix).to(device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
    import pdb; pdb.set_trace()
    preds = predict(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    df = pd.DataFrame.from_dict({"true_values": y.flatten().tolist(), "predictions": preds.flatten().tolist()})

    df.to_csv("/u/rajao/self_mix_result.csv", index=False)

if __name__ == "__main__":
    run_name = sys.argv[1]
    checkpoint_path = f"../scripts_mix/results/chemix_pearson/top1/best_model_dict_{run_name}.pt"
    conf_file = f"../scripts_mix/results/chemix_pearson/top1/hparams_chemix_{run_name}.json"
    config = OmegaConf.load(conf_file)

    main(
        config=config,
        checkpoint_path=checkpoint_path,
    )