import torch
from torch import nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import numpy as np
from typing import List

from chemix.model import build_chemix

def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
):
    model.eval()

    all_predictions = []
    with torch.no_grad():
        for batch in data_loader:
            features, _ = batch
            predictions = list(model(features.to(device)).cpu().numpy())
            for prediction in predictions:
                all_predictions.append(prediction)
    return np.stack(all_predictions)


def ensemble_predict(
    checkpoint_path_list: List[str],
    config_path_list: List[str],
    data_loader: DataLoader,
    device: str,
):
    ensemble_preds = []
    for checkpoint_path, config_path in zip(checkpoint_path_list, config_path_list):
        config = OmegaConf.load(config_path)
        model = build_chemix(config.chemix)
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))

        preds = predict(model, data_loader, device)
        ensemble_preds.append(preds)

    return ensemble_preds