from __future__ import annotations

# sets the path to the root of the repository
from typing import Optional
from pathlib import Path
import sys

from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder

root_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(root_path))
from prediction_head.data import (
    TaskType,
    TaskSpec,
    get_activation,
    get_loss_fn_dict,
    get_metrics_dict,
    get_loss_fn_dict,
    get_mask,
    get_loss_sum,
    get_optimizer_dict,
    use_loss_fn_dict,
)
from torch import nn
import numpy as np
import torch
import tqdm
import functools
import pandas as pd


class GLM(nn.Module):
    "Generalized Linear Model for basic predictive modelling."

    def __init__(self, input_dim: int, output_dim: int, tasktype: TaskType):
        super(GLM, self).__init__()
        self.tasktype = tasktype
        self.activation = get_activation(tasktype)()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.activation(self.linear(x))

    @classmethod
    def from_spec(cls, input_dim: int, spec: TaskSpec) -> GLM:
        return cls(
            input_dim=input_dim,
            output_dim=spec.dim,
            tasktype=spec.task,
        )


class GLMStructured(nn.Module):
    "Generalized Linear Models for predicting multiple outputs."

    def __init__(self, input_dim: int, tasks: dict[TaskSpec]):
        super(GLMStructured, self).__init__()
        self.models = nn.ModuleDict({
            name: GLM.from_spec(input_dim, task) for name, task in tasks.items()
        })

    def forward(self, x, dataset_name: Optional[str] = None):
        if dataset_name is None:
            return {name: model(x) for name, model in self.models.items()}
        else:
            return self.models[dataset_name](x)


def get_probability_calibration(predictions, y_test, metric) -> float:
    "Calibrate the probability threshold to the value that gives you the best value of a given metric."
    threshold = np.arange(0, 1, 0.01)
    scores = []
    for t in threshold:
        predictions_binary = np.where(predictions > t, 1, 0)
        scores.append(metric(y_test, predictions_binary))
    best_threshold = threshold[np.argmax(scores)]
    return best_threshold


# input structure should be a dictionary with the following keys:
# name, and then model, optimizer, loaders, loss func, etc.

# y_true = {name: TensorSpec}


def train_loop(
    datasets: dict[str, dict],
    epochs: int,
) -> dict:
    log = {k: [] for k in ["epoch", "train_loss", "val_loss", "val_metric", "dataset"]}
    for name, dataset_store in datasets.items():
        train_dataloader = dataset_store["train_dataloader"]
        test_dataloader = dataset_store["test_dataloader"]
        scaler = dataset_store["scaler"]
        tasktype = dataset_store["tasktype"]
        task_specs = dataset_store["task_specs"]
        glm_structured = GLMStructured(train_dataloader.dataset.x.shape[1], task_specs)
        loss_fn = get_loss_fn_dict(glm_structured.models)
        optimizers = get_optimizer_dict(glm_structured.models)

        epoch_number: int = 0
        pbar = tqdm.tqdm(range(epochs))

        for epoch in range(epochs):
            log["epoch"].append(epoch)
            log["dataset"].append(name)

            for model in glm_structured.models.values():
                model.train(True)

            training_loss = 0
            avg_train_loss = 0
            test_loss = 0
            avg_test_loss = 0

            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                for optimizer in optimizers.values():
                    optimizer.zero_grad()
                outputs: dict = glm_structured(inputs)
                print(labels, task_specs)
                label_mask: dict = get_mask(labels, task_specs, tasktype)
                loss: dict = use_loss_fn_dict(
                    loss_fn, outputs, labels, label_mask, task_specs, tasktype
                )
                loss["sum"] = get_loss_sum(loss)
                loss["sum"].backward()
                for optimizer in optimizers.values():
                    optimizer.step()  # TODO: check with Ben, is this correct?
                training_loss += loss["sum"].item()

            training_loss /= len(train_dataloader)
            avg_train_loss += training_loss
            log["train_loss"].append(training_loss)

            # print(f"{avg_loss=}")

            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                y_pred, y_test = {}, {}
                for name, model in glm_structured.models.items():
                    model.eval()
                    y_pred[name] = []
                    y_test[name] = []
                for i, vdata in enumerate(test_dataloader):
                    vinputs, vlabels = vdata
                    voutputs = glm_structured(vinputs)
                    vlabel_mask: dict = get_mask(vlabels, task_specs, tasktype)
                    vloss: dict = use_loss_fn_dict(
                        loss_fn, voutputs, vlabels, vlabel_mask, task_specs, tasktype
                    )
                    vloss["sum"] = get_loss_sum(vloss)
                    for name, output in voutputs.items():
                        y_pred[name].append(output)
                        y_test[name].append(vlabels)
                    test_loss += vloss["sum"].item()
                # concatenate all the predictions and labels
                for name, output in y_pred.items():
                    y_pred[name] = torch.cat(y_pred[name], dim=0)
                    y_test[name] = torch.cat(y_test[name], dim=0)
                # print(y_pred[name])
                # print(y_test[name])
                # Calculate metrics from the predictions of the model after the last epoch.
                test_loss /= len(test_dataloader)
                avg_test_loss += test_loss
                log["val_loss"].append(test_loss)
                metric_results = get_metrics_dict(
                    y_pred, y_test, task_specs, tasktype, scaler
                )
                log["val_metric"].append(metric_results)
                # print some statistics
                pbar.set_description(
                    f"Train: {training_loss:.4f} | Test: {test_loss:.4f} | Test metric: {metric_results} | Dataset: {name}"
                )
            # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss, ".5f"))

            epoch_number += 1

        # Calculate metrics from the predictions of the model after the last epoch.
        # check early stopping based on losses averaged of datasets
        avg_train_loss /= epochs
        avg_test_loss /= epochs
        log["epoch"].append(epoch)
        log["train_loss"].append(avg_train_loss)
        log["val_loss"].append(avg_test_loss)
        log["val_metric"].append(None)
        log["dataset"].append(name)
    log = pd.DataFrame(log)
    log.to_csv(f"results/training.csv", index=False)
