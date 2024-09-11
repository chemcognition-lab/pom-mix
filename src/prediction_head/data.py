# sets the path to the root of the repository
from pathlib import Path
import sys
from typing import Dict

root_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(root_path))

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import enum
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    accuracy_score,
)
import torchmetrics.functional as F
from scipy.stats import kendalltau, spearmanr
from prediction_head.loss import FocalLoss
import functools
import dataclasses

# Load Data (Create Dataset and Dataloader)
from torch.utils.data import DataLoader, Dataset

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

np.random.seed(2)


class TaskType(StrEnum):
    regression = enum.auto()
    binary = enum.auto()
    multiclass = enum.auto()
    multilabel = enum.auto()
    zero_inflated = enum.auto()


@dataclasses.dataclass
class TaskSpec:
    "Task specification for a GLM."
    dim: int
    task: TaskType


def get_activation(tasktype: TaskType) -> nn.Module:
    activations = {
        TaskType.regression: nn.Identity,
        TaskType.binary: nn.Identity,
        TaskType.multiclass: nn.Identity,  # functools.partial(nn.Softmax, dim=1),
        TaskType.multilabel: nn.Identity,
    }
    return activations[tasktype]


def get_loss_fn(task: TaskType):
    loss_fns: dict = {
        TaskType.regression: nn.MSELoss,
        TaskType.binary: nn.BCEWithLogitsLoss,  # requires logits
        TaskType.multiclass: nn.CrossEntropyLoss,  # requires logits
        TaskType.multilabel: nn.BCEWithLogitsLoss,  # requires logits
    }
    return loss_fns[task]


def get_loss_fn_dict(models: dict[str, nn.Module]):
    loss_fn_dict = {}
    for name, model in models.items():
        loss_fn_dict[name] = get_loss_fn(model.tasktype)
    return loss_fn_dict


def use_loss_fn_dict(
    loss_fn: dict,
    y_pred: dict[torch.Tensor],
    y_true: torch.Tensor,
    label_mask: dict[torch.Tensor],
    task_specs: dict,
    tasktype: TaskType,
) -> dict[torch.Tensor]:
    "Get the loss for each task in the structured prediction."

    loss_dict = {}
    for (name, loss_fn), y_pred_tensor, label_mask_tensor in zip(
        loss_fn.items(), y_pred.values(), label_mask.values()
    ):
        if tasktype == TaskType.zero_inflated:
            if task_specs[name].task == TaskType.binary:
                y_true = torch.where(y_true != 0, 1.0, 0.0)
        print(y_pred_tensor * label_mask_tensor, y_true * label_mask_tensor)
        loss_dict[name] = loss_fn()(
            y_pred_tensor * label_mask_tensor, y_true * label_mask_tensor
        )

    return loss_dict


def get_loss_sum(loss_dict: dict[torch.Tensor]) -> torch.Tensor:
    "Sum the loss for each task in the structured prediction from TensorSpec.tensor."
    loss_tensor_list = list(loss_dict.values())
    return sum(loss_tensor_list)


def get_metrics(tasktype: TaskType):
    metrics: dict[dict] = {
        TaskType.regression: {
            "r2": r2_score,
            "r": np.corrcoef,
            "rmse": F.root_mean_squared_error,
            "mse": mean_absolute_error,
        },
        TaskType.binary: {"auroc": roc_auc_score, "acc": accuracy_score},
        TaskType.multiclass: {
            "auroc": functools.partial(roc_auc_score, multi_class="ovr"),
            "acc": accuracy_score,
        },
        TaskType.multilabel: {
            "auroc": functools.partial(roc_auc_score, multi_class="ovr"),
            "acc": accuracy_score,
        },
        TaskType.zero_inflated: {
            "kendalltau": kendalltau,
            "spearmanr": spearmanr,
        },
    }
    return metrics[tasktype]


# "task_specs": {"regression": TaskSpec(1, TaskType.regression)},
def get_metrics_dict(
    y_pred_dict: dict[torch.Tensor],
    y_true_dict: dict[torch.Tensor],
    task_specs: dict,
    tasktype: TaskType,
    scaler,
) -> dict[dict]:
    metric_results = {}
    for name, task_spec in task_specs.items():
        y_pred = y_pred_dict[name]
        y_true = y_true_dict[name]
        if task_spec.task == TaskType.regression:
            y_pred = scaler.inverse_transform(y_pred.numpy())
            y_true = scaler.inverse_transform(y_true.numpy())
        if task_spec.task == TaskType.binary and tasktype == TaskType.zero_inflated:
            y_true = torch.where(y_true != 0, 1.0, 0.0)
        metrics: dict = get_metrics(task_spec.task)
        results = {}
        for metric_name, metric_fn in metrics.items():
            if metric_name == "r":
                results[metric_name] = metric_fn(y_true.flatten(), y_pred.flatten())[
                    0, 1
                ]
            elif metric_name == "acc":
                # convert y_pred to one-hot-encoding
                zeroes = np.zeros_like(y_true)
                y_pred = np.argmax(y_pred, axis=-1)
                zeroes[np.arange(len(y_pred)), y_pred] = 1
                results[metric_name] = metric_fn(y_true, zeroes)
            else:
                print(y_true.shape, y_pred.shape)
                results[metric_name] = metric_fn(y_true, y_pred)
        metric_results[name] = results
    results = {}
    if tasktype == TaskType.zero_inflated:
        metrics: dict = get_metrics(tasktype)
        # combine binary and regression task for y_pred and y_true
        y_pred = torch.cat([y_pred_dict["zi_binary"], y_pred_dict["zi_regression"]])
        y_true = torch.cat([y_true_dict["zi_binary"], y_true_dict["zi_regression"]])
        for metric_name, metric_fn in metrics.items():
            if metric_name == "kendalltau":
                results[metric_name] = metric_fn(y_true.flatten(), y_pred.flatten())[0]
            elif metric_name == "spearmanr":
                results[metric_name] = metric_fn(y_true.flatten(), y_pred.flatten())[0]
        metric_results["zero_inflated"] = results

    return metric_results


def get_mask(y_true: torch.tensor, task_specs: dict, tasktype: TaskType) -> dict:
    label_mask = {}
    for name, task_spec in task_specs.items():
        if tasktype == TaskType.zero_inflated:
            if task_spec.task == TaskType.binary:
                label_mask[name] = torch.tensor(np.where(y_true != 0, 1, 0))
            elif task_spec.task == TaskType.regression:
                label_mask[name] = torch.tensor(np.where(y_true > 1e-14, 1, 0))
        else:
            label_mask[name] = torch.ones_like(y_true)
    return label_mask


def get_optimizer_dict(models: dict[str, nn.Module]) -> dict:
    optimizer_dict = {}
    for name, model in models.items():
        optimizer_dict[name] = torch.optim.Adam(model.parameters(), lr=1e-3)
    return optimizer_dict


def min_max_scale(train_data, test_data):
    scaler = MinMaxScaler()
    x_scaler = scaler.fit(train_data)
    train_data = x_scaler.transform(train_data)
    test_data = x_scaler.transform(test_data)
    return train_data, test_data, scaler


class SyntheticSklearnDatasets(Dataset):
    def __init__(self, x, y):
        self.x: torch.tensor = x
        self.y: torch.tensor = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_regression_dataset(
    n_samples: int = 1000,
    n_features: int = 25,
    n_informative: int = 10,
    noise=0.1,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=22,
    )

    # Pre-processing
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22
    )

    # min-max scale
    x_train, x_test, scaler = min_max_scale(x_train, x_test)
    y_train, y_test, scaler = min_max_scale(
        y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    )

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    train_dataset = SyntheticSklearnDatasets(x_train, y_train)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "scaler": scaler,
        "tasktype": TaskType.regression,
        "task_specs": {"regression": TaskSpec(1, TaskType.regression)},
    }


def get_binary_dataset(
    n_samples: int = 1000,
    n_features: int = 25,
    n_informative: int = 10,
    n_redundant: int = 2,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        random_state=22,
    )

    # Pre-processing
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22
    )
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    # min-max scale
    x_train, x_test, scaler = min_max_scale(x_train, x_test)

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    train_dataset = SyntheticSklearnDatasets(x_train, y_train)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "scaler": scaler,
        "tasktype": TaskType.binary,
        "task_specs": {"binary": TaskSpec(1, TaskType.binary)},
    }


def get_multiclass_dataset(
    n_samples: int = 1000,
    n_features: int = 25,
    n_classes: int = 5,
    n_informative: int = 10,
    n_redundant: int = 2,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        n_redundant=n_redundant,
        random_state=22,
    )

    # Pre-processing
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22
    )
    # y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1

    # min-max scale
    x_train, x_test, scaler = min_max_scale(x_train, x_test)

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    train_dataset = SyntheticSklearnDatasets(x_train, y_train)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "scaler": scaler,
        "tasktype": TaskType.multiclass,
        "task_specs": {"multiclass": TaskSpec(1, TaskType.multiclass)},
    }


def get_multilabel_dataset(
    n_samples: int = 1000,
    n_features: int = 25,
    n_classes: int = 5,
    n_labels: int = 3,
    allow_unlabeled: bool = False,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=n_labels,
        random_state=22,
        allow_unlabeled=allow_unlabeled,
    )

    # Pre-processing
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22
    )

    # min-max scale
    x_train, x_test, scaler = min_max_scale(x_train, x_test)

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    train_dataset = SyntheticSklearnDatasets(x_train, y_train)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "scaler": scaler,
        "tasktype": TaskType.multilabel,
        "task_specs": {"multilabel": TaskSpec(n_classes, TaskType.multilabel)},
    }


def get_zeroinflated_dataset(
    n_samples: int = 1000,
    n_features: int = 25,
    n_informative: int = 10,
    noise: float = 0.1,
    zero_inflated_percent: float = 0.8,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=22,
    )

    # Add zero-inflated datapoints
    y = np.abs(y)
    # Set the y-value of 80% of the sample to 0
    is_zero = np.random.choice(
        y.shape[0], int(y.shape[0] * zero_inflated_percent), replace=False
    )
    y[is_zero] = 0

    # Pre-processing
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22
    )

    # min-max scale
    x_train, x_test, scaler = min_max_scale(x_train, x_test)
    y_train, y_test, scaler = min_max_scale(
        y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    )

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    train_dataset = SyntheticSklearnDatasets(x_train, y_train)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "scaler": scaler,
        "tasktype": TaskType.zero_inflated,
        "task_specs": {
            "zi_binary": TaskSpec(1, TaskType.binary),
            "zi_regression": TaskSpec(1, TaskType.regression),
        },
    }


def get_zeroinflated_negative_binomial_dataset(
    n_samples: int = 1000,
    n_features: int = 25,
    n_informative: int = 10,
    noise: float = 0.1,
    zero_inflated_percent: float = 0.8,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=22,
    )

    y = np.random.default_rng(seed=22).negative_binomial(size=n_samples, n=1, p=0.5)

    # Add zero-inflated datapoints by making zero_inflated_percent number of datapoints equal to 0
    is_zero = np.random.choice(
        y.shape[0], int(y.shape[0] * zero_inflated_percent), replace=False
    )
    y[is_zero] = 0

    # One-hot-encode the zero-inflated datapoints if the tasktype is zero-inflated-binary, //
    # and mask the zero-inflated datapoints if the tasktype is zero-inflated-regression
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22
    )

    # min-max scale
    x_train, x_test, scaler = min_max_scale(x_train, x_test)
    y_train, y_test, scaler = min_max_scale(
        y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    )

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    train_dataset = SyntheticSklearnDatasets(x_train, y_train)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "scaler": scaler,
        "tasktype": TaskType.zero_inflated,
        "task_specs": {
            "zi_binary": TaskSpec(1, TaskType.binary),
            "zi_regression": TaskSpec(1, TaskType.regression),
        },
    }


def get_zeroinflated_exponential_dataset(
    name: str = "zero_inflated_exponential",
    n_samples: int = 1000,
    n_features: int = 25,
    n_informative: int = 10,
    noise: float = 0.1,
    zero_inflated_percent: float = 0.8,
):
    # Dataset Generator (multi-dim regression, multilabel, multiclass, zero-inflated-exponential)
    x, y = datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=22,
    )

    y = np.random.default_rng(seed=22).exponential(scale=4, size=n_samples)

    # Add zero-inflated datapoints by making zero_inflated_percent number of datapoints equal to 0
    is_zero = np.random.choice(
        y.shape[0], int(y.shape[0] * zero_inflated_percent), replace=False
    )
    y[is_zero] = 0

    # Pre-processing
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22
    )

    # min-max scale
    x_train, x_test, scaler = min_max_scale(x_train, x_test)
    y_train, y_test, scaler = min_max_scale(
        y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    )

    # convert to Tensors
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    train_dataset = SyntheticSklearnDatasets(x_train, y_train)
    test_dataset = SyntheticSklearnDatasets(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "scaler": scaler,
        "tasktype": TaskType.zero_inflated,
        "task_specs": {
            "zi_binary": TaskSpec(1, TaskType.binary),
            "zi_regression": TaskSpec(1, TaskType.regression),
        },
    }
