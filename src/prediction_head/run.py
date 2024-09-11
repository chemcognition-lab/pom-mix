# sets the path to the root of the repository
from pathlib import Path
import sys
import os

import torch

root_path = os.path.dirname(os.path.abspath(""))
sys.path.append(str(root_path))
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from prediction_head.data import (
    get_regression_dataset,
    get_binary_dataset,
    get_multiclass_dataset,
    get_multilabel_dataset,
    get_zeroinflated_dataset,
    get_zeroinflated_negative_binomial_dataset,
    get_zeroinflated_exponential_dataset,
    TaskType,
    TaskSpec,
)

from prediction_head.GLM import GLM, train_loop, TaskSpec
from prediction_head.plot import (
    plot_regression_distribution,
    plot_classification_distribution,
    plot_ground_truth_vs_predictions,
)

# load data
datasets: dict = {
    "regression": get_regression_dataset(),
    "binary": get_binary_dataset(),
    # "multiclass": get_multiclass_dataset(), #TODO: issue with converting probability (negative btw) to class distribution
    "multilabel": get_multilabel_dataset(),
    "zero_inflated": get_zeroinflated_dataset(),
    "zero_inflated_negative_binomial": get_zeroinflated_negative_binomial_dataset(),
    "zero_inflated_exponential": get_zeroinflated_exponential_dataset(),
}

train_loop(datasets, 10)
