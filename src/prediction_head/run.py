# sets the path to the root of the repository
from .data import (
    get_regression_dataset,
    get_binary_dataset,
    get_multilabel_dataset,
    get_zeroinflated_dataset,
    get_zeroinflated_negative_binomial_dataset,
    get_zeroinflated_exponential_dataset,
)

from .GLM import train_loop

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
