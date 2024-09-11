from typing import Optional, List, Any

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as pygdl
import torchmetrics.functional as F

import random
import json
import numpy as np

from .gnn import graph_utils, GraphNets
from .data import GraphDataset


def set_seed(seed: int = 42):
    print(f'Seed set to {seed}')
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loss_function(task: str):
    # DEPRECATED, use the one in prediction_head.data
    loss_dict = {
        'multilabel': nn.BCEWithLogitsLoss(),
        'binary': nn.BCEWithLogitsLoss(),
        'multiclass': nn.CrossEntropyLoss(),        # also expects logits
    }
    return loss_dict[task]

def multilabel_auroc(pred: torch.Tensor, targ: torch.Tensor):
    # we remove columns that have no meaning in ground truth
    non_zero_ind = torch.sum(targ != 0, dim=0).nonzero().squeeze()
    targ = targ[:, non_zero_ind]
    pred = pred[:, non_zero_ind]
    return F.classification.multilabel_auroc(pred, targ.long(), targ.shape[-1])

def binary_accuracy(pred: torch.Tensor, targ: torch.Tensor):
    pred_class = pred > 0
    targ_class = targ > 0.5
    return ((pred_class.long() - targ_class.long()) == 0).double().mean()

                             
def get_metric_function(task: str):
    loss_dict = {
        'multilabel': lambda pred, targ: F.classification.multilabel_auroc(pred, targ.long(), targ.shape[-1]),
        'multiclass': lambda pred, targ: F.classification.multiclass_auroc(pred, targ.long(), targ.shape[-1]),
        'binary': lambda pred, targ: F.classification.binary_auroc(pred, targ.long()),
        'regression': lambda pred, targ: F.r2_score(pred, targ) #F.kendall_rank_corrcoef(pred, targ)
    }
    return loss_dict[task]

def is_jsonable(x: nn.Module):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def jsonify(val: Any):
    if is_jsonable(val): 
        return val
    else:
        return val.__name__

def split_into_batches(*tensors, q):
    # create batches through indexing to ensure backpropagation
    if not all(tensor.size(0) == tensors[0].size(0) for tensor in tensors):
        raise ValueError("All input tensors must have the same first dimension")
    
    n = tensors[0].size(0)
    num_batches = (n + q - 1) // q  # Ceiling division to handle cases where n is not divisible by q
    
    return tuple(
        [tensor.narrow(0, i * q, min(q, n - i * q)) for i in range(num_batches)]
        for tensor in tensors
    )
