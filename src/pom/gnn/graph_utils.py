import torch
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
from torch_geometric.nn.aggr import MultiAggregation, DegreeScalerAggregation

from ..data import GraphDataset


def get_graph_degree_histogram(train_dataset: GraphDataset):
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data[0].edge_index[1], num_nodes=data[0].num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data[0].edge_index[1], num_nodes=data[0].num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    
    return deg

def get_pooling_function(pool: str, pool_args: dict):
    pooling_func = {
        'mean': global_mean_pool,
        'add': global_add_pool,
        'pna': DegreeScalerAggregation(
            aggr= ['mean', 'min', 'max', 'std'], 
            scaler = ["identity", "amplification", "attenuation"],
            **pool_args) # PNA-style aggregation
    }
    return pooling_func[pool]

def get_model_parameters_count(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
