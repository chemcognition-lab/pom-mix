from typing import Union, List, Optional

import json

import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.utils import scatter, to_dense_batch
from torch_geometric.nn import MetaLayer, Linear, GATConv, GAT
from torch_geometric.nn.aggr import MultiAggregation

# inspired by DIONYSUS (https://github.com/aspuru-guzik-group/dionysus/blob/main/dionysus/models/modules.py)
# and (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.MetaLayer.html#torch_geometric.nn.models.MetaLayer)

class EdgeMLPModel(nn.Module):
    def __init__(self, edge_dim: int, hidden_dim: Optional[int] = 50, num_layers: Optional[int] = 1):
        super().__init__()
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.edge_mlp = get_mlp(hidden_dim, edge_dim, num_layers)

    def forward(self, src, dst, edge_attr, u, batch):
        # src, dst: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dst, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)
    
class EdgeFiLMModel(nn.Module):
    def __init__(self, edge_dim: int, hidden_dim: Optional[int] = 50, num_layers: Optional[int] = 1, dropout: Optional[float] = 0.):
        super().__init__()
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # FiLM
        self.gamma = get_mlp(hidden_dim, edge_dim, num_layers, dropout=dropout)
        self.gamma_act = nn.Sigmoid()       # sigmoidal gating Dauphin et al.
        self.beta = get_mlp(hidden_dim, edge_dim, num_layers, dropout=dropout)

    def forward(self, src, dst, edge_attr, u, batch):
        # src, dst: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        cond = torch.cat([src, dst, u[batch]], 1)
        gamma = self.gamma_act(self.gamma(cond))
        beta = self.beta(cond)

        return gamma * edge_attr + beta
    

class NodeMLPModel(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: Optional[int] = 50, num_layers: Optional[int] = 1):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.node_mlp_1 = get_mlp(hidden_dim, node_dim, num_layers)
        self.node_mlp_2 = get_mlp(hidden_dim, node_dim, num_layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, u: torch.Tensor, batch: torch.Tensor):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0),
                      reduce='mean')
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)
    

class NodeAttnModel(nn.Module):
    def __init__(self, 
                 node_dim: int, 
                 hidden_dim: Optional[int] = 50, 
                 num_heads: Optional[int] = 5,
                 dropout: Optional[int] = 0., 
                 num_layers: Optional[int] = 1
                ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers

        # self attention layer
        self.self_attn = GAT(node_dim, hidden_dim, num_layers=num_layers, dropout=dropout, v2=True, heads=num_heads)
        self.output_mlp = get_mlp(hidden_dim, node_dim, num_layers=2)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, u: torch.Tensor, batch: torch.Tensor):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        attn = self.self_attn(x, edge_index, edge_attr)
        out = self.norm1(x + self.dropout_layer(attn))
        out = self.norm2(out + self.dropout_layer(self.output_mlp(out)))
        return out


class GlobalMLPModel(nn.Module):
    def __init__(self, global_dim: int, hidden_dim: Optional[int] = 50, num_layers: Optional[int] = 1):
        super().__init__()
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.global_mlp = get_mlp(hidden_dim, global_dim, num_layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, u: torch.Tensor, batch: torch.Tensor):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([
            u,
            scatter(x, batch, dim=0, reduce='mean'),
        ], dim=1)
        return self.global_mlp(out)
    

class GlobalPNAModel(nn.Module):
    def __init__(self, 
                 global_dim: int, 
                 hidden_dim: Optional[int] = 50, 
                 num_layers: Optional[int] = 2
                ):
        super().__init__()
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.pool = MultiAggregation(["mean", "std", "max", "min"])
        self.global_mlp = get_mlp(hidden_dim, global_dim, num_layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, u: torch.Tensor, batch: torch.Tensor):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        aggr = self.pool(x, batch)
        out = torch.cat([u, aggr], dim=1)
        return self.global_mlp(out)


class GlobalAttnModel(nn.Module):
    def __init__(self, global_dim: int, hidden_dim: Optional[int] = 50, num_layers: Optional[int] = 1, num_heads: Optional[int] = 5):
        super().__init__()
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.q_head = Linear(-1, global_dim)
        self.k_head = Linear(-1, global_dim)
        self.v_head = Linear(-1, global_dim)
        self.u_head = Linear(-1, global_dim)

        self.self_attn = nn.MultiheadAttention(global_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(global_dim, num_heads=num_heads, batch_first=True)
        self.global_mlp = get_mlp(global_dim, num_layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, u: torch.Tensor, batch: torch.Tensor):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        node_attr, node_mask = to_dense_batch(x, batch) # N, max_n_nodes, node_dim
        Q = self.q_head(node_attr)  # N, max_n_nodes, global_dim
        K = self.k_head(node_attr)
        V = self.v_head(node_attr)

        # self-attention layer
        n_attn, _ = self.self_attn(Q, K, V, key_padding_mask=~node_mask)    # N, max_n_nodes, global_dim
        n_attn = scatter(n_attn[node_mask], batch, dim=0, reduce='mean')

        u = self.u_head(u)  # N, global_dim

        # cross attention layer 
        u_attn, _ = self.cross_attn(n_attn, u, u)

        # final output layers
        out = torch.cat([n_attn, u_attn], dim=1)    # residual => sum
        
        return self.global_mlp(out)




##### Helper functions #####

def get_mlp(hidden_dim: int, output_dim: int, num_layers: int, dropout: Optional[float] = 0.):
    """
    Helper function to produce MLP with specified hidden dimension and layers
    """
    assert num_layers >= 0, 'Enter an integer larger than or equal to 0.'
    
    layers = nn.ModuleList()
    for _ in range(num_layers-1):
        layers.append(Linear(-1, hidden_dim))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.SELU())
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(Linear(-1, output_dim))
    return nn.Sequential(*layers)

def get_graphnet_mlp_layer(node_dim: int, edge_dim: int, global_dim: int, num_layers: Optional[int] = 2):
    """
    Helper function to produced GraphNets layer. 
    """
    node_net = NodeMLPModel(node_dim, num_layers=num_layers)
    edge_net = EdgeMLPModel(edge_dim, num_layers=num_layers)
    global_net = GlobalMLPModel(global_dim, num_layers=num_layers)
    return MetaLayer(edge_net, node_net, global_net)

def get_graphnet_attn_layer(node_dim: int, edge_dim: int, global_dim: int, num_layers: Optional[int] = 2):
    """
    Helper function to produced GraphNets layer. 
    """
    node_net = NodeMLPModel(node_dim, num_layers=num_layers)
    edge_net = EdgeMLPModel(edge_dim, num_layers=num_layers)
    global_net = GlobalAttnModel(global_dim, num_layers=num_layers)
    return MetaLayer(edge_net, node_net, global_net)


def get_graphnet_layer(
        node_dim: int, 
        edge_dim: int, 
        global_dim: int,
        hidden_dim: Optional[int] = 50,
    ):
    """
    Helper function to produced GraphNets layer. 
    """
    node_net = NodeAttnModel(node_dim, hidden_dim=hidden_dim)
    edge_net = EdgeFiLMModel(edge_dim, hidden_dim=hidden_dim)
    global_net = GlobalPNAModel(global_dim, hidden_dim=hidden_dim)
    return MetaLayer(edge_net, node_net, global_net)


class GraphNets(nn.Module):
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: int, 
                 global_dim: int, 
                 hidden_dim: Optional[int] = 50,
                 depth: Optional[int] = 3,
                 **kwargs
                ):
        super(GraphNets, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                get_graphnet_layer(
                    node_dim, 
                    edge_dim, 
                    global_dim, 
                    hidden_dim=hidden_dim
                ) for _ in range(depth)
            ]
        )

    def forward(self, data: pyg.data.Data):
        x, edge_index, edge_attr, u, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.u,
            data.batch,
        )

        for layer in self.layers:
            x, edge_attr, u = layer(x, edge_index, edge_attr, u, batch)
        
        return u
    
    @classmethod
    def from_json(cls, json_path: str):
        params = json.load(open(json_path, 'r'))
        return cls(**params)


