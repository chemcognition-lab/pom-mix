from typing import List, Union, Optional, Callable

import torch
import torch.nn as nn

import torch.utils.data
from torch_geometric.data import Data
from torch_geometric.data.data import size_repr
from torch_scatter import scatter_sum

from .graph_utils import get_pooling_function


class RevIndexedData(Data):
    def __init__(self, orig):
        super(RevIndexedData, self).__init__()
        if orig:
            for key in orig.keys():
                self[key] = orig[key]
            edge_index = self["edge_index"]
            revedge_index = torch.zeros(edge_index.shape[1]).long()
            for k, (i, j) in enumerate(zip(*edge_index)):
                edge_to_i = edge_index[1] == i
                edge_from_j = edge_index[0] == j
                revedge_index[k] = torch.where(
                    edge_to_i & edge_from_j)[0].item()
            self["revedge_index"] = revedge_index

    def __inc__(self, key, value, *args, **kwargs):
        if key == "revedge_index":
            return self.revedge_index.max().item() + 1
        else:
            return super().__inc__(key, value)

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))


class ChemProp(nn.Module):
    def __init__(self, hidden_dim: int, node_dim: int, edge_dim: int, pooling_fn: Callable, depth: Optional[int] = 3):
        super(ChemProp, self).__init__()
        self.act_func = nn.ReLU()
        self.W1 = nn.Linear(node_dim + edge_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W3 = nn.Linear(node_dim + hidden_dim, hidden_dim, bias=True)
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.pool = pooling_fn
        self.depth = depth

    def forward(self, data):
        x, edge_index, revedge_index, edge_attr, num_nodes, batch = (
            data.x,
            data.edge_index,
            data.revedge_index,
            data.edge_attr,
            data.num_nodes,
            data.batch,
        )

        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        h0 = self.act_func(self.W1(init_msg))

        # directed message passing over edges
        h = h0
        for _ in range(self.depth - 1):
            m = self.directed_mp(h, edge_index, revedge_index)
            h = self.act_func(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = self.aggregate_at_nodes(num_nodes, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))

        return self.pool(node_attr, batch)


    
    @staticmethod
    def directed_mp(message, edge_index, revedge_index):
        m = scatter_sum(message, edge_index[1], dim=0)
        m_all = m[edge_index[0]]
        m_rev = message[revedge_index]
        return m_all - m_rev

    @staticmethod
    def aggregate_at_nodes(num_nodes, message, edge_index):
        m = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
        return m[torch.arange(num_nodes)]
