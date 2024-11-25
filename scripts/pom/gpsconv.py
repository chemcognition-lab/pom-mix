import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:-2])
sys.path.append(str(base_dir / "src/"))

import argparse
import os.path as osp
from typing import Any, Dict, Optional

from sklearn.metrics import roc_auc_score

import numpy as np

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention

from torch_geometric.data import Dataset
from dataloader import DatasetLoader

from torch.nn.functional import binary_cross_entropy_with_logits




class GSLFDataset(Dataset):
    def __init__(self, x, y, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.x = x
        for data in self.x:
            data.x = data.x[:,0]
            data.edge_attr = data.edge_attr[:,0]

        if self.pre_transform is not None:
            self.x = [self.pre_transform(data) for data in self.x]
        self.y = y

    def len(self):
        return len(self.x)

    def get(self, idx):
        return self.x[idx], torch.tensor(self.y[idx], dtype=torch.float32)




dl = DatasetLoader()
dl.load_dataset('gs-lf')
dl.featurize('pyg_molecular_graphs')
transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
ds = GSLFDataset(dl.features, dl.labels, '.', pre_transform=transform)

split = np.load('../pom/gs-lf_models/gs-lf_0.2.npz')
train_ind, test_ind = split["train_ind"], split["test_ind"]

train_dataset = ds[train_ind]
val_dataset = ds[test_ind]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--attn_type', default='multihead',
    help="Global attention type such as 'multihead' or 'performer'.")
args = parser.parse_args()


class GPS(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()

        self.node_emb = Embedding(119, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(22, channels) # based on ogb

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn), heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 138),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        x = global_add_pool(x, batch)
        return self.mlp(x)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
attn_kwargs = {'dropout': 0.5}
model = GPS(channels=64, pe_dim=8, num_layers=10, attn_type=args.attn_type,
            attn_kwargs=attn_kwargs).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                              min_lr=0.00001)


def train():
    model.train()

    total_loss = 0
    for data, y in train_loader:
        data = data.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        model.redraw_projection.redraw_projections()
        out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                    data.batch)
        loss = binary_cross_entropy_with_logits(out, y)
        # loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_error = 0
    y_pred, y_true = [], []
    for data, y in loader:
        data = data.to(device)
        y = y.to(device)
        out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                    data.batch)
        
        y_pred.append(out.detach().cpu().numpy())
        y_true.append(y.detach().cpu().numpy())

        # total_error += binary_cross_entropy_with_logits(out, y).item()
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    auc = roc_auc_score(y_true, y_pred)
    loss = binary_cross_entropy_with_logits(torch.tensor(y_pred), torch.tensor(y_true)).item()
    return loss, auc


best_val = 0
best_epoch = 0
for epoch in range(1, 301):
    loss = train()
    val_loss, auc = test(val_loader)
    scheduler.step(val_loss)
    
    if best_val < auc:
        best_val = auc
        best_epoch = epoch

    print(f'Epoch: {epoch:02d}, Loss: {loss:.5f}, Val loss: {val_loss:.5f}, Val AUC: {auc:.5f}')
    print(f'Best epoch: {best_epoch}, best auc: {best_val}')