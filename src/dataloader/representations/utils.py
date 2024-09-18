import torch
import numpy as np

def pna(feat):
    """
    Input tensor of shape (n, m, d), where n is the number of samples, m is the number of features, and d is the dimesnion of features
    will produce a tensor of shape (n, 4*d) where each feature is replaced by its mean, variance, max, and min
    Collapse along the second dimension
    """
    if torch.is_tensor(feat):
        feat[feat == -999] = torch.nan
        new_feat = torch.zeros((len(feat), feat.shape[-1]*4))
        for i, x in enumerate(feat):
            x = x[~torch.isnan(x).any(dim=1)]
            var = torch.zeros(x.shape[1]) if len(x)==1 else x.var(0)
            new_feat[i,:] = torch.cat([x.mean(0), var, x.max(0)[0], x.min(0)[0]])
        return new_feat
    else:
        feat[feat == -999] = np.nan
        new_feat = np.zeros((len(feat), feat.shape[-1]*4))
        for i, x in enumerate(feat):
            x = x[~np.isnan(x).any(axis=1)]
            var = np.zeros(x.shape[1]) if len(x) == 1 else x.var(axis=0)
            new_feat[i, :] = np.concatenate([x.mean(axis=0), var, x.max(axis=0), x.min(axis=0)])
        return new_feat