
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch

class DistSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.n_layers, self.activation = n_layers, activation
        self.dropout = nn.Dropout(dropout)
        dims = [in_feats] + [n_hidden] * (n_layers - 1) + [n_classes]
        self.layers = nn.ModuleList([
            dgl.nn.pytorch.SAGEConv(dims[i], dims[i+1], "mean") for i in range(n_layers)
        ])

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
