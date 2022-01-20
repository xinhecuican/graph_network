import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.GraphAttentionLayer import GraphAttentionLayer, SpGraphAttentionLayer
from layers.HopAttentionlayer import SpHopAttentionLayer


class ExtendGAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(ExtendGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class ExtendSpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, hop_num, hop_alpha):
        """Sparse version of GAT."""
        super(ExtendSpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpHopAttentionLayer(nfeat,
                                               nhid,
                                               dropout=dropout,
                                               alpha=alpha,
                                               hop_num=hop_num,
                                               hop_alpha=hop_alpha,
                                               concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpHopAttentionLayer(nhid * nheads,
                                           nclass,
                                           dropout=dropout,
                                           alpha=alpha,
                                           hop_num=hop_num,
                                           hop_alpha=hop_alpha,
                                           concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
