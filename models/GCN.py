import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu

from layers.GraphConvolutionLayer import GraphConvolutionLayer


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolutionLayer(nfeat, nhid, True)
        self.conv2 = GraphConvolutionLayer(nhid, nclass, False)
        self.dropout = dropout

    def forward(self, x, adj):
        x = relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return F.softmax(x, dim=1)
