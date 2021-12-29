import math

import torch
from torch.nn.functional import relu
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolutionLayer(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 初始化权重

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        x = torch.mm(inputs, self.weight)
        output = torch.spmm(adj, x)
        if self.bias:
            return output + self.bias
        else:
            return output