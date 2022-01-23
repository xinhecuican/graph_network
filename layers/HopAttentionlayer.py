import pdb

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_sparse import spspmm, SparseTensor


class HopAttentionLayer(Module):

    def __init__(self, in_features, out_features, dropout, alpha, hop_num, hop_alpha, concat=True):
        super(HopAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活
        self.hop_num = hop_num
        self.hop_alpha = hop_alpha
        self.w = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        h = torch.mm(inp, self.w)
        N = h.size()[0]
        # repeat: 重复若干次，例如(2708, 8) 经过(1, N)的repeat后变为了 (2708, 8 * 2708)
        # view: 将矩阵按行展开成一维(内存东西没变)然后按照给定的shape生成新的矩阵(N * N, -1)后变成(2708*2708, 8)
        # 前面一个矩阵经过处理之后形式为:第一个2708行中的内容全是第0个点的特征，依此类推
        # 后面一个矩阵形式为: 第一个2708行是第0个到第2707节点的特征，第二个2708又是第0个到第2707个节点特征
        # 也就是说最后的矩阵中第一行的前8个特征是第一个点的特征，而后8个特征是0-2707节点的特征
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        hop_num = e
        temp = e
        for i in range(1, self.hop_num + 1):
            weight = self.hop_alpha * ((1 - self.hop_alpha) ** i)
            hop_num = torch.mm(hop_num, e)
            temp = temp + hop_num * weight
        e = temp
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpecialSpspmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, shape):
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        (edge, edge_e) = spspmm(
                        indexA=a._indices(),
                        indexB=b._indices(),
                        valueA=a._values(),
                        valueB=b._values(),
                        m=shape[0],
                        k=shape[0],
                        n=shape[0])
        return torch.sparse_coo_tensor(edge, edge_e, torch.Size([shape[0], shape[0]]))

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        save_b = ctx.saved_tensors[1]
        grad_values = grad_b = grad_b_values = None
        if ctx.needs_input_grad[1]:
            (edge, edge_e) = spspmm(
                indexA=grad_output._indices(),
                indexB=save_b.t()._indices(),
                valueA=grad_output._values(),
                valueB=save_b.t()._values(),
                m=ctx.N,
                k=ctx.N,
                n=ctx.N)
            pdb.set_trace()
            grad_a_dense = torch.sparse_coo_tensor(edge, edge_e, torch.Size([ctx.N, ctx.N]))
            # grad_a_dense = grad_output.matmul(b.t())
            # edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            # grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            (edge, edge_e) = spspmm(
                indexA=a.t()._indices(),
                indexB=grad_output._indices(),
                valueA=a.t()._values(),
                valueB=grad_output._values(),
                m=ctx.N,
                k=ctx.N,
                n=ctx.N)
            grad_b = torch.sparse_coo_tensor(edge, edge_e, torch.Size([ctx.N, ctx.N]))
            # grad_b = a.t().matmul(grad_output)
        return grad_a_dense, grad_b, None


class SpeicalSpspmm(nn.Module):

    def forward(self, a, b, shape):
        return SpecialSpspmmFunction.apply(a, b, shape)


class SpHopAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, hop_num, hop_alpha, att_type, concat=True):
        super(SpHopAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.hop_num = hop_num
        self.hop_alpha = hop_alpha
        self.att_type = att_type

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        if self.att_type == 'li' or self.att_type == 'mx':
            self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 2 * out_features)))
            nn.init.xavier_normal_(self.a.data, gain=1.414)
        elif self.att_type == 'dp':
            pass

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.special_spspmm = SpeicalSpspmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        if adj.is_sparse:
            edge = adj._indices()
        else:
            edge = adj.nonzero().t()
        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E
        if self.att_type == 'li':
            edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).sum(0).squeeze()))
        elif self.att_type == 'dp':
            conv_tensor = h[edge[0, :], :] * h[edge[1, :], :]
            edge_e = torch.exp(-self.leakyrelu(conv_tensor.sum(1).squeeze()))
        elif self.att_type == 'mx':
            conv_tensor = h[edge[0, :], :] * h[edge[1, :], :]
            edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).sum(0) * torch.sigmoid(conv_tensor.sum(1).squeeze())))
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        # tmp_edge = edge
        # tmp_edge_e = edge_e
        # tmp_ans = torch.sparse_coo_tensor(edge, edge_e, torch.Size([N, N])).to_dense()
        # for i in range(1, self.hop_num + 1):
        #     weight = self.hop_alpha * ((1 - self.hop_alpha) ** i)
        #     b = torch.sparse_coo_tensor(tmp_edge, tmp_edge_e, torch.Size([N, N])).to_dense()
        #     tmp_ans = tmp_ans + self.special_spmm(edge, edge_e, torch.Size([N, N]), b) * weight
        #     A = torch.sparse_coo_tensor(tmp_edge, tmp_edge_e, torch.Size([N, N]))
        #     B = torch.sparse_coo_tensor(edge, edge_e, torch.Size([N, N])).to_dense()
        #     temp_sparse = torch.sparse.mm(A, B).to_sparse()
        #     tmp_edge = temp_sparse._indices()
        #     tmp_edge_e = temp_sparse._values()
        # tmp_ans = tmp_ans.to_sparse()
        # edge = tmp_ans._indices()
        # edge_e = tmp_ans._values()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        # e_rowsum = torch.where(e_rowsum == 0, torch.full_like(e_rowsum, 1e-8), e_rowsum)
        e_rowsum = e_rowsum + 1e-8
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
