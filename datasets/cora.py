import torch.utils.data
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import eye
from utils.encoders import encode_onehot, sparse_mx_to_torch_sparse_tensor
from utils.normalize import normalize


class cora(torch.utils.data.Dataset):

    def __init__(self, path, model, state="train"):
        self.path = path
        self.state = state
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, "cora"), dtype=np.dtype(str))
        self.features = csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        # labels有七种，表示这篇论文属于那种门类
        self.labels = encode_onehot(idx_features_labels[:, -1])

        self.idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        self.idx_map = {j: i for i, j in enumerate(self.idx)}

        edges_unordered = np.genfromtxt("{}{}.cites".format(path, "cora"), dtype=np.int32)
        edges = np.array(list(map(self.idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            edges_unordered.shape)
        self.adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                   shape=(self.labels.shape[0], self.labels.shape[0]), dtype=np.float32)
        self.adj = self.adj + self.adj.T.multiply(self.adj.T > self.adj) - self.adj.multiply(self.adj.T > self.adj)
        self.adj = normalize(self.adj + eye(self.adj.shape[0]))
        if model == 'gat':
            self.adj = torch.FloatTensor(np.array(self.adj.todense()))
        else:
            self.adj = sparse_mx_to_torch_sparse_tensor(self.adj)
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        self.features = torch.FloatTensor(np.array(self.features.todense()))
        self.labels = torch.LongTensor(np.where(self.labels)[1])

        self.idx_train = torch.LongTensor(idx_train)
        self.idx_val = torch.LongTensor(idx_val)
        self.idx_test = torch.LongTensor(idx_test)

    def __len__(self):
        if self.state == "train":
            return len(self.idx_train)
        elif self.state == "test":
            return len(self.idx_test)
        elif self.state == "val":
            return len(self.idx_val)

    def __getitem__(self, index):
        return self.features, self.adj
