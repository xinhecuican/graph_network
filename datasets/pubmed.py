import torch.utils.data
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import eye
from torch_geometric.datasets import Planetoid

from utils.encoders import encode_onehot, sparse_mx_to_torch_sparse_tensor
from utils.normalize import normalize
from datasets.cora import cora


class pubmed(torch.utils.data.Dataset):

    def __init__(self, model, state="train"):
        dataset_pubmed = Planetoid(root='./data/pubmed/', name='Pubmed')
        self.state = state
        self.features = dataset_pubmed.data.x
        self.labels = dataset_pubmed.data.y
        self.idx_train = range(dataset_pubmed.data.train_mask.sum())
        self.idx_val = range(500, 1000)
        self.idx_test = range(2000, 3000)

        edges = dataset_pubmed.data.edge_index.T.numpy()
        # edges = np.array(list(map(self.idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
        #     edges_unordered.shape)
        self.adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                              shape=(self.labels.shape[0], self.labels.shape[0]), dtype=np.float32)
        self.adj = self.adj + self.adj.T.multiply(self.adj.T > self.adj) - self.adj.multiply(self.adj.T > self.adj)
        self.adj = normalize(self.adj + eye(self.adj.shape[0]))
        if model == 'gat':
            self.adj = torch.FloatTensor(np.array(self.adj.todense()))
        else:
            self.adj = sparse_mx_to_torch_sparse_tensor(self.adj)
