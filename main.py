import argparse
import torch
from torch import optim

from datasets.citeseer import citeseer
from datasets.cora import cora
from datasets.pubmed import pubmed
from models.ExtendGAT import ExtendSpGAT
from models.GAT import GAT, SpGAT
from models.GCN import GCN
from models.SGC import SGC
from trainer import train_gcn, train_sgc, accuracy, visualize, accuracy_class
from utils.utils import sgc_precompute
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during traing pass')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate')
# 权重衰减
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability)')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--model', default='egat')
parser.add_argument('--dataset', default='cora')
parser.add_argument('--debug', default=False)
parser.add_argument('--hop_num', type=int, default=3)
parser.add_argument('--hop_alpha', type=float, default=0.15)
parser.add_argument('--att_type', default='mx')

# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def main(args, dataset=None):
    if dataset is None:
        if args.dataset == 'cora':
            dataset = cora('data\\cora\\', args.model)
        elif args.dataset == 'pubmed':
            dataset = pubmed(args.model)
        elif args.dataset == 'citeseer':
            dataset = citeseer(args.model)
        else:
            dataset = None
    if args.model == 'gcn':
        model = GCN(dataset.features.shape[1], args.hidden, dataset.labels.max().item() + 1, dropout=args.dropout)
    elif args.model == 'gat':
        if args.debug:
            model = GAT(nfeat=dataset.features.shape[1],
                        nhid=args.hidden,
                        nclass=dataset.labels.max().item() + 1,
                        dropout=args.dropout,
                        nheads=args.nb_heads,
                        alpha=args.alpha)
        else:
            model = SpGAT(nfeat=dataset.features.shape[1],
                          nhid=args.hidden,
                          nclass=dataset.labels.max().item() + 1,
                          dropout=args.dropout,
                          nheads=args.nb_heads,
                          alpha=args.alpha)
    elif args.model == 'egat':
        model = ExtendSpGAT(nfeat=dataset.features.shape[1],
                            nhid=args.hidden,
                            nclass=dataset.labels.max().item() + 1,
                            dropout=args.dropout,
                            nheads=args.nb_heads,
                            alpha=args.alpha,
                            hop_num=args.hop_num,
                            hop_alpha=args.hop_alpha,
                            att_type=args.att_type)
    elif args.model == 'sgc':
        model = SGC(nfeat=dataset.features.shape[1],
                    nclass=dataset.labels.max().item() + 1)
    else:
        model = None
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda and not args.debug:
        device = torch.device('cuda:0')
        dataset.features = dataset.features.to(device)
        dataset.adj = dataset.adj.to(device)
        dataset.labels = dataset.labels.to(device)
        model = model.to(device)
    if args.model == 'sgc':
        dataset.features, _ = sgc_precompute(dataset.features, dataset.adj, args.degree)
    val = 0
    for epoch in range(args.epochs):
        if args.model == 'sgc':
            val = train_sgc(model, optimizer, dataset, args, epoch)
        elif args.model == 'gcn':
            val = train_gcn(model, optimizer, dataset, args, epoch)
        elif args.model == 'gat' or args.model == 'egat':
            val = train_gcn(model, optimizer, dataset, args, epoch)

    def compute_test():
        model.eval()
        if args.model == 'sgc':
            output = model(dataset.features)
        else:
            output = model(dataset.features, dataset.adj)
        acc_test = accuracy(output[dataset.idx_test], dataset.labels[dataset.idx_test])
        acc_class = accuracy_class(output[dataset.idx_test], dataset.labels[dataset.idx_test])
        print('eval',
              'acc_val: {:.4f}'.format(acc_test.item()))
        for acc in acc_class:
            print(acc)
        return acc_test

    return compute_test()


if __name__ == '__main__':
    main(args)
