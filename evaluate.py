import argparse
import json

import torch

from main import main

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during traing pass')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate')
# 权重衰减
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters)')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability)')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--model', default='gcn')
parser.add_argument('--dataset', default='cora')
# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
default_param = {
    'alpha': 0.2,
    'dataset': args.dataset,
    'degree': 2,
    'dropout': 0.5,
    'epoch': 200,
    'hidden': 16,
    'lr': 0.01,
    'nb_heads': 8,
    'seed': 42,
    'weight_decay': 5e-4
}


def write_ans(args, ans):
    dict = {
        'args': {
            'alpha': args.alpha,
            'dataset': args.dataset,
            'degree': args.degree,
            'dropout': args.dropout,
            'epoch': args.epochs,
            'hidden': args.hidden,
            'lr': args.lr,
            'nb_heads': args.nb_heads,
            'seed': args.seed,
            'weight_decay': args.weight_decay
        },
        'result': ans}
    path = "res/" + args.dataset + "_"
    for (key, value) in dict['args'].items():
        if default_param[key] != value:
            path += key + "_" + str(value)
    path += '.json'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dict))


ans = {}
run_time = 5
for model in ['gcn', 'gat', 'sgc']:
    args.model = model
    tmp_res = 0
    for i in range(run_time):
        res = main(args)
        tmp_res += res
    ans[model] = float(tmp_res / run_time)

write_ans(args, ans)
