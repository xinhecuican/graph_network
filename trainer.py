import torch.nn.functional as F


def accuracy(output, labels):
    # 使用type_as(tesnor)将张量转换为给定类型的张量。
    preds = output.max(1)[1].type_as(labels)
    # 记录等于preds的label eq:equal
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train_gcn(model, optimizer, dataset, args, epoch):
    model.train()
    optimizer.zero_grad()

    output = model(dataset.features, dataset.adj)
    loss_train = F.cross_entropy(output[dataset.idx_train], dataset.labels[dataset.idx_train])
    acc_train = accuracy(output[dataset.idx_train], dataset.labels[dataset.idx_train])
    loss_train.backward()
    optimizer.step()
    if not args.fastmode:
        model.eval()
        output = model(dataset.features, dataset.adj)

    loss_val = F.nll_loss(output[dataset.idx_val], dataset.labels[dataset.idx_val])
    acc_val = accuracy(output[dataset.idx_val], dataset.labels[dataset.idx_val])

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))
    return acc_val.item()


def train_sgc(model, optimizer, dataset, args, epoch):
    model.train()
    optimizer.zero_grad()
    output = model(dataset.features)
    loss_train = F.cross_entropy(output[dataset.idx_train], dataset.labels[dataset.idx_train])
    loss_train.backward()
    optimizer.step()

    acc_train = accuracy(output[dataset.idx_train], dataset.labels[dataset.idx_train])
    if not args.fastmode:
        model.eval()
        output = model(dataset.features)

    loss_val = F.nll_loss(output[dataset.idx_val], dataset.labels[dataset.idx_val])
    acc_val = accuracy(output[dataset.idx_val], dataset.labels[dataset.idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))
    return acc_val.item()
