import networkx as nx
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn import manifold
from visdom import Visdom


def accuracy(output, labels):
    # 使用type_as(tesnor)将张量转换为给定类型的张量。
    preds = output.max(1)[1].type_as(labels)
    # 记录等于preds的label eq:equal
    correct = preds.eq(labels).double()
    correct_class = []
    for i in range(labels.max().item()):
        correct_class.append((correct[labels.eq(i).nonzero()].sum() / len(labels.eq(i))).item())
    correct = correct.sum()
    return correct / len(labels)


def accuracy_class(output, labels):
    # 使用type_as(tesnor)将张量转换为给定类型的张量。
    preds = output.max(1)[1].type_as(labels)
    # 记录等于preds的label eq:equal
    correct = preds.eq(labels).double()
    correct_class = []
    for i in range(labels.max().item()):
        correct_class.append((correct[labels.eq(i).nonzero()].sum() / len(labels.eq(i).nonzero())).item())
    return correct_class


def visualize(model, args, dataset):
    def test(model, dataset):
        model.eval()  # model转为测试模式
        output = model(dataset.features, dataset.adj)
        loss_test = F.nll_loss(output[dataset.idx_test], dataset.labels[dataset.idx_test])
        acc_test = accuracy(output[dataset.idx_test], dataset.labels[dataset.idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return output  # 可视化返回output

    # t-SNE 降维
    def t_SNE(output, dimention):
        # output:待降维的数据
        # dimention：降低到的维度
        tsne = manifold.TSNE(n_components=dimention, init='pca', random_state=0)
        result = tsne.fit_transform(output)
        return result

    # Visualization with visdom
    def Visualization(vis, result, labels, title):
        # vis: Visdom对象
        # result: 待显示的数据，这里为t_SNE()函数的输出
        # label: 待显示数据的标签
        # title: 标题
        vis.scatter(
            X=result,
            Y=labels + 1,  # 将label的最小值从0变为1，显示时label不可为0
            opts=dict(markersize=4, title=title),
        )

    # Testing
    output = test(model, dataset)  # 返回output

    # 计算预测值
    preds = output.max(1)[1].type_as(dataset.labels)

    # output的格式转换
    output = output.cpu().detach().numpy()
    labels = dataset.labels.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()

    # Visualization with visdom
    vis = Visdom(env='pyGCN Visualization', server='http://localhost', port=8097)

    # ground truth 可视化
    result_all_2d = t_SNE(output, 2)
    Visualization(vis, result_all_2d, labels,
                  title='[ground truth of all samples]\n Dimension reduction to %dD' % (result_all_2d.shape[1]))
    result_all_3d = t_SNE(output, 3)
    Visualization(vis, result_all_3d, labels,
                  title='[ground truth of all samples]\n Dimension reduction to %dD' % (result_all_3d.shape[1]))

    # 预测结果可视化
    result_test_2d = t_SNE(output[dataset.idx_test.cpu().detach().numpy()], 2)
    Visualization(vis, result_test_2d, preds[dataset.idx_test.cpu().detach().numpy()],
                  title='[prediction of test set]\n Dimension reduction to %dD' % (result_test_2d.shape[1]))
    result_test_3d = t_SNE(output[dataset.idx_test.cpu().detach().numpy()], 3)
    Visualization(vis, result_test_3d, preds[dataset.idx_test.cpu().detach().numpy()],
                  title='[prediction of test set]\n Dimension reduction to %dD' % (result_test_3d.shape[1]))


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
