import json
import os

import matplotlib.pyplot as plt

default_param = {
    'alpha': 0.2,
    'dataset': 'cora',
    'degree': 2,
    'dropout': 0.5,
    'epoch': 200,
    'hidden': 8,
    'lr': 0.01,
    'nb_heads': 8,
    'seed': 42,
    'weight_decay': 5e-4,
    'att_type': 'li'
}


def quick(arr: list, left, r):
    if left < r:
        i = left
        j = r
        key = float(list(arr[left].keys())[0])
        while i < j:
            while i < j and float(list(arr[j].keys())[0]) >= key:
                j = j - 1
            while i < j and float(list(arr[i].keys())[0]) <= key:
                i = i + 1
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
        temp = arr[left]
        arr[left] = arr[i]
        arr[i] = temp
        quick(arr, left, i - 1)
        quick(arr, i + 1, r)


def get_keys(data):
    ans = []
    for ele in data:
        ans.append(list(ele.keys())[0])
    return ans


def get_data(prefix, dir='res/', dataset='cora'):
    files = os.listdir(dir)
    ans = []
    for file in files:
        if file.find(prefix) != -1 and file.find(dataset) != -1:
            with open('res/' + file, 'r') as f:
                result = json.load(f)['result']
            ele = {
                file.split('_')[-1][: -5]: result
            }
            ans.append(ele)
    with open(f'res/{dataset}_.json', 'r') as f:
        result = json.load(f)['result']
    ans.append({default_param[prefix]: result})
    quick(ans, 0, len(ans) - 1)
    return ans


parameter = 'nb_heads'
dataset = 'citeseer'
data = get_data(parameter, dataset=dataset)
x = get_keys(data)
gcn = []
gat = []
sgc = []
egat = []
for ele in data:
    for (key, value) in ele.items():
        gcn.append(value['gcn'])
        gat.append(value['gat'])
        sgc.append(value['sgc'])
        egat.append(value['egat'])

plt.plot(x, gcn, '.-', label='gcn')
plt.plot(x, gat, '.-', label='gat')
plt.plot(x, sgc, '.-', label='sgc')
plt.plot(x, egat, '.-', label='egat')
plt.title(parameter)
plt.legend()
plt.show()
