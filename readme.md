# 实验环境

windows11 python3.8

matplotlib==3.3.4

networkx==2.5

numpy==1.20.1

scikit_learn==1.0.2

scipy==1.6.2

torch==1.10.1

torch_geometric==2.0.3

torch_sparse==0.6.12

tqdm==4.59.0

visdom==0.1.8.9

# 数据集下载

cora数据集已提供，位于data中

citeseer运行时自动下载

# 运行方式

运行main.py可以根据参数进行训练

运行evaluate.py生成评估文件

运行ShowGraph显示图表

# 实验结果

core为egat选取最好的参数运行10次得到平均值，而citeseer为选取默认参数进行

| **方法** | **Cora**  | **Citeseer** |
| -------- | --------- | ------------ |
| **gcn**  | **82.41** | **67.8**     |
| **Gat**  | **82.48** | **66.9**     |
| **Sgc**  | **81.9**  | **65.2**     |
| **Egat** | **83.7**  | **67.4**     |

