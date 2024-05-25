# GML_learning

接下来开始在OpenHGNN上复现HGMAE模型

## 2024.5.24

看了一遍openhgnn的pipeline，自己实现了RGCN。读完论文，了解了HGMAE模型基本流程。正在看论文官方源码。

## 2024.5.17

了解了dgl异质图消息传递过程，实现了自定义的messgefunc，reducedfunc，apply_edge

## 2024.5.10

完成week7，在DGL上实现节点和链接分类demo：[week7-DGL](./DGL_learning/week7-DGL.ipynb)，在PYG上实现链接和图分类demo：[week7-PYG](./PYG_learning/week7-PYG.ipynb)

了解了HAN和RGCN原理，看了眼GammaGL和OpenHGNN里HAN源代码。GammaGL中使用TensorLayerX来搭建模型，它的backend可以是torch，tensorflow，paddle。tensorlayerx用起来还是与torch有相似之处的，比如tlx.nn.Linear之类的

- [x] HAN

- [x] GGL

## 2024.5.3

学习了PYG和DGL中自定义消息传递，聚合。

笔记 [DGL中的消息传递函数和聚合函数](./DGL_learning/DGL中的消息传递函数和聚合函数.md)

笔记 [PYG中的消息传递和聚合函数](./PYG_learning/PYG中的消息传递和聚合函数.md)

完成了DGL实现GraphConv, GATConv, SAGEConv [code](./DGL_learning/week6-DGL实现GraphConv,GATConv,SAGEConv.ipynb)

完成了PYG实现GraphConv, GATConv, SAGEConv [code](./PYG_learning/week6-PYG实现GraphConv,GATConv,SAGEConv.ipynb)

## 2024.4.26

在DGL官网上跟着[A Blitz Introduction to DGL](https://docs.dgl.ai/tutorials/blitz/index.html)做完了6个chapter，了解了DGL基本用法，包括如何自定义模型，mess_func和reduce_func。用DGL实现GraphConv, GATConv, SAGEConv，即[week5](https://colab.research.google.com/drive/1xSMe9xdEN6EziexnhnYghEXIlPbTC9B5?usp=sharing)和[week6](https://colab.research.google.com/drive/1xSMe9xdEN6EziexnhnYghEXIlPbTC9B5?usp=sharing)(week6的GAT和SAGE还正在写)

下周计划过一下PYG基本用法，搭建模型，用实现GraphConv, GATConv, SAGEConv

## 2024.4.19

过了一遍pytorch基本用法，搭建了一个UNet网络跑了一下。

看了deepwalk论文，了解了GCN，GAT，GraphSAGE实现原理
