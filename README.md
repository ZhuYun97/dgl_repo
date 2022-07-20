## Abstract
The aim of this repo is to build up my own easy tools for many tasks (e.g. node classification, link prediction, etc)

## Datasets
In `dataset.py` file, there are three main functions: 
1. `load_dataset`: load public datasets (e.g. Cora, Citeseer, Pubmed)
2. `load_sbm_dataset`: generate graphs(used for node classification) through SBM, the main arguments of this function are `homo, num_classes, nodes_per_class, feat_dim, train_ratio, feat_num_per_class, degree`
    - homo: specify the homophily of generated graph
    - num_classes: how many classes of nodes
    - nodes_per_class: each class contains how many nodes
    - feat_dim: the number of feature dimension of nodes
    - train_ratio: the ratio of training nodes
    - feat_num_per_class: specify the number of feature dimension to distinguish each class
    - degree: the degree of each node
3. `load_diy_dataset`: generate graphs(used for node classification).

## Tasks

### Node classifiaction
The commond is `CUDA_VISIBLE_DEVICES=0 python link_prediction.py`
You can specify the these args(e.g. dataset, gnn, etc) according to your needs.
### Link Prediction
The commond is `CUDA_VISIBLE_DEVICES=0 python link_prediction.py`
Also, you can specify the these args(e.g. dataset, gnn, etc) according to your needs.
### Contrastive Learning
GRACE is implemented, but the codes maybe a little confusing. Still organizing!
More illustrations will come soon!

## Utils
In `utils.py` file, there some commonly used tools:
1. `split_train_test_edges`: For link prediction task, you need to split existing edges into train/val/test edges and generate negative edges. This function will generate training graph, positive training graph(only contains positive training edges), negative training graph(only contains training negative edges), positive and negative testing graph.
2. `set_determinatal`: This function will set random seed for repeated experiments.
3. `drop_feature`, `drop_edge`, `drop_feat_edge`: This functions are used for augmentation.
4. `adj2edges`: This function will transform dense adj into edge index `[N,N]->[2, E]`

## Metrics
1. `compute_linkpred_binary_loss`: Computing the loss of link prediction.
2. `compute_linkpred_auc`: Used for calculating the auc of link prediction task.
3. `accuracy`: Computing the accuracy of node classification or other tasks.

## Models
1. `MLP`: For uniform usage, we change the `forward` function of MLP, althrough it does not make use of graph. 
2. `GNN`: You can specify the type of gnn (e.g. gcn, gat, sage)
3. `GRACE`: A contrastive model, more illustrations will come soon!

