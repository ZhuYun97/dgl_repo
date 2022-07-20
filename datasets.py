import dgl.data
import dgl
import networkx as nx
import torch
import random
import numpy as np
from utils import adj2edges


class DatasetWrapper():
    def __init__(self, g, num_classes) -> None:
        self.g = g
        self.num_classes = num_classes
    
    def __getitem__(self, idx):
        return self.g[idx]
    
    def __len__(self):
        return len(self.g)


def load_dataset(name='Cora'):
    assert name in ['Cora', 'Citeseer', 'Pubmed'], 'Only support Cpra, Citeseer, Pubmed datasets, but you can add other datasets freely'
    
    datasets = {
        'Cora': dgl.data.CoraGraphDataset,
        'Citeseer': dgl.data.CiteseerGraphDataset,
        'Pubmed': dgl.data.PubmedGraphDataset
    }
        
    return datasets[name]()


def load_sbm_dataset(homo, num_classes, nodes_per_class=100, feat_dim=64, train_ratio=0.1, feat_num_per_class=8, degree=5):
    node_num = nodes_per_class * num_classes
    sizes = np.ones(num_classes)*nodes_per_class
    sizes = sizes.astype(np.int64)
    probs = np.eye(num_classes)*homo
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            rest_ratio = 1 - probs[i][:j].sum()
            # assert rest_ratio < 0
            if rest_ratio <= 1e-8: # we will ignore the small remaining part
                break
            else:
                if j == num_classes-1:
                    probs[i][j] = probs[j][i] = rest_ratio
                else:
                    sampled_ratio = round(random.uniform(0, rest_ratio), 2)
                    if rest_ratio - sampled_ratio < 1e-2:
                        sampled_ratio = rest_ratio
                    # sampled_ratio = min(sampled_ratio, rest_ratio)
                    probs[i][j] = probs[j][i] = sampled_ratio
    print(probs) 
    probs = probs/nodes_per_class * degree
    nx_g = nx.stochastic_block_model(sizes=sizes, p=probs)
    dgl_g = dgl.from_networkx(nx_g) # keep the same format with other datasets
    # create features and labels
    all_feats = []
    all_labels = []
    for i in range(num_classes):
        feats = torch.normal(mean=i, std=1, size=(nodes_per_class, feat_dim))
        # feats = torch.rand(size=(nodes_per_class, feat_dim))
        # feats = torch.ones(size=(nodes_per_class, feat_dim))
        if i == 0:
            feats[:, i*feat_num_per_class:(i+1)*feat_num_per_class] = torch.normal(mean=1, std=1, size=(nodes_per_class, feat_num_per_class))
        else:
            shift = feat_num_per_class // 2
            feats[:, i*feat_num_per_class-shift:(i+1)*feat_num_per_class-shift] = torch.normal(mean=1, std=1, size=(nodes_per_class, feat_num_per_class))
        
        all_feats.append(feats)
        
        labels = torch.ones(nodes_per_class)*i
        all_labels.append(labels)
    all_feats = torch.cat(all_feats)
    dgl_g.ndata['feat'] = all_feats
    all_labels = torch.cat(all_labels).long()
    dgl_g.ndata['label'] = all_labels
    dgl_g.num_classes = num_classes
    # split into  train/val/test nodes
    val_num = train_num = int(nodes_per_class*train_ratio)
    assert train_num + val_num < nodes_per_class
    
    train_mask = []
    val_mask = []
    test_mask = []
    for i in range(num_classes):
        tmp_train_mask = torch.zeros(nodes_per_class)
        tmp_train_mask[:train_num] = 1
        train_mask.append(tmp_train_mask)
        
        tmp_val_mask = torch.zeros(nodes_per_class)
        tmp_val_mask[train_num:train_num+val_num] = 1
        val_mask.append(tmp_val_mask)
        
        tmp_test_mask = torch.zeros(nodes_per_class)
        tmp_test_mask[train_num+val_num:] = 1
        test_mask.append(tmp_test_mask)
    train_mask = torch.cat(train_mask)
    val_mask = torch.cat(val_mask)
    test_mask = torch.cat(test_mask)
    
    dgl_g.ndata['train_mask'] = train_mask.bool()
    dgl_g.ndata['val_mask'] = val_mask.bool()
    dgl_g.ndata['test_mask'] = test_mask.bool()
    
    dataset = DatasetWrapper([dgl_g], num_classes)
    return dataset


def load_diy_dataset(class_num, feat_dim=64, node_num=100, degree=5, homo=0.5, fea_num_per_class=4, train_ratio=0.1):
    assert class_num <= feat_dim/(fea_num_per_class//2), 'the number of classes is too large, you can diminish it or enlarge the feature dimension'
    assert node_num % class_num == 0, 'Each class should have the same number of nodes'
    labels = []
    adj = torch.zeros(size=(node_num, node_num))
    all_feas = []
    node_num_per_class = node_num//class_num
    
    val_num_per_class = train_num_per_class = int(node_num_per_class*train_ratio)
    train_mask = []
    val_mask = []
    test_mask = []
    
    for i in range(class_num):
        for node_idx in range(i*node_num_per_class, (i+1)*node_num_per_class):
            for d in range(degree):
                if np.random.rand()>homo:
                    flag = True
                    while flag:
                        j = np.random.randint(0, node_num) # choose from all nodes
                        flag = j in range(i*node_num_per_class, (i+1)*node_num_per_class) # resample if node j belongs to the same class
                else:
                    j = np.random.randint(i*node_num_per_class, (i+1)*node_num_per_class) # choose from node belongs to the same class
                adj[node_idx][j] = 1
                adj[j][node_idx] = 1
                # print(f"For node {node_idx} choose {j}")
            # adj[node_idx][node_idx] = 1 # add self-loop
        init_feas = torch.normal(mean=0, std=1, size=(node_num_per_class, feat_dim))
        if i == 0:
            fea_idx = torch.arange(i*fea_num_per_class, (i+1)*fea_num_per_class)
        else:
            shift = fea_num_per_class // 2
            fea_idx = torch.arange(i*fea_num_per_class-shift, (i+1)*fea_num_per_class-shift)
        # generate node features for each class
        valid_feas = torch.normal(mean=1, std=1, size=(node_num_per_class, fea_num_per_class))
        init_feas[:, fea_idx] = valid_feas
        all_feas.append(init_feas)
        # generate labels
        tmp_labels = torch.ones(node_num_per_class)*i
        labels.append(tmp_labels)
        # split nodes into train/val/test set
        train_mask_per_class = torch.zeros(node_num_per_class)
        train_mask_per_class[:train_num_per_class] = 1
        train_mask.append(train_mask_per_class)
        val_mask_per_class = torch.zeros(node_num_per_class)
        val_mask_per_class[train_num_per_class: train_num_per_class+val_num_per_class] = 1
        val_mask.append(val_mask_per_class)
        test_mask_per_class = torch.zeros(node_num_per_class)
        test_mask_per_class[train_num_per_class+val_num_per_class:]=1
        test_mask.append(test_mask_per_class)
    all_feas = torch.cat(all_feas)
    labels = torch.cat(labels).long()
    train_mask = torch.cat(train_mask).bool()
    val_mask = torch.cat(val_mask).bool()
    test_mask = torch.cat(test_mask).bool()
    edges = adj2edges(adj)
    # construct dgl graph
    dgl_g = dgl.graph((edges[0], edges[1]), num_nodes=node_num)
    dgl_g.ndata['feat'] = all_feas
    dgl_g.ndata['label'] = labels
    dgl_g.ndata['train_mask'] = train_mask
    dgl_g.ndata['val_mask'] = val_mask
    dgl_g.ndata['test_mask'] = test_mask
    
    dataset = DatasetWrapper([dgl_g], class_num)
    return dataset

