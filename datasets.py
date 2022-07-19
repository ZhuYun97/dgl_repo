import dgl.data
import dgl
import networkx as nx
import torch
import random
import numpy as np


def load_dataset(name='Cora'):
    assert name in ['Cora', 'Citeseer', 'Pubmed', 'SBM']
    
    if name == 'Cora':
        dataset = dgl.data.CoraGraphDataset()
    elif name == 'Citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
        
    if name == 'SBM':
        # sizes = [100, 100, 100, 100, 100]
        # probs = [[1, 0, 0, 0], 
        #          [0, 1, 0, 0], 
        #          [0, 0, 1, 0], 
        #          [0, 0, 0, 1]]
        # sizes = [3, 3, 3]
        # probs = [[0.5, 0.0, 0.5], 
        #          [0.0, 0.3, 0.5], 
        #          [0.5, 0.5, 0.2]]
        nx_g = nx.stochastic_block_model(sizes=sizes, p=probs)
        dataset = [dgl.from_networkx(nx_g)] # keep the same format with other datasets
    return dataset


def load_diy_dataset(homo, num_classes, nodes_per_class=100, feat_dim=64, train_ratio=0.1, feat_num_per_class=8, degree=5):
    node_num = nodes_per_class * num_classes
    sizes = np.ones(num_classes)*nodes_per_class
    sizes = sizes.astype(np.int64)
    probs = np.eye(num_classes)*homo
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            rest_ratio = 1 - probs[i][:j].sum()
            # assert rest_ratio < 0
            if rest_ratio <= 0:
                break
            else:
                if j == num_classes-1:
                    probs[i][j] = probs[j][i] = rest_ratio
                else:
                    sampled_ratio = round(random.uniform(0, rest_ratio), 2)
                    sampled_ratio = min(sampled_ratio, rest_ratio)
                    probs[i][j] = probs[j][i] = sampled_ratio
    print(probs) 
    probs = probs/(nodes_per_class/degree)
    nx_g = nx.stochastic_block_model(sizes=sizes, p=probs)
    dgl_g = dgl.from_networkx(nx_g) # keep the same format with other datasets
    # create features and labels
    all_feats = []
    all_labels = []
    for i in range(num_classes):
        feats = torch.normal(mean=0, std=0.001, size=(nodes_per_class, feat_dim))
        feats[:, i*feat_num_per_class:(i+1)*feat_num_per_class] = 1
        # if i == 0:
        #     feats[:, i*feat_num_per_class:(i+1)*feat_num_per_class] = torch.normal(mean=1, std=0.1, size=(nodes_per_class, feat_num_per_class))
        # else:
        #     shift = feat_num_per_class // 2
        #     feats[:, i*feat_num_per_class-shift:(i+1)*feat_num_per_class-shift] = torch.normal(mean=1, std=1, size=(nodes_per_class, feat_num_per_class))
        
        all_feats.append(feats)
        
        labels = torch.ones(nodes_per_class)*i
        all_labels.append(labels)
    all_feats = torch.cat(all_feats)
    dgl_g.ndata['feat'] = all_feats
    all_labels = torch.cat(all_labels).long()
    # perm_idx = torch.randperm(node_num)
    # all_labels = all_labels[perm_idx]
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
    
    return [dgl_g]