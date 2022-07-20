import numpy as np
import scipy.sparse as sp
import dgl
import torch
import random
from torch.distributions import Bernoulli
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


'''
split existing edges into training and test positive edges and
generate negative edges from non-existing edges
'''
def split_train_test_edges(g, ratio=0.1):
    u,v = g.edges()
    
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids)*ratio)
    train_size = g.number_of_edges() - test_size
    
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    
    # collect negative edges
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0) # if adj already contains self-loop, the diagnoal elements will be minus value. In this situation, this condition(adj_neg != 0) will have problems.
    
    neg_ids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_ids[:test_size]], neg_v[neg_ids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_ids[test_size:]], neg_v[neg_ids[test_size:]]
    
    # In order to avoid information leak, we should remove test edges during training.
    train_g = dgl.remove_edges(g, eids[:test_size])
    
    # obtain positive and negative graphs for latter simple usage
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
    
    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g

'''
set repeated experiments
'''
def set_determinatal(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True

'''
augmentation method: randomly masking some features
'''
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

'''
augmentation method: randomly masking some edges
'''
def drop_edge(g, p=0):
    if p == 0:
        return g
    dist = Bernoulli(p)
    for c_etype in g.canonical_etypes:
        samples = dist.sample(torch.Size([g.num_edges(c_etype)]))
        eids_to_remove = g.edges(form='eid', etype=c_etype)[samples.bool().to(g.device)]
        g = g.clone()
        g.remove_edges(eids_to_remove, etype=c_etype)
    return g

'''
augmentation method: randomly masking some edges and features
'''
def drop_feat_edge(g, d_feat=0.2, d_edge=0.2):
    # do not modify the original graph
    x = g.ndata['feat']
    masked_x = drop_feature(x, drop_prob=d_feat)
    
    aug_g = drop_edge(g, p=d_edge)
    aug_g.ndata['feat'] = masked_x
    
    return aug_g
                      
def tsne(h, color, fig_path):
    z = TSNE(n_components=2, perplexity=50, n_iter=2000, learning_rate=200).fit_transform(h.numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    # plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    z_min, z_max = z.min(0), z.max(0)
    z_norm = (z-z_min)/(z_max-z_min)
    for i in range(z.shape[0]):
        c = int(color[i].item())
        plt.text(z_norm[i, 0], z_norm[i, 1], str(c), color=plt.cm.Set1(c), 
             fontdict={'weight': 'bold', 'size': 9})
    plt.savefig(fig_path, format='svg')
    
'''transform dense adj into edge index [N,N]->[2, E]'''
def adj2edges(adj):
    assert len(adj.shape) ==2 and adj.shape[0] == adj.shape[1]
    assert adj.max() == 1 and adj.min() == 0
    
    node_num = adj.shape[0]
    
    source_idx = []
    target_idx = []
    for i in range(node_num):
        for j in range(node_num):
            if adj[i][j] == 1:
                source_idx.append(i)
                target_idx.append(j)
    return torch.tensor([source_idx, target_idx])