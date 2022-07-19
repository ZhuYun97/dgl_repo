from datasets import load_dataset
import dgl
from models.model import GNN, DotPredictor
from models.prompt import PromptedGAT
from utils import split_train_test_edges, set_determinatal
import itertools
from metric import compute_linkpred_auc, compute_linkpred_auc, compute_linkpred_binary_loss
import torch
from tqdm import tqdm
import numpy as np


set_determinatal(42)

def train(train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, model, pred):
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=1e-2)
    # training encoder
    all_logits = []
    loop = tqdm(range(100))

    for i in loop:
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_linkpred_binary_loss(pos_score, neg_score)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 5 == 0:
            loop.set_postfix({'epoch': i, 'loss': loss.item()})
            
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        auc = compute_linkpred_auc(pos_score, neg_score)
        print('AUC', auc)
    return auc


# model = GNN(train_g.ndata['feat'].shape[1], 16, 16, gnn_type='GAT').to(device)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load graph data
    dataset = load_dataset('Cora')
    g = dataset[0]
    
    auc_list = []
    for run in range(20):
        # generate positive and negative edges
        train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = split_train_test_edges(g, 0.1)
        train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = train_g.to(device), train_pos_g.to(device), train_neg_g.to(device), test_pos_g.to(device), test_neg_g.to(device)
        # create models
        model = PromptedGAT(train_g.ndata['feat'].shape[1], 32, 32, prompt_len=5).to(device)
        # model = GNN(train_g.ndata['feat'].shape[1], 32, 32, gnn_type='GAT').to(device)
        pred = DotPredictor().to(device)
        auc = train(train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, model, pred)
        auc_list.append(auc)
    print(f"test auc: {round(np.mean(auc_list)*100, 2)} ± {round(np.std(auc_list)*100, 2)}")
