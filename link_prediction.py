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
import argparse


def train(train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, model, pred, epochs=100):
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=1e-2)
    # training encoder
    all_logits = []
    loop = tqdm(range(args.epochs))

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Cora", help='Choose dataset [Cora, Citeseer, Pubmed]')
    parser.add_argument('--gnn-type', type=str, default="GCN", help='Specify the kind of GNN [GCN, GAT, SAGE]')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--runs', type=int, default=10, help='How many runs of experiments')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions. (For GAT)')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--train-ratio', type=float, default=0.1, help='The training ratio of edges')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    set_determinatal(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load graph data
    dataset = load_dataset(args.dataset)
    g = dataset[0]
    
    auc_list = []
    for run in range(args.runs):
        # generate positive and negative edges
        train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = split_train_test_edges(g, args.train_ratio)
        train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = train_g.to(device), train_pos_g.to(device), train_neg_g.to(device), test_pos_g.to(device), test_neg_g.to(device)
        # create models
        # model = PromptedGAT(train_g.ndata['feat'].shape[1], 32, 32, prompt_len=5).to(device)
        model = GNN(train_g.ndata['feat'].shape[1], args.hidden, args.hidden, gnn_type=args.gnn_type, num_heads=args.nb_heads).to(device)
        pred = DotPredictor().to(device)
        auc = train(train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, model, pred, epochs=args.epochs)
        auc_list.append(auc)
    print(f"test auc: {round(np.mean(auc_list)*100, 2)} ± {round(np.std(auc_list)*100, 2)}")
