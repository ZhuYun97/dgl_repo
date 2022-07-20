from cProfile import label
import dgl
import torch
import torch.nn.functional as F
import dgl.data
from models import GNN, PromptedGAT, MLP
from utils import set_determinatal
from datasets import load_dataset, load_sbm_dataset, load_diy_dataset
import numpy as np
from tqdm import tqdm
from metric import accuracy
import argparse
# from dgl.transforms import RowFeatNormalizer


# transform = RowFeatNormalizer(subtract_min=True,
#                               node_feat_names=['feat'], edge_feat_names=[])
# g = transform(g)

def train(g, model, epochs=200, lr=5e-4, weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    loop = tqdm(range(epochs))
    for e in loop:
        # Forward
        logits = model(g, features)
        # print(logits)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = accuracy(logits[train_mask], labels[train_mask])
        val_acc = accuracy(logits[val_mask], labels[val_mask])
        test_acc = accuracy(logits[test_mask], labels[test_mask])

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc <= val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        loop.set_postfix_str('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
            e, loss, val_acc, best_val_acc, test_acc, best_test_acc))
        # print('In epoch {}, loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
        #     e, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc))
    return best_test_acc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Cora", help='Choose dataset [Cora, Citeseer, Pubmed]')
    parser.add_argument('--gnn-type', type=str, default="GCN", help='Specify the kind of GNN [GCN, GAT, SAGE]')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--runs', type=int, default=10, help='How many runs of experiments')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions. (For GAT)')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    set_determinatal(args.seed)
    dataset = load_dataset(args.dataset)
    # dataset = load_sbm_dataset(homo=0.1, num_classes=5, nodes_per_class=100, feat_dim=128, feat_num_per_class=4, degree=10, train_ratio=0.1)
    # dataset = load_diy_dataset(class_num=5, feat_dim=64, node_num=1000, degree=5, homo=0.1, fea_num_per_class=4, train_ratio=0.1)

    print('Number of categories:', dataset.num_classes)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = dataset[0].to(device)
    model = GNN(g.ndata['feat'].shape[1], args.hidden, dataset.num_classes, gnn_type=args.gnn_type, num_heads=args.nb_heads).to(device)
    # model = MLP(g.ndata['feat'].shape[1], 64, dataset.num_classes).to(device)
    
    test_acc_list = []
    for run in range(args.runs):
        model.reset_parameters()
        test_acc = train(g, model, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)
        test_acc_list.append(test_acc)
    print(test_acc_list)
    print(f"test acc: {round(np.mean(test_acc_list)*100, 2)} ± {round(np.std(test_acc_list)*100, 2)}")