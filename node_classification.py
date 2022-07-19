from cProfile import label
import dgl
import torch
import torch.nn.functional as F
import dgl.data
from models import GNN, PromptedGAT
from utils import set_determinatal
from datasets import load_dataset, load_diy_dataset
import numpy as np
from tqdm import tqdm
from metric import accuracy
# from dgl.transforms import RowFeatNormalizer


# transform = RowFeatNormalizer(subtract_min=True,
#                               node_feat_names=['feat'], edge_feat_names=[])
set_determinatal(42)
# g = transform(g)

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    loop = tqdm(range(100))
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
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        loop.set_postfix_str('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
            e, loss, val_acc, best_val_acc, test_acc, best_test_acc))
        print('In epoch {}, loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
            e, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc))
    return best_test_acc
# model = GNN(g.ndata['feat'].shape[1], 64, dataset.num_classes, gnn_type='GAT').to(device)

if __name__ == '__main__':
    # dataset = load_dataset('Cora')
    dataset = load_diy_dataset(homo=1, num_classes=3, nodes_per_class=100, feat_dim=128, degree=10, train_ratio=0.1)
    print(dataset[0].ndata['feat'])
    print('Number of categories:', dataset[0].num_classes)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(dataset[0].ndata['label'])
    g = dataset[0].to(device)
    # model = PromptedGAT(g.ndata['feat'].shape[1], 128, dataset.num_classes, prompt_len=5).to(device)
    model = GNN(g.ndata['feat'].shape[1], 64, dataset[0].num_classes, gnn_type='GCN').to(device)
    
    test_acc_list = []
    for run in range(1):
        model.reset_parameters()
        test_acc = train(g, model)
        test_acc_list.append(test_acc)
    print(test_acc_list)
    print(f"test acc: {round(np.mean(test_acc_list)*100, 2)} ± {round(np.std(test_acc_list)*100, 2)}")