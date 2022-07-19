from regex import F
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def compute_linkpred_binary_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0], device=pos_score.device), torch.zeros(neg_score.shape[0], device=neg_score.device)])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_linkpred_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    print('p', preds)
    print('l', labels)
    acc = (preds == labels).sum() / labels.shape[0]
    return acc.item()