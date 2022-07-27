from statistics import mode
from models.model import GRACE, GNN, LogReg
from models.prompt import PromptedGAT
import torch
from datasets import load_dataset, load_diy_dataset, load_sbm_dataset
from utils import drop_feature, drop_feat_edge, set_determinatal, tsne
from tqdm import tqdm
from metric import accuracy
import numpy as np
# from dgl import RowFeatNormalizer
import argparse


def train(g, model, epochs=100, drop_feature_rate_1=0.3, drop_feature_rate_2=0.4, 
          drop_edge_rate_1=0.2, drop_edge_rate_2=0.4):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
    loop = tqdm(range(epochs))
    for i in loop:
        optimizer.zero_grad()
        
        g_1 = drop_feat_edge(g, drop_feature_rate_1, drop_edge_rate_1)
        g_2 = drop_feat_edge(g, drop_feature_rate_2, drop_edge_rate_2)
        
        l1_z1, l2_z1 = model(g_1, g_1.ndata['feat'])
        l1_z2, l2_z2 = model(g_2, g_2.ndata['feat'])
        loss1 = model.loss(l1_z1, l1_z2, batch_size=0)
        loss2 = model.loss(l2_z1, l2_z2, batch_size=0)
        loss = loss1*0.5+loss2*0.5
        loss.backward()
        optimizer.step()
        
        # acc = test(g, model)
        # print(acc)
    return model

def finetune(g, model):
    # print("GRACE", [p.shape for p in model.parameters()])
    model = model.encoder
    model.train()
    # print("GAT", [n for n, p in model.named_parameters()])
    xent = torch.nn.CrossEntropyLoss()
    log = LogReg(128, dataset.num_classes).to(device)
    # print(list(model.parameters()))
    # group parameters
    prompt_params = []
    encoder_params = []
    for n, p in model.named_parameters():
        if n in ['adapt_weights1', 'adapt_weights2']:
            # print(n, p.requires_grad)
            prompt_params += [p]
        else:
            encoder_params += [p]
    # params_id = list(map(id, prompt_params)) + list(map(id, encoder_params))
    # other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    opt = torch.optim.Adam([{'params': encoder_params, 'lr': 1e-4}, {'params': log.parameters(), 'lr': 1e-1}, {'params': prompt_params, 'lr': 1e-2}])
    
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']
    
    best_acc_from_val = 0
    best_val = 0
    best_t = 0
    for i in range(200):
        opt.zero_grad()
        
        _,z = model(g, g.ndata['feat'])
        train_embs = z[train_mask]
        val_embs = z[val_mask]
        test_embs = z[test_mask]
        
        logits = log(train_embs)
        loss = xent(logits, labels[train_mask].long())

        with torch.no_grad():
            ltra = log(train_embs)
            lv = log(val_embs)
            lt = log(test_embs)
            train_acc = accuracy(ltra, labels[train_mask])
            val_acc = accuracy(lv, labels[val_mask])
            test_acc = accuracy(lt, labels[test_mask])
            # print(test_acc)
            # print("EPOCH", i, "TRAIN:", train_acc.cpu().item(),"VAL: ", val_acc.cpu().item(), "TEST: ", test_acc.cpu().item())

            if val_acc > best_val:
                best_acc_from_val = test_acc
                best_val = val_acc
                best_t = i

        loss.backward()
        opt.step()
        
    return best_acc_from_val, best_val

def test(g, model):
    model.eval()
    xent = torch.nn.CrossEntropyLoss()

    z = model.embed(g, g.ndata['feat'])
    log = LogReg(128, dataset.num_classes).to(device)
    # log.reset_parameters()
    opt = torch.optim.Adam(log.parameters(), lr=1e-1, weight_decay=0.0)

    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']

    train_embs = z[train_mask]
    val_embs = z[val_mask]
    test_embs = z[test_mask]

    best_acc_from_val = 0
    best_val = 0
    best_t = 0

    log.train()
    for i in range(200):
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, labels[train_mask].long())

        with torch.no_grad():
            ltra = log(train_embs)
            lv = log(val_embs)
            lt = log(test_embs)
            train_acc = accuracy(ltra, labels[train_mask])
            val_acc = accuracy(lv, labels[val_mask])
            test_acc = accuracy(lt, labels[test_mask])
            # print(test_acc)
            # print("EPOCH", i, "TRAIN:", train_acc.cpu().item(),"VAL: ", val_acc.cpu().item(), "TEST: ", test_acc.cpu().item())

            if val_acc > best_val:
                best_acc_from_val = test_acc
                best_val = val_acc
                best_t = i

        loss.backward()
        opt.step()
    return best_acc_from_val, best_val
    
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--proj-hidden', type=int, default=128)
    args = parser.parse_args()
    
    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)
    
    return args

if __name__ == '__main__':
    set_determinatal(39788)
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = load_dataset(args.dataset)
    # dataset = load_diy_dataset(class_num=5, feat_dim=64, node_num=200, degree=5, homo=0.5, fea_num_per_class=4, train_ratio=0.1)
    g = dataset[0].to(device)
    
    # transform = RowFeatNormalizer(subtract_min=True,
    #                           node_feat_names=['feat'], edge_feat_names=[])
    # g = transform(g)
    
    # encoder = GNN(g.ndata['feat'].shape[1], args.hidden, args.hidden, gnn_type='GAT')
    # encoder = PromptedGAT(g.ndata['feat'].shape[1], args.hidden, num_classes=dataset.num_classes, prompt_len=0)
    # model = GRACE(encoder, args.hidden, args.proj_hidden, tau=0.4, learnable_proto=True, num_classes=dataset.num_classes).to(device)
    
    # model.load_state_dict(torch.load('last.pt'))
    # h = model.embed(g, g.ndata['feat'])
    # from utils import tsne
    # tsne(h.cpu(), g.ndata['label'].cpu(), 'cora.svg')
    test_acc_list = []
    cl_test_acc_list = []
    prompt_cl_test_acc_list = []
    for run in range(args.runs):
        # model.reset_parameters()
        # model.encoder.set_prompt_len(0) # disable prompt
        encoder = PromptedGAT(g.ndata['feat'].shape[1], args.hidden, num_classes=dataset.num_classes, prompt_len=0)
        model = GRACE(encoder, args.hidden, args.proj_hidden, tau=0.4, learnable_proto=True, num_classes=dataset.num_classes).to(device)
        model = train(g, model, epochs=args.epochs)
        # torch.save(model.state_dict(), "last.pt")
        
        test_acc, _ = finetune(g, model)
        test_acc_list.append(test_acc)
        # print(f"RUN {run} test acc:",test_acc)
        
        # model.load_state_dict(torch.load('last.pt', map_location=device))
        # model.train()
        # model.train_proto(g, epochs=200, finetune=True)
        # torch.save(model.state_dict(), "proto_last.pt")
        # proto = model.encoder.proto.cpu()
        # h = model.embed(g, g.ndata['feat'])
        # h = torch.cat((h.cpu(), proto))
        # labels = torch.cat((g.ndata['label'].cpu(), torch.tensor([111111,222222,333333333,4444444,5555555555,66666666,777777])))
        # tsne(h.detach(), labels, 'cora.svg')
        
        # cl_test_acc, _ = finetune(g, model)
        # cl_test_acc_list.append(cl_test_acc)
        
        # model.load_state_dict(torch.load('proto_last.pt'))
        model.encoder.set_prompt_len(dataset.num_classes)
        # use GAT parameters to initilize mlp, test the performance to verify whether the structure is important?
        # model.encoder.conv1
        # model.encoder.conv2
        # pro_test_acc, _ = finetune(g, model)
        # prompt_cl_test_acc_list.append(pro_test_acc)
        # print(f"RUN {run} test acc:", 'prompt acc:', pro_test_acc)
        
        # _, _, test_acc = model.predict(g)
        print("=====adapt weights===", model.encoder.adapt_weights1, model.encoder.adapt_weights2)
        # print(f"RUN {run} test acc: {test_acc} \t cl {cl_test_acc} \t prompt {pro_test_acc}")
        
    print(f"ft test acc: {round(np.mean(test_acc_list)*100, 2)} ± {round(np.std(test_acc_list)*100, 2)}")
    print(f"cl test acc: {round(np.mean(cl_test_acc_list)*100, 2)} ± {round(np.std(cl_test_acc_list)*100, 2)}")
    print(f"prompt test acc: {round(np.mean(prompt_cl_test_acc_list)*100, 2)} ± {round(np.std(prompt_cl_test_acc_list)*100, 2)}")
        