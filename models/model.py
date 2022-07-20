from turtle import forward
from dgl.nn import GraphConv, GATConv, SAGEConv, GINConv
import torch.nn.functional as F
import torch.nn as nn
import dgl.function as fn
import torch
from metric import accuracy


class MLP(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(MLP, self).__init__()
        
        self.mlp1 = nn.Linear(in_features=in_feats, out_features=h_feats)
        self.mlp2 = nn.Linear(in_features=h_feats, out_features=num_classes)
        
    def forward(self, g, x):
        h = F.relu(self.mlp1(x))
        h = self.mlp2(h)
        return h
    
    def reset_parameters(self):
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()


class GNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, gnn_type='GCN', num_heads=1, aggre="mean"):
        super(GNN, self).__init__()
        assert gnn_type in ['GCN', 'GAT', 'SAGE']

        self.conv1 = self.obtain_gnn(gnn_type, in_feats, h_feats, num_heads, aggre)
        self.conv2 = self.obtain_gnn(gnn_type, h_feats, num_classes, num_heads, aggre)
        

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h1 = F.relu(h)
        h2 = self.conv2(g, h1)
        h2 = F.relu(h2) # if you use this model as a classifier, you should remove this activation function
        # h2 = h2.squeeze()
        # print(h.shape)
        return h1.squeeze(), h2.squeeze()

    def obtain_gnn(self, gnn_type, in_feats, out_feats, num_heads, aggre):
        gnn_dict = {
            'GCN': GraphConv(in_feats, out_feats, allow_zero_in_degree=True),
            'GAT': GATConv(in_feats, out_feats, num_heads, allow_zero_in_degree=True),
            'SAGE': SAGEConv(in_feats, out_feats, aggre)
        }
        return gnn_dict[gnn_type]
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]
        

class GRACE(nn.Module):
    def __init__(self, encoder: GNN, num_hidden: int, num_proj_hidden: int, num_classes: int, tau: float=0.5, learnable_proto=False):
        super(GRACE, self).__init__()
        self.encoder: GNN = encoder
        self.num_hidden: int = num_hidden
        self.num_proj_hidden: int = num_proj_hidden
        self.tau: float = tau
        self.num_classes = num_classes
        
        self.projector1 = torch.nn.Sequential(torch.nn.Linear(num_hidden, num_proj_hidden),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(num_proj_hidden, num_hidden))
        self.projector2 = torch.nn.Sequential(torch.nn.Linear(num_hidden, num_proj_hidden),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(num_proj_hidden, num_hidden))
        self.learnable_proto = learnable_proto
        self.init_proto()
        
    def forward(self, g, feat):
        return self.encoder(g, feat)
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    
    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0, layer: int = 1):
        # h1 = self.projection(z1)
        # h2 = self.projection(z2)
        assert layer in [1,2]
        if layer == 1:
            h1 = self.projector1(z1)
            h2 = self.projector1(z2)
        else:
            h1 = self.projector2(z1)
            h2 = self.projector2(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    
    def embed(self, g, feat):
        z = self.encoder(g, feat)
        if isinstance(z, tuple):
            _, z = z
        return z.detach()
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        for m in self.projector1.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        for m in self.projector2.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
                
        if self.learnable_proto:
            self.proto1.data.normal_(mean=0, std=0.1)
            torch.nn.init.kaiming_normal_(self.proto1, a=0, mode='fan_in', nonlinearity='relu')
            self.proto2.data.normal_(mean=0, std=0.1)
            torch.nn.init.kaiming_normal_(self.proto2, a=0, mode='fan_in', nonlinearity='relu')
                
    @staticmethod
    def pcl_sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1,0))

    def pcl_loss(self, v_ins, layer=2):
        assert layer in [1,2]
        
        loss = 0.
        num = v_ins.shape[0] # C, num, D
        if self.learnable_proto:
            # instance-prototype loss
            sim_mat = torch.exp(self.pcl_sim(v_ins, self.proto1)) if layer == 1 else torch.exp(self.pcl_sim(v_ins, self.proto2))
            num = sim_mat.shape[1]
            
            for i in range(num):
                pos_score = torch.diag(sim_mat[:,i,:])
                neg_score = (sim_mat[:,i,:].sum(1) - pos_score)
                loss += - torch.log(pos_score / (pos_score + neg_score)).sum()
            loss = loss / (num * self.num_classes * self.num_classes)
            # print("ins-pro", loss)

        # instance-instance loss
        loss_ins = 0.
        for i in range(v_ins.shape[0]):
            sim_instance = torch.exp(self.pcl_sim(v_ins, v_ins[i]))
            pos_ins = sim_instance[i]
            neg_ins = (sim_instance.sum(0) - pos_ins).sum(0)
            loss_ins += - torch.log(pos_ins / (pos_ins + neg_ins)).sum()
        loss_ins = loss_ins / (num * self.num_classes * num * self.num_classes)
        # print("ins-ins", loss_ins)
        loss = loss + loss_ins

        return loss

    def init_proto(self):
        if self.learnable_proto:
            self.proto1 = torch.nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.num_classes, self.num_hidden)))
            # self.proto.requires_grad = True
            torch.nn.init.kaiming_normal_(self.proto1, a=0, mode='fan_in', nonlinearity='relu')
            self.proto2 = torch.nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.num_classes, self.num_hidden)))
            # self.proto.requires_grad = True
            torch.nn.init.kaiming_normal_(self.proto2, a=0, mode='fan_in', nonlinearity='relu')

    def train_proto(self, data, epochs=200, finetune=False):
        self.train()
        # print("GRACE train proto", [n for n, p in self.named_parameters()])
        
        if self.learnable_proto: # prototypes are learnable vectors
            if finetune:
                optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) # we will finetune the whole model
            else:
                optimizer = torch.optim.Adam([{'params': self.projector.parameters(), 'lr': 1e-4}, {'params': self.proto, 'lr': 1e-4}])
        else: # prototypes are the mean values of reps which belongs to the same class
            if finetune:
                optimizer = torch.optim.Adam([{'params': self.parameters(), 'lr': 1e-4}])
            else:
                optimizer = torch.optim.Adam([{'params': self.projector.parameters(), 'lr': 1e-4}])
        
        loss = 0.
        best_val = 0
        test_acc_from_val = 0
        for epoch in range(epochs):
            # obtain node representations
            embeds1 = []
            embeds2 = []
            if finetune:
                outputs1, outputs2 = self(data, data.ndata['feat'])
            else:
                with torch.no_grad():
                    outputs1, outputs2 = self.embed(data, data.ndata['feat'])
            train_mask = data.ndata['train_mask']
            masked_train_labels = torch.where(train_mask == 1, data.ndata['label'], -1) # because the train_mask is for all nodes, we should use the train labels for all nodes(nodes have label -1 if they do no belong to trainset)
            zeros = torch.zeros_like(train_mask)
            for c in range(self.num_classes):
                train_mask_class_c = torch.where(masked_train_labels==c, train_mask, zeros)
                embeds1.append(outputs1[train_mask_class_c])
                embeds2.append(outputs2[train_mask_class_c])
            
            embeds1 = torch.stack(embeds1)
            embeds2 = torch.stack(embeds2)
            x1 = self.projector1(embeds1)
            x2 = self.projector2(embeds2)
            optimizer.zero_grad()
            loss1 = self.pcl_loss(x1, layer=1)
            loss2 = self.pcl_loss(x2, layer=2)
            loss = 0.5*loss1 +  0.5*loss2
            loss.backward()
            optimizer.step()
        
        if not self.learnable_proto:
            self.proto1 = embeds1.mean(1)
            self.proto2 = embeds2.mean(1)
        # we shold transfer the prototypes to encoder
        self.encoder.initialize_prompts(self.proto1.detach(), self.proto2.detach(), self.num_classes)

        print("Total epoch: {}. ProtoVerb loss: {}".format(epochs, loss))
        return test_acc_from_val
    
    
    @torch.no_grad()
    def predict(self, data):
        train_mask = data.ndata['train_mask']
        val_mask = data.ndata['val_mask']
        test_mask = data.ndata['test_mask']
        labels = data.ndata['label']
        proto_logits = self.pcl_sim(self.projector(self.encoder(data, data.ndata['feat'])), self.encoder.proto)
        train_logits = proto_logits[train_mask]
        val_logits = proto_logits[val_mask]
        test_logits = proto_logits[test_mask]
        train_acc = accuracy(train_logits, labels[train_mask])
        val_acc = accuracy(val_logits, labels[val_mask])
        test_acc = accuracy(test_logits, labels[test_mask])
        return train_acc, val_acc, test_acc
    
    
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret