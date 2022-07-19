import torch
from models.model import GNN
from models.module import PromptedGATConv
import torch.nn as nn
import torch.nn.functional as F
import copy
            

class PromptedGAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, prompt_len, num_heads=1):
        super(PromptedGAT,  self).__init__()
        self.h_feats = h_feats
        self.conv1 = PromptedGATConv(in_feats, h_feats, num_heads)
        self.conv2 = PromptedGATConv(h_feats, h_feats, num_heads)
        
        # prompt_len = num_classes
        self.prompt_len = prompt_len
        self.num_classes = num_classes
        # if prompt_len != 0:
        #     self.prompt_len = prompt_len
        self.sources = torch.nn.Parameter(torch.ones(size=(num_classes*2, h_feats)))
        self.targets = torch.nn.Parameter(torch.ones(size=(num_classes*2, h_feats)))
        
        self.adapt_weights = torch.nn.Parameter(torch.ones(num_classes*2))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        torch.nn.init.constant_(self.adapt_weights, val=1.)
        proto_size = self.adapt_weights.shape[0] // 2
        self.adapt_weights.data[:proto_size] = self.adapt_weights.data[:proto_size]*1
        self.adapt_weights.data[proto_size:] = self.adapt_weights.data[proto_size:]*(-1)
        self.prompt_len = 0 # disable prompt
        
        # we should reinitialzie the prompts, but the train_proto function will overload prompts, so we do not need to reinitialzie them in such a situation
        # if self.prompt_len != 0:
            # torch.nn.init.normal_(self.sources, mean=0, std=0.001)
            # torch.nn.init.normal_(self.targets, mean=0, std=0.001)
            # torch.nn.init.uniform_(self.adapt_weights, a=0, b=1)
            # # torch.nn.init.constant_(self.adapt_weights, 1/self.prompt_len)
            # gain = nn.init.calculate_gain('relu')
            # nn.init.xavier_normal_(self.sources, gain=gain)
            # nn.init.xavier_normal_(self.targets, gain=gain)
            # self.adapt_weights = torch.nn.Parameter(torch.ones(self.prompt_len)/self.prompt_len)
            # torch.nn.init.normal_(self.proto, mean=0, std=0.001)
            # gain = nn.init.calculate_gain('relu')
            # nn.init.xavier_normal_(self.proto, gain=gain)
    
    def forward(self, g, feat):
        if self.prompt_len == 0:
            prompts = None
        else:
            prompts = [self.sources, self.targets, self.adapt_weights]
            # prompts = [self.proto, self.proto, self.adapt_weights]
        h = self.conv1(g, feat)
        h = F.relu(h)
        h = h.squeeze()
        h = self.conv2(g, h, prompts)
        h = F.relu(h)
        return h.squeeze()
    
    def set_prompt_len(self, prompt_len):
        self.prompt_len = prompt_len
    
    def initialize_prompts(self, proto, prompt_len=4):
        # self.prompt_len = prompt_len
        # init_sources = torch.normal(mean=0, std=0.001, size=(self.prompt_len, self.h_feats))
        # init_targets = torch.normal(mean=0, std=0.001, size=(self.prompt_len, self.h_feats))
        # self.sources = torch.nn.Parameter(init_sources).to(device) # different heads use the same prompts
        # self.targets = torch.nn.Parameter(init_targets).to(device)
        proto_size = proto.shape[0]
        perm_idx = torch.cat((torch.arange(1, proto_size), torch.tensor([0])))
        perm_proto = proto[perm_idx]
        self.sources.data = torch.cat((proto, proto))
        self.targets.data = torch.cat((proto, perm_proto))
        torch.nn.init.constant_(self.adapt_weights, val=0.1)
        self.adapt_weights.data[:proto_size] = self.adapt_weights.data[:proto_size]*1
        self.adapt_weights.data[proto_size:] = self.adapt_weights.data[proto_size:]*(-100)

        # if learnable_proto:
        #     self.proto = torch.nn.Parameter(torch.normal(mean=0, std=0.001, size=(self.prompt_len, self.h_feats))).to(device)
        #     gain = nn.init.calculate_gain('relu')
        #     nn.init.xavier_normal_(self.proto, gain=gain)
        
        # nn.init.xavier_normal_(self.sources, gain=gain)
        # nn.init.xavier_normal_(self.targets, gain=gain)
        