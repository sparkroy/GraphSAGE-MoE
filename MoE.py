import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
from matplotlib import pyplot as plt

import torch.nn as nn
import torch.nn.functional as F


class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, task='node'):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))
        
        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.task = task
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = args.dropout
        self.num_layers = args.num_layers

    def build_conv_model(self, model_type):
        return GraphSage

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            #emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.task == 'graph':
            x = pyg_nn.global_max_pool(x, batch)

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GraphSage(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, reducer='mean', 
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')

        self.num_all_experts = 4
        self.k_experts = 4
        self.lin = nn.Linear(in_channels, out_channels)
        self.lins = nn.ModuleList([nn.Linear(in_channels, out_channels) for i in range(self.num_all_experts)]) # TODO
        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)
        self.agg_lins = nn.ModuleList([nn.Linear(in_channels + out_channels, out_channels) for i in range(self.num_all_experts)])
        self.W_g_msg = nn.Linear(in_channels, self.num_all_experts)
        self.W_g = nn.Linear(in_channels + out_channels, self.num_all_experts)
        self.W_1 = nn.Linear(in_channels , self.num_all_experts)
        self.W_2 = nn.Linear(out_channels , self.num_all_experts)
        self.W_noise = nn.Linear(in_channels + out_channels, self.num_all_experts)
        self.softmax = nn.Softmax(dim=1)
        self.plus = nn.Softplus()

        #nn.init.xavier_uniform(self.W_g.weight)
        #nn.init.xavier_uniform(self.W_noise.weight)

        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)

    def message(self, x_j, edge_index, size):
        '''
        Wg_msg_x = self.W_g_msg(x_j)

        weights = self.softmax(Wg_msg_x)
        lin_outs = []
        #expert_chosen = np.random.choice(len(self.agg_lins),size = self.k_experts,replace=False)
        #print(expert_chosen)
        for i, lin in enumerate(self.lins):
            #if i not in expert_chosen:
            #    continue
            weight_i = weights[:,(i,)]
            #if weight_i==0:
            #    continue
            lin_out_i = F.relu(lin(x_j))
            lin_outs.append(lin_out_i * weight_i)
        lin_out = sum(lin_outs)
        x_j = lin_out
        '''
        x_j = F.relu(self.lin(x_j))

        return x_j

    def update(self, aggr_out, x):

        x_cat = torch.cat((x, aggr_out), 1)
        Wg_xcat = self.W_g(x_cat)
        
        '''
        W1 = self.W_g_msg(x)
        W2 = self.W_2(aggr_out)
        Wg_xcat = W1 + W2
        '''
        '''
        Hx = Wg_xcat + np.random.standard_normal() * self.plus(self.W_noise(x_cat))
        topK = self.top_K(Hx, self.k_experts)
        weights = self.softmax(topK).reshape(-1)
        '''
        weights = self.softmax(Wg_xcat)
        
        aggr_outs = []
        #expert_chosen = np.random.choice(len(self.agg_lins),size = self.k_experts,replace=False)
        #print(expert_chosen)
        for i, agg_lin in enumerate(self.agg_lins):
            #if i not in expert_chosen:
            #    continue
            #print(i)
            weight_i = weights[:,(i,)]
            '''
            weight_i = weights[i]
            if weight_i==0:
                continue
            '''
            aggr_out_i = F.relu(agg_lin(x_cat))
            aggr_outs.append(aggr_out_i * weight_i)
        aggr_out = sum(aggr_outs)
        
        #aggr_out = F.relu(self.agg_lin(torch.cat((x, aggr_out),1)))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=1)
        return aggr_out

    def top_K(self, x, k):
        x = torch.mean(x, 0, keepdim=True)
        #print(x)
        topk, indices = torch.topk(x, k)
        mask = -np.inf*torch.ones(x.shape)#.to(device)
        #print(x,topk, indices, mask)
        mask = mask.scatter(1, indices, topk)
        return mask
