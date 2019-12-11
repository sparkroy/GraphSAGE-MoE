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
    def __init__(self, input_dim, hidden_dim, output_dim, args, num_roles, task='node'):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.num_roles = num_roles
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim, num_roles))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim,num_roles))

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
            #print('layer ',i)
            x_roles = x[:,:self.num_roles]
            x_features = self.convs[i](x, edge_index) #-3 = +3
            #emb = x
            x_features = F.relu(x_features)
            x_features = F.dropout(x_features, p=self.dropout, training=self.training)
            x = torch.cat((x_roles, x_features),1)
        if self.task == 'graph':
            x = pyg_nn.global_max_pool(x, batch)

        x = x[:,self.num_roles:]
        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GraphSage(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, num_roles, reducer='mean', 
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')

        self.n_roles = num_roles
        #in_channels -= self.n_roles
        print('in_channels',in_channels)
        self.W_g = nn.Linear()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_channels, out_channels))
        for i in range(self.n_roles-1):
            self.linears.append(nn.Linear(in_channels, out_channels))

        self.agg_list = nn.ModuleList()
        self.agg_list.append(nn.Linear(in_channels + out_channels, out_channels))
        for i in range(self.n_roles-1):
            self.agg_list.append(nn.Linear(in_channels + out_channels, out_channels))

        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)

    def message(self, x_i, x_j, edge_index, size):
        roles = x_j[:,:self.n_roles]
        #roles = x_i[:,:self.n_roles]
        x_j = x_j[:, self.n_roles:]
        if self.n_roles==1:
            return F.relu(self.linears[0](x_j))
        x_j_interm = roles[:,(0,)]*self.linears[0](x_j)
        for r in range(self.n_roles - 1):
            x_j_interm += roles[:,(r,)]*self.linears[r](x_j)
        #x_j = F.relu(roles[:,(0,)]*self.lin1(x_j)+roles[:,(1,)]*self.lin2(x_j)+roles[:,(2,)]*self.lin3(x_j))
        x_j = F.relu(x_j_interm)

        #x_j = F.relu(self.lin(x_j)) # TODO
        #print(x_j.shape)
        return x_j

    def update(self, aggr_out, x):        
        x_features = x[:, self.n_roles:]
        roles = x[:, :self.n_roles]
        #print('updating...',x_features.shape,aggr_out.shape)
        cat = torch.cat((x_features, aggr_out),1)
        #print('catted',cat.shape)
        if self.n_roles==1:
            agg = self.agg_list[0](cat)
        else:
            agg = roles[:,(0,)]*self.agg_list[0](cat)
            for r in range(self.n_roles - 1):
                agg += roles[:,(r,)]*self.agg_list[r](cat)
        #print('agged',agg.shape)
        aggr_out = F.relu(agg) # TODO
        
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=1) # TODO

        return aggr_out

