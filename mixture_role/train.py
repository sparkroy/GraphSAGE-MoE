import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CoraFull
import torch_geometric.datasets as tds

from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn


from graphrole import RecursiveFeatureExtractor, RoleExtractor

from matplotlib import pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

import MoE_role
import optimizer

def extract_role(G, n_roles=None):
    G=pyg_utils.to_networkx(G)
    feature_extractor = RecursiveFeatureExtractor((G))
    features = feature_extractor.extract_features()
    role_extractor = RoleExtractor(n_roles)
    role_extractor.extract_role_factors(features)
    return role_extractor.role_percentage

def train(dataset, task, args,num_roles):
  
    if task == 'graph':
        # graph classification: separate dataloader for test set
        data_size = len(dataset)
        loader = DataLoader(
                dataset[:int(data_size * 0.8)], batch_size=args.batch_size, shuffle=True)
        
        test_loader = DataLoader(
                dataset[int(data_size * 0.8):], batch_size=args.batch_size, shuffle=True)
    elif task == 'node':
        # use mask to split train/validation/test
        test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        raise RuntimeError('Unknown task')

    # build model
    model = GNNStack(dataset.num_node_features-num_roles, args.hidden_dim, dataset.num_classes, 
                            args, num_roles, task=task)
    print('dsf:', dataset.num_node_features-num_roles)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    test_acc_list = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        print(total_loss)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            print(test_acc,   '  test')
            test_acc = test(test_loader, model)
            test_acc_list.append(test_acc)
    return test_acc_list

def test(loader, model, is_validation=True):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            #print('\n\nMASK!!!\n\n',data.val_mask.shape, sum(data.val_mask))
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.val_mask if is_validation else data.test_mask).item() #torch.sum(data.test_mask).item()
    return correct / total
  
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

n_epochs = 500
period = 10
model_='GraphSage'

args = {'model_type': model_, 'dataset': 'cora' , 'num_layers': 2, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.5, 'epochs': n_epochs, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01}
#args = {'model_type': model_, 'dataset': 'enzymes', 'num_layers': 2, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.0, 'epochs': n_epochs, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.001}

args = objectview(args)
if args.dataset == 'enzymes':
    #dataset = TUDataset(root='/tmp/AIDS', name='AIDS')
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    dataset = dataset.shuffle()
    task = 'graph'
elif args.dataset == 'cora':
    ###dataset = tds.Amazon(root='/tmp/Amazon',name='photo')
    #dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    task = 'node'
##############
num_roles = 4

print(dataset.data.x.shape)
print('extracting roles...')
if num_roles == 1:
    roles = np.ones((dataset.data.x.shape[0],1))
    dataset.data.x = torch.cat((torch.from_numpy(roles).float(), dataset.data.x), 1)
else:
    if args.dataset == 'cora':
        roles = extract_role(dataset.data, num_roles)
        #roles.to_csv('r.csv',index=False)
        #print(roles)
        #roles=pd.read_csv('r.csv')
        #print(roles)
        print('extracted roles!')
        roles = torch.from_numpy(roles.values).float()
        dataset.data.x = torch.cat((roles, dataset.data.x), 1)
    else:
        for i in range(len(dataset)):
            print(i)
            roles = extract_role(dataset[i], num_roles)
            roles = torch.from_numpy(roles.values)
            dataset[i].x = torch.cat((roles.float(), dataset[i].x), 1)

print(dataset.data.x.shape)
print('training start!')
##############
test_acc_list = train(dataset, task, args, num_roles)
plt.plot(range(int(n_epochs/period)), test_acc_list, label = model_)
plt.legend()
