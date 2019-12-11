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

import MoE
import optimizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train(dataset, task, args):
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
    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, 
                            args, task=task)#.to(device)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    test_acc_list = []
    loss_list = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            #batch = batch.to(device)
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
        loss_list.append(total_loss)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            print(test_acc,   '  test')
            test_acc = test(test_loader, model)
            test_acc_list.append(test_acc)
    return test_acc_list, loss_list

def test(loader, model, is_validation=True):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            #print('\n\nMASK!!!\n\n',data.val_mask.shape, sum(data.val_mask))
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

def main():
  n_epochs = 300
  period = 10
  model_list = ['GraphSage']
  for model_ in model_list:
    #args = {'model_type': model_, 'dataset': 'cora' , 'num_layers': 2, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.5, 'epochs': n_epochs, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01}
    args = {'model_type': model_, 'dataset': 'enzymes', 'num_layers': 2, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.0, 'epochs': n_epochs, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.001}
    #args = {'model_type': model_, 'dataset': 'pubmed', 'num_layers': 2, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.0, 'epochs': n_epochs, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.001}

    args = objectview(args)
    if args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        dataset = dataset.shuffle()
        task = 'graph'
    elif args.dataset == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        task = 'node'
    elif args.dataset == 'pubmed':
        dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
        task = 'node'
    test_acc_list, loss_list = train(dataset, task, args)
    plt.figure(0)
    plt.plot(range(int(n_epochs/period)), test_acc_list, label = model_+' acc')
    plt.legend()
    plt.figure(1)
    plt.plot(range(n_epochs), loss_list, label = model_+' loss')
    plt.legend()

if __name__ == '__main__':
    main()