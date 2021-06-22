import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

import numpy as np
import torch
import torch.nn as nn


def build_layers(inputsize,outputsize,features):
    layers = []
    layers.append(nn.Linear(inputsize,features[0]))
    for hidden_i in range(1,len(features)):
        layers.append(nn.Linear(features[hidden_i-1],features[hidden_i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(features[-1],outputsize))
    return layers


class MLP(nn.Module):
    def __init__(self,config):
        super(MLP, self).__init__()

        self.config = config

        self.mlp = build_layers(config['inputsize'], config['outputsize'],config['layers'])

        
    def forward(self, g):

        node_data = torch.cat(
            [g.ndata[inputname] for inputname in self.config['inputs']],dim=1)
        
        g.ndata[self.config['output']] = self.mlp(node_data)
    
        return g


