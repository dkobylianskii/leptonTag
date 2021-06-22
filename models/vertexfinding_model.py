import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

import numpy as np
import torch
import torch.nn as nn

from models.mlp import MLP
from models.deepset import DeepSet
from models.set2graph import Set2Graph
from models.condensation import CondNet
from models.uutranspose import ObjectAssignment

class VertexFindingModel(nn.Module):
    def __init__(self,config):
        super(VertexFindingModel, self).__init__()

        if config['node embedding model']['model type']=='mlp':
            self.node_rep_net = MLP(config['node embedding model'])
        elif config['node embedding model']['model type']=='deepset':
            self.node_rep_net = DeepSet(config['node embedding model'])
        elif config['node embedding model']['model type'] == 'mpnn':
            pass
            
        if config['output model']['model type'] == 'set2graph':
            self.net = Set2Graph(config['output model'])
        elif config['output model']['model type'] == 'uutranspose':
            self.net = ObjectAssignment(config['output model'])
        elif config['output model']['model type'] == 'condensation':
            self.net = CondNet(config['output model'])
        elif config['output model'] == 'mlp':
            pass #self.net = EdgeMLP(config['output model'])
        

    def forward(self, g):

        g = self.node_rep_net(g)


        g = self.net(g)

        
        return g


    def predict(self,g):
        #print(g.device)
        with torch.no_grad():
            self(g)
            
        return self.net.predict(g)