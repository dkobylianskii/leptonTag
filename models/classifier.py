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
    return nn.Sequential(*layers)


class JetEdgeNetwork(nn.Module):
    def __init__(self,inputsize,outputsize):
        super(JetEdgeNetwork, self).__init__()

        self.net = nn.Sequential( nn.Linear(inputsize,outputsize,bias=True),nn.Tanh() )
        
    
    def forward(self,edges):

        input_data = torch.cat( [ torch.sigmoid(edges.data['edge prediction']).unsqueeze(1).float() ,edges.src['node hidden rep'],
            torch.argmax( edges.src['node prediction'] ,dim=1).unsqueeze(1).float(),
                                   edges.dst['node hidden rep'],torch.argmax( edges.dst['node prediction'] ,dim=1).unsqueeze(1).float(),
                                   edges.dst['node_type_embedding'],
                                   edges.src['node_type_embedding'],
                                  edges.dst['mean node hidden rep']] ,dim=1)
        
        return {'edge message' : self.net(input_data)}


class JetNodeNetwork(nn.Module):
    def __init__(self,inputsize,outputsize):
        super(JetNodeNetwork, self).__init__()

        self.net = nn.Sequential( nn.Linear(inputsize,outputsize,bias=True),nn.ReLU() )
        

    def forward(self,nodes):

        message = torch.sum( nodes.mailbox['edge message'] ,dim=1 )
        
       
        input_data = torch.cat([nodes.data['node hidden rep'],message, nodes.data['node_type_embedding'],torch.argmax( nodes.data['node prediction'] ,dim=1).unsqueeze(1).float()],dim=1)
        
        
        return {'node hidden rep' : self.net(input_data) }

class JetClassifier(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config
        

        self.n_blocks = config['number of blocks']
        
        self.edge_networks = nn.ModuleList()
        self.node_networks = nn.ModuleList()

        gn_sizes = config['gn sizes']
        

        for gn_block_i in range(self.n_blocks):
            
            node_rep_size = gn_sizes[gn_block_i]
            output_node_rep_size = gn_sizes[gn_block_i+1]
            

            edge_net_input = node_rep_size*3+2*1+1+2*5 #node class and edge label
           
            node_net_inputsize = node_rep_size*2+1+5 #node class
            
            self.edge_networks.append(JetEdgeNetwork(edge_net_input,node_rep_size))
            self.node_networks.append(JetNodeNetwork(node_net_inputsize,output_node_rep_size))

        self.classifier_input_size = config['classifier layers'][0]

        self.jets_classifier = build_layers(self.classifier_input_size,outputsize=config['classifier layers'][-1],features=config['classifier layers'][1:-1])

    def forward(self, g):
        
        if g.num_nodes('nodes') > 0:
            
            graph_representation = dgl.mean_nodes(g,'node hidden rep', weight=None,ntype='nodes')

            g.nodes['nodes'].data['mean node hidden rep'] = dgl.broadcast_nodes(g, graph_representation,ntype='nodes')

            for gn_block_i in range(self.n_blocks):
                
                g.update_all(self.edge_networks[gn_block_i],self.node_networks[gn_block_i],etype='node_to_node')
                graph_representation = dgl.mean_nodes(g,'node hidden rep', weight=None,ntype='nodes')
                g.nodes['nodes'].data['mean node hidden rep'] = dgl.broadcast_nodes(g, graph_representation,ntype='nodes')
        
        
            g.update_all(fn.copy_src('node hidden rep','m'),fn.mean('m','global rep'),etype='to_global')
        else:
            
            hidden_repsize = self.classifier_input_size-g.nodes['global node'].data['jet features'].shape[1]
            g.nodes['global node'].data['global rep'] = torch.zeros(g.num_nodes('global node'),hidden_repsize,device=g.device)

        graph_representation = torch.cat([g.nodes['global node'].data['global rep'], g.nodes['global node'].data['jet features']],dim=1 )
    
        #graph_representation = dgl.mean_nodes(g,'node hidden rep', weight=None,ntype='nodes')

        prediction = self.jets_classifier(graph_representation)
        
        return prediction



