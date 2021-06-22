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


class EdgeNetwork(nn.Module):
    def __init__(self,layers):
        super(EdgeNetwork, self).__init__()

    
    
        self.net = build_layers(layers[0],layers[-1],layers[1:-1])

    def forward(self, edges):
        
        edge_input = torch.cat( 
            [edges.dst['node hidden rep'],edges.src['node hidden rep']],dim=1 )
        
        node_data = torch.cat(
            [edges.src[inputname] for inputname in self.inputs],dim=1)

        return {'edge weight': edge_weight, 'edge message' : node_data }


class JetClassifier(nn.Module):
    def __init__(self,config):
        super(JetClassifier, self).__init__()

        self.config = config

        self.edge_networks = nn.Sequential( *build_layers(config['edge_inputsize'],outputsize=1,features=config['edge classifer layers']) )
        


        self.node_classifier = nn.Sequential( *build_layers(config['node_inputsize'],outputsize=config['n classes'],features=config['node classifer layers']) )

    def edge_function(self,edges):

        edge_weight = torch.sum( edges.dst['query']*edges.src['key'],dim=1 )/self.d_k
        
        node_data = torch.cat(
            [edges.src[inputname] for inputname in self.inputs],dim=1)

        return {'edge weight': edge_weight, 'edge message' : node_data }
    
    def node_attention(self,nodes):
    
        #these will have the shape (n_nodes, n_connected_nodes, features )
        edge_weights = nodes.mailbox['edge weight']
        edge_messages = nodes.mailbox['edge message']
        
        edge_weights = torch.softmax(edge_weights,dim=1)

        node_rep = torch.sum((edge_messages*edge_weights.unsqueeze(2)),dim=1)

        return {'node attention': node_rep}


    def forward(self, g):

        
        
        g.ndata['mean node hidden rep'] = dgl.broadcast_nodes(g,dgl.mean_nodes(g, 'node hidden rep', weight=None)) 
        
        g.apply_edges(self.classify_edges)


        node_data = torch.cat([g.ndata['node features'], g.ndata['node hidden rep'],
                            g.ndata['mean node hidden rep'] 
                           ],dim=1)

        g.ndata['node prediction'] = self.node_classifier( node_data )
        
        return g


    def predict(self,g):

        self.forward(g)

        symmetrize_edge_scores(g)
        get_cluster_assignment(g)

        node_pred = g.ndata['node prediction']
        
        #get only the half of the edges (since they are symetric)
        edge_subset = np.where( g.edges()[0].cpu() < g.edges()[1].cpu() )[0]

        edge_scores = g.edata['edge score'][edge_subset]

        cluster_assignments = g.ndata['node idx']

        return node_pred, edge_scores, cluster_assignments
