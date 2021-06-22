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

def EdgePredictionLabelFunction(edges):

	edge_labels = ((edges.src['node idx'] > 0) & (edges.src['node idx']==edges.dst['node idx']))

	return {'edge predicted label': edge_labels.int().float() }

def message_pass(edges):
    
    edge_scores = edges.data['edge score']
    #edge_scores[edges.src['node class']!=edges.dst['node class']]
    
    return {'edge message': edges.src['node idx'], 'edge score' : edge_scores  }

def adjust_node_idx(nodes):
    
    edge_messages = nodes.mailbox['edge message']
    
    max_val = torch.max(nodes.mailbox['edge message'])+1
    
    edge_messages[ nodes.mailbox['edge score'] < 0.5 ] = max_val
    
    min_idx = torch.min(edge_messages,dim=1,keepdim=False)[0]
    
    min_idx = torch.stack([min_idx,nodes.data['node idx']],dim=1)
    
    min_idx = torch.min(min_idx,dim=1,keepdim=False)[0]

    return {'node idx': min_idx}

def get_cluster_assignment(g):

    node_class = torch.argmax( g.nodes['nodes'].data['node prediction'],dim=1 )
    
    g.nodes['nodes'].data['node class'] = node_class

    N = g.number_of_nodes(ntype='nodes')
    g.nodes['nodes'].data['node idx'] = torch.cat([torch.arange(N_i,device=g.device) for N_i in g.batch_num_nodes(ntype='nodes')])
    
    node_idx = g.nodes['nodes'].data['node idx']

    # node index becomes the cluster assigment - nodes that are connected by edges will 
    # get the same index 
    g.update_all(message_pass,adjust_node_idx,etype='node_to_node')

    while np.any( (g.nodes['nodes'].data['node idx'] != node_idx).cpu().data.numpy() ):
        node_idx = g.nodes['nodes'].data['node idx']
        g.update_all(message_pass,adjust_node_idx,etype='node_to_node')
        
    g.nodes['nodes'].data['node idx'][node_class==0] = -1

    return g


def symmetrize_edge_scores(g):

    #divide edges to two groups that one group contains
    #the edge i->j and the other j->i
    edge_set1 = torch.where( g.edges(etype='node_to_node')[0] > g.edges(etype='node_to_node')[1] )[0]
    
    
    edge_set2 = g.edge_ids(g.edges(etype='node_to_node')[1][edge_set1], g.edges(etype='node_to_node')[0][edge_set1],etype='node_to_node')

    edge_scores = torch.zeros(g.number_of_edges(etype='node_to_node'),device=g.device)
    

    edge_scores[edge_set2] = g.edges['node_to_node'].data['edge prediction'][edge_set1]+g.edges['node_to_node'].data['edge prediction'][edge_set2]
    edge_scores[edge_set1] = g.edges['node_to_node'].data['edge prediction'][edge_set1]+g.edges['node_to_node'].data['edge prediction'][edge_set2]

    g.edges['node_to_node'].data['edge score'] = edge_scores
    
    g.edges['node_to_node'].data['edge score'] = torch.sigmoid(g.edges['node_to_node'].data['edge score']).view(-1)

    
    return g

class Set2Graph(nn.Module):
    def __init__(self,config):
        super(Set2Graph, self).__init__()

        self.config = config

        self.edge_classifier = nn.Sequential( *build_layers(config['edge_inputsize'],outputsize=1,features=config['edge classifer layers']) )
        


        self.node_classifier = nn.Sequential( *build_layers(config['node_inputsize'],outputsize=config['n classes'],features=config['node classifer layers']) )

    def classify_edges(self,edges):
        
        input_data = torch.cat( [ edges.src['node features'],  edges.src['node hidden rep'], edges.src['common variables'], edges.dst['common variables'],
                                 edges.dst['node features'],  edges.dst['node hidden rep'], edges.dst['node_type_embedding'],
                                 edges.src['node_type_embedding'],
                                  edges.dst['mean node hidden rep']] ,dim=1)
        
        
        return {'edge prediction' : self.edge_classifier(input_data).squeeze(1) }


    def forward(self, g):

        
        
        g.nodes['nodes'].data['mean node hidden rep'] = dgl.broadcast_nodes(g,dgl.mean_nodes(g, 'node hidden rep', weight=None,ntype='nodes'),ntype='nodes') 
        
        g.apply_edges(self.classify_edges,etype='node_to_node')


        node_data = torch.cat([g.nodes['nodes'].data['node features'], g.nodes['nodes'].data['node hidden rep'],
                            g.nodes['nodes'].data['mean node hidden rep'], g.nodes['nodes'].data['node_type_embedding'],g.nodes['nodes'].data['common variables']
                           ],dim=1)

        g.nodes['nodes'].data['node prediction'] = self.node_classifier( node_data )
        
        return g


    def predict(self,g):
        
        symmetrize_edge_scores(g)
        get_cluster_assignment(g)

        node_pred = g.nodes['nodes'].data['node prediction']
        
        #get only the half of the edges (since they are symetric)
        edge_subset = np.where( g.edges(etype='node_to_node')[0].cpu() < g.edges(etype='node_to_node')[1].cpu() )[0]

        edge_scores = g.edges['node_to_node'].data['edge score'][edge_subset]

        cluster_assignments = g.nodes['nodes'].data['node idx']

        return node_pred, edge_scores, cluster_assignments
