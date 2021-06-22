import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph


import numpy as np
import torch
import torch.nn as nn


def build_layers(inputsize,outputsize,features,add_activation=None):
    layers = []
    layers.append(nn.Linear(inputsize,features[0]))
    for hidden_i in range(1,len(features)):
        layers.append(nn.Linear(features[hidden_i-1],features[hidden_i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(features[-1],outputsize))

    if add_activation!=None:
        layers.append(add_activation)
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

    node_class = torch.argmax( g.ndata['node prediction'],dim=1 )
    
    g.ndata['node class'] = node_class

    N = g.number_of_nodes()
    g.ndata['node idx'] = torch.cat([torch.arange(N_i) for N_i in g.batch_num_nodes()]).to(g.device)
    
    node_idx = g.ndata['node idx']

    # node index becomes the cluster assigment - nodes that are connected by edges will 
    # get the same index 
    g.update_all(message_pass,adjust_node_idx)

    while np.any( (g.ndata['node idx'] != node_idx).cpu().data.numpy() ):
        node_idx = g.ndata['node idx']
        g.update_all(message_pass,adjust_node_idx)
        
    g.ndata['node idx'][node_class==0] = -1

    return g


def symmetrize_edge_scores(g):

    #divide edges to two groups that one group contains
    #the edge i->j and the other j->i
    edge_set1 = torch.where( g.edges()[0] > g.edges()[1] )[0]
    
    
    edge_set2 = g.edge_ids(g.edges()[1][edge_set1], g.edges()[0][edge_set1])

    edge_scores = torch.zeros(g.number_of_edges())
    
    edge_scores = edge_scores.to(g.device)

    edge_scores[edge_set2] = g.edata['edge prediction'][edge_set1]+g.edata['edge prediction'][edge_set2]
    edge_scores[edge_set1] = g.edata['edge prediction'][edge_set1]+g.edata['edge prediction'][edge_set2]

    g.edata['edge score'] = edge_scores
    
    g.edata['edge score'] = torch.sigmoid(g.edata['edge score']).view(-1)

    
    return g

class ObjectAssignment(nn.Module):
    def __init__(self,config):
        super(ObjectAssignment, self).__init__()

        self.config = config
        
        self.object_prediction = nn.Sequential( *build_layers(config['node_inputsize'],outputsize=config['N max objects'],features=config['object_prediction layers'],add_activation=nn.ReLU() ) )
        


        self.node_classifier = nn.Sequential( *build_layers(config['node_inputsize'],outputsize=config['n classes'],features=config['node classifer layers']) )

    def compute_edge_pred(self,edges):
        
        edge_pred = torch.sum( edges.src['object prediction']*edges.dst['object prediction'] ,dim=1)

    
        return {'edge prediction' : edge_pred }


    def forward(self, g):

        
        g.ndata['mean node hidden rep'] = dgl.broadcast_nodes(g,dgl.mean_nodes(g, 'node hidden rep', weight=None)) 
        
        
        node_data = torch.cat([g.ndata['node features'], g.ndata['node hidden rep'],
                            g.ndata['mean node hidden rep'] 
                           ],dim=1)


        g.ndata['object prediction'] = self.object_prediction( node_data )


        g.apply_edges(self.compute_edge_pred)

        g.ndata['node prediction'] = self.node_classifier( node_data )
        
        return g


    def predict(self,g):
        
        symmetrize_edge_scores(g)
        #get_cluster_assignment(g)

        node_pred = g.ndata['node prediction']
        node_class_pred = torch.argmax(node_pred,dim=1)
        #get only the half of the edges (since they are symetric)
        edge_subset = np.where( g.edges()[0].cpu() < g.edges()[1].cpu() )[0]

        edge_scores = g.edata['edge score'][edge_subset]

        cluster_assignments = torch.argmax(g.ndata['object prediction'],dim=1)
        cluster_assignments[node_class_pred==0] = -1

        return node_pred, edge_scores, cluster_assignments
