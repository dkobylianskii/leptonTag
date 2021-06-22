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





class CondNet(nn.Module):
    def __init__(self,config):
        super(CondNet, self).__init__()

        self.config = config

        self.beta_net = nn.Sequential( *build_layers(config['node_inputsize'],outputsize=1,features=config['beta net layers'],add_activation=nn.Sigmoid()) )
        self.x_net = nn.Sequential( *build_layers(config['node_inputsize'],outputsize=config['x size'],features=config['x net layers'],add_activation=nn.Tanh()) )
        self.nodeclass_net = nn.Sequential( *build_layers(config['node_inputsize'],outputsize=config['n classes'],features=config['node classifer layers']) )


        self.t_d = self.config['t_d']
        self.t_b = self.config['t_b']


    def forward(self, g):

        
        
        g.ndata['mean node hidden rep'] = dgl.broadcast_nodes(g,dgl.mean_nodes(g, 'node hidden rep', weight=None)) 
        
        


        node_data = torch.cat([g.ndata['node features'], g.ndata['node hidden rep'],
                            g.ndata['mean node hidden rep'] 
                           ],dim=1)

        g.ndata['node prediction'] = self.nodeclass_net( node_data )
        
        g.ndata['beta'] = self.beta_net( node_data )
        g.ndata['x'] = self.x_net( node_data )

        
        return g
    
    def EdgeDistanceFunction(self,edges):

        edge_dist = torch.sum( (edges.src['x']-edges.dst['x'])**2 ,dim=1 )

        return {'edge distance': edge_dist }

    def EdgeScoreFunction(self,edges):

        edge_score = ((edges.src['cluster assignment'] > -1) & (edges.src['cluster assignment']==edges.dst['cluster assignment'])).int().float()

        return {'edge score': edge_score }

    def message_pass(self,edges):
        
        
        return {'edge distance': edges.data['edge distance'], 'beta message' : edges.src['beta'].view(-1),
            'node class' : edges.src['node class pred'], 'node idx message' : edges.src['node idx']}


    def DetermineCondensationNode(self,nodes):
        

    
        distances = nodes.mailbox['edge distance']
        N_nodes = distances.shape[0]
        betas = nodes.mailbox['beta message']
        node_class = nodes.mailbox['node class']
        node_idxs = nodes.mailbox['node idx message']
        
        betas[distances > self.t_d] = -1
        betas[betas < self.t_b] = -1
        
        bkg_points = torch.all( betas < 0 ,dim=1)
        object_points = torch.logical_not( bkg_points )
        
        max_betas, max_indices = torch.max(betas,dim=1)
        
        n_range = torch.arange(N_nodes).to(betas.device)
        
        cluster_assignment = node_idxs[n_range,max_indices]
        node_class_assignment = node_class[n_range,max_indices]
        
        cluster_assignment[bkg_points] = -1
        node_class_assignment[bkg_points] = -1
        
        
        return {'cluster assignment': cluster_assignment, 'node class pred' :  node_class_assignment}

        
    def predict(self,g):
        
        print( 'g.batch_num_nodes() ', g.batch_num_nodes() )
        g.ndata['node idx'] = torch.cat([torch.arange(N_i) for N_i in g.batch_num_nodes()]).to(g.device)
        g.ndata['node class pred'] = torch.argmax( g.ndata['node prediction'],dim=1)
        
        g_with_loop = dgl.add_self_loop(g)
        g_with_loop.apply_edges(self.EdgeDistanceFunction)
        
        g_with_loop.update_all(self.message_pass,self.DetermineCondensationNode)
           
        g.ndata['cluster assignment'] =   g_with_loop.ndata['cluster assignment']
        g.ndata['node class pred'] =   g_with_loop.ndata['node class pred']
        

        g.apply_edges(self.EdgeScoreFunction)

        #get only the half of the edges (since they are symetric)
        edge_subset = np.where( g.edges()[0].cpu() < g.edges()[1].cpu() )[0]

        edge_scores = g.edata['edge score'][edge_subset]

        cluster_assignments = g.ndata['cluster assignment']

        node_pred = g.ndata['node prediction']

        return node_pred, edge_scores, cluster_assignments