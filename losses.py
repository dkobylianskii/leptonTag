
import torch
import torch.nn as nn
from dataloader import EdgeLabelFunction
import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph




def EdgeWeightFunction(edges):

    

    condition = ( (edges.dst['node labels']==3) | (edges.dst['node labels']==2) ) 
    condition = condition & ( (edges.src['node labels']==3) | (edges.src['node labels']==2) ) 
    device = condition.device

    edge_weights = torch.where(condition ,torch.tensor(2.0,device=device),torch.tensor(1.0,device=device))

    
    return {'edge loss weight': edge_weights.float() }


def f1_loss(g):

    edata = g.edges['node_to_node'].data

    edata['y hat'] = torch.sigmoid(edata['edge prediction'])
    edata['tp'] = edata['y hat']*edata['edge label']*edata['edge loss weight']
    edata['fn'] = (1.0-edata['y hat'])*edata['edge label']*edata['edge loss weight']
    edata['fp'] = edata['y hat']*(1.0-edata['edge label'])*edata['edge loss weight']


    tp = dgl.sum_edges(g,'tp',etype='node_to_node')
    
    fn =  dgl.sum_edges(g,'fn',etype='node_to_node')
    fp =  dgl.sum_edges(g,'fp',etype='node_to_node')

    
    loss = - ((2 * tp) / (2 * tp + fp + fn + 1e-10)).mean()
    
    return loss

class EdgeLossBCE(nn.Module):
    def __init__(self):
        super(EdgeLossBCE, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self,g):

        


        g.edges['node_to_node'].data['bce loss'] = self.BCE( g.edges['node_to_node'].data['edge prediction'], g.edges['node_to_node'].data['edge label'] 
                                                        )*g.edges['node_to_node'].data['edge loss weight']
        


        graph_bce = dgl.mean_edges(g, 'bce loss',etype='node_to_node')
        #print('--------')
        #print(torch.sigmoid( g.edata['edge prediction'][:10]) )
        #print(g.edata['edge label'][:10])
        #print('--------')
        return torch.mean(graph_bce,dim=0)
        



class VertexFindingLoss(nn.Module):
    def __init__(self,config):
        super(VertexFindingLoss, self).__init__()

        self.config = config

        self.node_loss = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1.0,1.0,1.0]),reduction='none', ignore_index=-1)
        
        self.edge_BCE = EdgeLossBCE()
    
    def move_from_tracksleptons_to_nodes(self,g,track_info,lep_info,target_name):

        g.update_all(fn.copy_src(track_info,'m'),fn.sum('m',target_name),etype=('tracks','track_to_node','nodes'))
        tracks_only_data = g.nodes['nodes'].data[target_name]
        g.update_all(fn.copy_src(lep_info,'m'),fn.sum('m',target_name),etype=('leptons','leptons_to_node','nodes'))
        g.nodes['nodes'].data[target_name] = g.nodes['nodes'].data[target_name]+tracks_only_data 

    def move_from_tracksleptonscells_to_nodes(self,g,track_info,lep_info,cell_info,target_name):

        g.update_all(fn.copy_src(track_info,'m'),fn.sum('m',target_name),etype=('tracks','track_to_node','nodes'))
        tracks_only_data = g.nodes['nodes'].data[target_name]
        g.update_all(fn.copy_src(lep_info,'m'),fn.sum('m',target_name),etype=('leptons','leptons_to_node','nodes'))
        leptons_only_data = g.nodes['nodes'].data[target_name]
        g.update_all(fn.copy_src(cell_info,'m'),fn.sum('m',target_name),etype=('cells','cells_to_node','nodes'))
        g.nodes['nodes'].data[target_name] = g.nodes['nodes'].data[target_name]+tracks_only_data + leptons_only_data

    def forward(self, g):

        
        for var in ['node labels','node vtx idx']:
            self.move_from_tracksleptonscells_to_nodes(g,var,var,var,var)
            g.nodes['nodes'].data[var] = g.nodes['nodes'].data[var].long()

        n_prediction = g.nodes['nodes'].data['node prediction']
        n_label =  g.nodes['nodes'].data['node labels'].long()
        g.apply_edges(EdgeLabelFunction,etype='node_to_node')

        g.nodes['nodes'].data['node loss'] = self.node_loss(n_prediction,n_label)


        node_loss = dgl.sum_nodes(g,'node loss',ntype='nodes').mean()

        g.apply_edges(EdgeWeightFunction,etype='node_to_node')
        edge_bce = self.edge_BCE(g)
        edge_f1 = f1_loss(g)
        loss = node_loss*self.config['loss weights']['node loss'] + edge_bce*self.config['loss weights']['edge bce']+ edge_f1*self.config['loss weights']['edge f1']
        
        return {'loss':loss,'node loss':node_loss.item(),'edge bce' : edge_bce.item(), 'edge f1': edge_f1.item() }



class JetClassificationLoss(nn.Module):
    def __init__(self,config):
        super(JetClassificationLoss, self).__init__()

        self.config = config

        self.jet_loss = nn.CrossEntropyLoss()
        

        
    def forward(self, predictions,targets):

        loss = self.jet_loss(predictions,targets)
        
        return { 'loss':loss , 'classification loss' : loss.item() }

