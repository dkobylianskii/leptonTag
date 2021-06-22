import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph

import numpy as np
import torch
import torch.nn as nn

import importlib 
import models.vertexfinding_model as vertexfinding_model
importlib.reload(vertexfinding_model)
from models.vertexfinding_model import VertexFindingModel
from models.classifier import JetClassifier
from models.calo_net import CaloNet

def build_layers(config):

    inputsize = config['inputsize']
    features = config['layers']
    outputsize = config['outputsize']

    layers = []
    layers.append(nn.Linear(inputsize,features[0]))
    layers.append(nn.ReLU())
    for hidden_i in range(1,len(features)):
        layers.append(nn.Linear(features[hidden_i-1],features[hidden_i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(features[-1],outputsize))
    
    return nn.Sequential(*layers)

class JetTaggerNetwork(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.config = config



        self.track_init = build_layers(config['track initializer'])
        self.lepton_init = build_layers(config['lepton initializer'])
        self.cell_init = build_layers(config['cell initializer'])
        # self.calo_net = CaloNet()

        self.node_type_embd = nn.Embedding(num_embeddings=5,embedding_dim=5)
        self.use_vertexing = config['load pre-trained vertex finder']

        self.vertex_finder = VertexFindingModel(config['vertex finder'])

        if self.use_vertexing:
            checkpoint_path = config['vertex finder checkpoint']
            print(checkpoint_path)
            checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))




            state_dict = checkpoint['state_dict']
            vertex_state_dict = {key: state_dict['net.vertex_finder.'+key] for key in self.vertex_finder.state_dict().keys()}
            self.vertex_finder.load_state_dict(vertex_state_dict)

            track_init_sd = {key: state_dict['net.track_init.'+key] for key in self.track_init.state_dict().keys()}
            self.track_init.load_state_dict(track_init_sd)
            lepton_init_sd = {key: state_dict['net.lepton_init.'+key] for key in self.lepton_init.state_dict().keys()}
            self.lepton_init.load_state_dict(lepton_init_sd)
            

            n_embd_sd = {key: state_dict['net.node_type_embd.'+key] for key in self.node_type_embd.state_dict().keys()}
            self.node_type_embd.load_state_dict(n_embd_sd)


        if self.config['train classifier']:
            self.jet_classifier = JetClassifier(config['classifier config'])


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
    
    def init_nodes(self,g):
        
        g.nodes['tracks'].data['node_type_embedding'] = self.node_type_embd(g.nodes['tracks'].data['node_type'])
        g.nodes['leptons'].data['node_type_embedding'] = self.node_type_embd(g.nodes['leptons'].data['node_type'])
        g.nodes['cells'].data['node_type_embedding'] = self.node_type_embd(g.nodes['cells'].data['node_type'])

        self.move_from_tracksleptonscells_to_nodes(g,'node_type_embedding','node_type_embedding','node_type_embedding', 'node_type_embedding')

        g.nodes['tracks'].data['track rep'] = self.track_init(g.nodes['tracks'].data['track variables'])
        g.nodes['leptons'].data['lepton rep'] =  self.lepton_init(g.nodes['leptons'].data['lep variables'])
        g.nodes['cells'].data['cell rep'] = self.cell_init(g.nodes['cells'].data['cell variables'])

        self.move_from_tracksleptonscells_to_nodes(g,'track rep','lepton rep','cell rep', 'node features')
        # g.update_all(fn.copy_src('track rep','m'),fn.sum('m', 'track rep'),etype=('tracks','track_to_node','nodes'))
        # tracks_only_data = g.nodes['nodes'].data['node features']
        # g.update_all(fn.copy_src('lepton rep','m'),fn.sum('m', 'lepton rep'),etype=('leptons','leptons_to_node','nodes'))
        # leptons_only_data = g.nodes['nodes'].data['node features']
        # g.update_all(fn.copy_src('cell rep','m'),fn.sum('m', 'cell rep'),etype=('cells','cells_to_node','nodes'))
        # g.nodes['nodes'].data['node features'] = g.nodes['nodes'].data['node features']+tracks_only_data + leptons_only_data
        
        self.move_from_tracksleptons_to_nodes(g,'common variables','common variables','common variables')

        
    def forward(self, g):
        if g.num_nodes('nodes') > 0:
            if self.use_vertexing:
                with torch.no_grad():
                    self.init_nodes(g)
                    self.vertex_finder(g)
            else:
                self.init_nodes(g)
                # print(g.nodes['nodes'].data['common variables'].shape)
                self.vertex_finder(g)
        
        out = None
        if self.config['train classifier']:
            out = self.jet_classifier(g)
  
        return out, g



    def predict(self,g, calo_array=None):
        with torch.no_grad():
            self.init_nodes(g)
            node_pred, edge_pred, cluster_assignments = self.vertex_finder.predict(g)
            jetclass_predict = None
            if self.config['train classifier']:
                jetclass_predict = self.jet_classifier(g)
        return node_pred, edge_pred, cluster_assignments, jetclass_predict






