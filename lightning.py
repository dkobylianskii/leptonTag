

from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from models.vertexfinding_model import VertexFindingModel
from dataloader import JetsDataset, collate_graphs
import numpy as np
from losses import VertexFindingLoss
from object_cond_loss import ObjectCondenstationLoss
from models.classification_model import JetTaggerNetwork
from losses import JetClassificationLoss

class VertexFindingLightning(LightningModule):

    def __init__(self,config):
        super().__init__()
        
        self.config = config
        
        self.net = JetTaggerNetwork(self.config)


        self.train_vertexing = self.config['train vetex finder']
        self.train_classification =  self.config['train classifier']
        self.use_pretrained_vertexfinder = self.config['load pre-trained vertex finder']

        if self.train_vertexing:
            self.vertex_loss_type = config['vertex finding loss type']

            if self.vertex_loss_type== 'condensation':
                self.vertexing_loss = ObjectCondenstationLoss(config)
                
            else:
                self.vertexing_loss = VertexFindingLoss(config)
        
        if self.train_classification:
            self.jet_class_loss = JetClassificationLoss(config)


    def forward(self, g):

        return self.net(g)

    
    def training_step(self, batch, batch_idx):
        
        g,jet_flav = batch
        
        jet_flav_prediction, g = self(g)
        


        if self.train_vertexing and self.train_classification:
            vtx_loss = self.vertexing_loss(g)
            loss = self.jet_class_loss(jet_flav_prediction,jet_flav)
            loss['loss'] = loss['loss']+vtx_loss['loss']
            for key in vtx_loss.keys():
                if key!='loss':
                    loss[key] = vtx_loss[key]

        elif self.train_vertexing:
            loss = self.vertexing_loss(g)
        else:
            loss = self.jet_class_loss(jet_flav_prediction,jet_flav)

        
        return_dict = {
            'loss' : loss['loss']
        }

        return_dict['log'] = {}
        for loss_type in self.config['loss types']:
            return_dict['log'][loss_type] = loss[loss_type]
 
        return return_dict

    def validation_step(self, batch, batch_idx):
        
        g,jet_flav= batch
        jet_flav_prediction, g = self(g)

        if self.train_vertexing and self.train_classification:
            vtx_loss = self.vertexing_loss(g)
            loss = self.jet_class_loss(jet_flav_prediction,jet_flav)
            loss['loss'] = loss['loss']+vtx_loss['loss']
            for key in vtx_loss.keys():
                if key!='loss':
                    loss[key] = vtx_loss[key]

        elif self.train_vertexing:
            loss = self.vertexing_loss(g)
        else:
            loss = self.jet_class_loss(jet_flav_prediction,jet_flav)

        
        return_dict = {
            'val_loss' : loss['loss'],
            'preds' : jet_flav_prediction,
            'target': jet_flav
        }

        return_dict['log'] = {}
        for loss_type in self.config['loss types']:
            return_dict['log'][loss_type] = loss[loss_type]
 
        return return_dict
    
    #def validation_epoch_end(self, outputs):
    #    preds = torch.cat([tmp['preds'] for tmp in outputs]).cpu()
    #    targets = torch.cat([tmp['target'] for tmp in outputs]).cpu()
    #    self.logger.experiment.log_confusion_matrix(
    #        targets,
    #        preds,
    #        title="Confusion Matrix, Epoch #%d" % (self.current_epoch + 1),
    #        file_name="confusion-matrix-%03d.json" % (self.current_epoch + 1),
    #    )
        

    def configure_optimizers(self):

        #if self.use_pretrained_vertexfinder:
         #   return torch.optim.Adam(self.net.jet_classifier.parameters(), lr=self.config['learningrate'])

        return torch.optim.Adam(self.parameters(), lr=self.config['learningrate'])

    
    def train_dataloader(self):
        
        dataset = JetsDataset(self.config['path_to_train'],self.config,reduce_ds=self.config['reduce_dataset'])

        loader = DataLoader(dataset, batch_size=self.config['batchsize'], 
            num_workers=self.config['num_workers'], shuffle=True,collate_fn=collate_graphs,pin_memory=False)
        return loader
    
    def val_dataloader(self):
        dataset = JetsDataset(self.config['path_to_valid'],self.config,reduce_ds=self.config['reduce_dataset'])

        loader = DataLoader(dataset, batch_size=self.config['batchsize'], num_workers=self.config['num_workers'], 
        shuffle=False,collate_fn=collate_graphs,pin_memory=False)
        return loader

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        
        return_dict = {'val_loss': avg_loss, 'log': {'val_loss': avg_loss}}
        for loss_type in self.config['loss types']:
            return_dict['log']['val_'+loss_type] = np.mean([x['log'][loss_type] for x in outputs])

        return return_dict
