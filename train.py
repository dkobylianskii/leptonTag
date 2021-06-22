import comet_ml
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import dgl
import dgl.function as fn
from dgl import DGLGraph as DGLGraph

from lightning import VertexFindingLightning

from pytorch_lightning import Trainer
from dataloader import JetsDataset, collate_graphs
import sys
import json
import glob
import torch

import os

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

if __name__ == "__main__":

    config_path = sys.argv[1]

    with open(config_path, "r") as fp:
        config = json.load(fp)

    net = VertexFindingLightning(config)

    if len(sys.argv) > 3:
        checkpoint_path = sys.argv[3]
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        net.load_state_dict(checkpoint["state_dict"])

    comet_logger = CometLogger(
        api_key="eDscEuoqRQaUAjYb3cBNeKHfn",
        project_name="electrontag",
        workspace="dkobylianskii",
        save_dir=".",
        experiment_name=config["name"],
    )
    comet_logger.experiment.log_asset(config_path, file_name="classifier.json")
    all_files = glob.glob("./*.py") + glob.glob("models/*.py")
    for fpath in all_files:
        comet_logger.experiment.log_asset(fpath)
    print("creating trainer")
    trainer = Trainer(max_epochs=50, gpus=1, logger=comet_logger)
    trainer.fit(net)
