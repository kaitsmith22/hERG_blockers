"""
Fine tune hyperparameters using RayTune
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler
from torchvision import models, transforms
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

import numpy as np
# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
# Pytorch and Pytorch Geometric
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader

from torch_geometric.loader import DataLoader
from ..utils.MoleculeDataset import MoleculeDataset
from ..models.GNNClassifier import GNNClassifier
from ..utils.Experiment import train_model
from sklearn.metrics import accuracy_score, precision_score, f1_score

def run_ray_tune(config, checkpoint_dir=None):
    """
    Function to run ray tune experiment for each sample from the parameter space
    :param config: parameters selected by RayTune tuner
    :param checkpoint_dir: set to None to allow for checkpointing
    """

    # load the data
    train_data = MoleculeDataset(transform=None, data_split="train")
    val_data = MoleculeDataset(transform=None, data_split="valid")
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    dataloaders = {"train": train_loader,
                   "val": val_loader}

    datasets_sizes = {
        'train': len(train_data),
        'val': len(val_data)
    }


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    layer_names = ["GAT"]

    model_dict = {}
    model_results = {}

    # create models

    model_ft = GNNClassifier(79, 1024, 1, 'GAT', num_heads=5)

    criterion = torch.nn.BCELoss()

    # Here I am fine-tuning all features in the model, but I could also
    # explore freezing layers and fine-tuning certain layers
    # I also prefer to use SGD over Adam, especially for more simple classification problems
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=config["lr"], weight_decay= config["weight_decay"])

    # To restore a checkpoint, use `session.get_checkpoint()`.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        model_ft.load_state_dict(model_state)
        optimizer_ft.load_state_dict(optimizer_state)

    model_ft = model_ft.to(device)

    # Decay learning rate by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=config["step_size"], gamma=0.1)

    model_ft, optimizer_ft, accuracy, loss = train_model(model_ft,
                                                         criterion,
                                                         optimizer_ft,
                                                         exp_lr_scheduler,
                                                         dataloaders,
                                                         datasets_sizes,
                                                         device,
                                                         track_out= False,
                                                         num_epochs=max_epoch)

    os.makedirs("drill_classifier_checkpoints", exist_ok=True)
    torch.save(
        (model_ft.state_dict(), optimizer_ft.state_dict()), "drill_classifier_checkpoints/checkpoint.pt")
    checkpoint = Checkpoint.from_directory("drill_classifier_checkpoints")
    session.report({"loss": loss, "accuracy": accuracy}, checkpoint=checkpoint)


if __name__ == '__main__':

    num_samples = 4

    max_epoch = 15

    config = {
        "lr": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
        "batch_size": tune.choice([16, 64]),
        "step_size": tune.choice([5,10]),
        "weight_decay": tune.choice([.001, .01, .1])
    }
    scheduler = ASHAScheduler(
        max_t=max_epoch,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(run_ray_tune),
            resources={"cpu": 2, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()