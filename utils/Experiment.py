""" Script to define the training loop and plotting functions for training """

import torch
import time
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def train_model(model, criterion, optimizer, scheduler, dataloaders,
                dataset_sizes, device, track_out, num_epochs=8):
    """

    :param model: initialized model to train
    :param criterion: loss function
    :param dataloaders: dictionary of dataloaders for training and validation
    :param dataset_sizes: dictionary of dataset sizes for training and validation
    :param device: CPU or GPU to train on
    :param track_out: whether or not to track the running accuracy and loss (for later plotting)
    :param num_epochs:
    :return: model, optimizer, best accuracy, best loss (optionally also running training/validation
    los and accuracy)
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_optim = copy.deepcopy(optimizer.state_dict())
    best_acc = 0.0
    best_loss = float("inf")

    if track_out:
        train_loss = []
        train_acc = []

        val_loss = []
        val_acc = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            probs = []
            label_list = []

            for data in dataloaders[phase]:

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(data.x, data.edge_index, data.batch)
                    outputs = outputs.squeeze(dim=-1)

                    data.y = data.y.float()

                    loss = criterion(outputs, data.y)

                    # propagate loss and step optimizer if in training stage
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    probs.extend(outputs.detach().cpu())
                    label_list.extend(data.y.detach().cpu())

                # get loss
                running_loss += loss.item() * data.y.size(0)


            # update the learning rate if this is training iteration
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_auprc = roc_auc_score(label_list, probs)

            if track_out:
                if phase == "train":
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_auprc)

                else:
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_auprc)

            print(f'{phase} Loss: {epoch_loss:.4f} ROC AUC: {epoch_auprc:.4f}')

            # save model if accuracy has improved
            if phase == 'val' and epoch_loss < best_loss:
                best_auprc = epoch_auprc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_optim = copy.deepcopy(optimizer.state_dict())
                best_loss = epoch_loss

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val AUPRC: {epoch_auprc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    optimizer.load_state_dict(best_optim)

    if track_out:
        return model, optimizer, best_auprc, best_loss, train_acc, train_loss, val_acc, val_loss

    return model, optimizer, best_auprc, best_loss

def plot_metrics(tloss, vloss, tacc, vacc, save = None):
    # plot training and validation loss
    plt.figure()
    x = range(1, len(tloss) + 1)
    plt.plot(x, tloss, label="Training Loss")
    plt.plot(x, vloss, label="Validation Loss")
    plt.legend()
    if not save:
        plt.show()
    else:
        plt.savefig(save + 'loss.png')

    # plot training and validation accuracy
    plt.figure()
    plt.plot(x, tacc, label="Training AUPRC")
    plt.plot(x, vacc, label="Validation AUPRC")
    plt.legend()
    if not save:
        plt.show()
    else:
        plt.savefig(save + "auprc.png")

def inference_model(model, dataloader):

    model.eval()  # Set model to evaluate mode

    probs = []
    label_list = []

    for data in dataloader:
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(data.x, data.edge_index, data.batch)
            outputs = outputs.squeeze(dim=-1)

            data.y = data.y.float()

            probs.extend(outputs.detach().cpu())
            label_list.extend(data.y.detach().cpu())

    return probs, label_list
