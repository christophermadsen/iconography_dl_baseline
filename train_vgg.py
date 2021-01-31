""" This file is used for training the VGG networks used in the thesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from architecture import conv_vgg
from preprocessing.squarepad import SquarePad
import os
import time
import copy
import numpy as np

# Data transformations used for training and validation sets.
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.02, 0.20)),
        SquarePad(),
        transforms.Resize((224, 224)),
    ]),

    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        SquarePad(),
        transforms.Resize((224, 224)),
    ]),
}

# Training image counts per class
mary = 9513
antony = 115
dominic = 234
francis = 784
jerome = 939
john = 783
paul = 419
peter = 949
sebastian = 448
magdalene = 727

# Calculate the class weights for the CrossEntropyLoss function.
loss_class_weights = [
    mary/mary, mary/antony, mary/dominic, mary/francis, mary/jerome,
    mary/john, mary/paul, mary/peter, mary/sebastian, mary/magdalene
]

# Output dimension for the classifier
num_classes = 10

# Amount of processes to run parallel. Adjust this based on the machine.
num_workers = 3

# Amount of samples to load at a time. Adjust this based on the machine.
batch_size = 8

# Number of times to go through the data during training. Adjust this based on
# the data size (varies with augmentation methods) and training strategies such
# as learning rate.
num_epochs = 200

# Device to run model on and copy data to
device = torch.device("cuda:0")

# Weights for CrossEntropyLoss
loss_class_weights = torch.FloatTensor(loss_class_weights).to(device)

""" Helper function for training the VGG model. Takes a predefined model,
loaded Torch Dataloaders, a predefined loss function, a predefined loss function
and a desired numbers of epochs to train for.
"""
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, metrics_save_path='metrics.csv', weights_save_path='weights.pth'):
    val_acc_list = []
    val_loss_list = []
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            batch_count = 0
            for inputs, labels in dataloaders[phase]:
                batch_count += batch_size
                if batch_count % 1000 == 0:
                    print(f'{batch_count}/{len(dataloaders[phase].dataset)} images done', end='\r')

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # training metrics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # training metrics for an entire epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # Saving validation metrics and weights.
            if phase == 'val':
                val_loss_list.append(epoch_loss)
                val_acc_list.append(epoch_acc)
                metric_csv = np.array([val_loss_list, val_acc_list])
                np.savetxt(metrics_save_path, metric_csv, delimiter=",")
                del metric_csv

                # Keeping track of model which performed best on validation set
                # and saving the weights if its the best.
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), weights_save_path)

                # Saving weights every 25th epoch for further inspection
                # if (epoch + 1) % 25 == 0:
                    # torch.save(model.state_dict(), f'further/vgg_test_epoch{epoch+1}.pth')

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print(f'Epoch took {time.time() - start:.2f} sec')
        print()

""" Freezes the parameters for n layers given any model. This is vital when
applying transfer learning.
"""
def freeze_layers(model, freeze_n=12):
    frozen = 0
    for param in model.parameters():
        frozen += 1
        if frozen > freeze_n:
            break
        param.requires_grad = False

""" Main call. This is absolutely vital for windows machines when dealing with
multi processing. If this wrapper does not exist every worker would redefine the
all the functions and variables in this file.
"""
if __name__ == '__main__':
    input = input('Train model with strategy 1, 2, or 3? (1/2/3) ')

    if input == '1':
        metrics_save_path = "best_vgg_1st_strategy/vgg_training_metrics.csv"
        weights_save_path = 'best_vgg_1st_strategy/vgg_test_best.pth'
    elif input =='2':
        metrics_save_path = "best_vgg_2nd_strategy/vgg_training_metrics.csv"
        weights_save_path = 'best_vgg_2nd_strategy/vgg_test_best.pth'
    elif input == '3':
        metrics_save_path = "best_vgg_3rd_strategy/vgg_training_metrics.csv"
        weights_save_path = 'best_vgg_3rd_strategy/vgg_test_best.pth'
    else:
        print('Try again with a valid input.')
        quit()

    # Data paths
    if input == '2':
        data_dir = "../DataFolders/data_by_class_no_dup"
    else:
        data_dir = "../DataFolders/data_by_class"

    # Initialize model
    model = conv_vgg.ConvVGG(num_classes=num_classes)

    # TRAIN FURTHER
    # Uncomment this to train further on an already trained model. Note that the
    # fully convolutional VGG model is always pretrained on ImageNet.
    if input == '3':
        weights_path = 'best_vgg_1st_strategy/vgg_test_best_1st_strategy.pth' # give the path to the weights
        model.load_state_dict(torch.load(weights_path), strict=True)

    # FREEZE
    # Freeze the desired number of layers. 12 layers results in the input layer
    # and the first convolutional block, plus its batchnorm layers, being frozen
    if input == '2':
        freeze_layers(model, 12)
    else:
        freeze_layers(model, 16)

    # Show the model
    print(model)

    # Show the parameters to be trained
    print('Parameters being updated through SGD')
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    # Definining the loss function
    # criterion = nn.CrossEntropyLoss(weight=loss_class_weights)
    criterion = nn.CrossEntropyLoss() # For Milani et al strategy

    # Defining the optimizer
    if input == '2':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Define training and validation data sets
    datasets_dict = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    # Create loaders for the datasets
    dataloaders_dict = {x: torch.utils.data.DataLoader(datasets_dict[x], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) for x in ['train', 'val']}

    # Bring the model to cuda
    model.to('cuda')

    # The length of the training set
    dn = len(datasets_dict['train'])

    # Traing the model using the helper function
    train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, metrics_save_path=metrics_save_path, weights_save_path=weights_save_path)

    # Save the final optimizer state
    # torch.save(optimizer.state_dict(), 'further/sgd_state.pth')
