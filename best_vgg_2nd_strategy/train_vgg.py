import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
# from architecture import conv_alexnet
from preprocessing.squarepad import SquarePad
import os
import time
import copy
import numpy as np
import PIL



###
# from architecture import vgg
from architecture import conv_vgg
###



# Transformations for the images
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#
#     'val': transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.02, 0.20)),
        SquarePad(),
        # transforms.Resize((224, 224)),
        transforms.Resize((224, 224)),
    ]),

    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        SquarePad(),
        # transforms.Resize((224, 224)),
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

loss_class_weights = [
    mary/mary, mary/antony, mary/dominic, mary/francis, mary/jerome,
    mary/john, mary/paul, mary/peter, mary/sebastian, mary/magdalene
]



# Data path
# data_dir = "../DataFolders/data_by_class"
data_dir = "../DataFolders/data_by_class_no_dup"
# data_dir = "../DataFolders/tl_learning_data/hymenoptera_data"

# Output dimension for the classifier
num_classes = 10

# Amount of processes to run parallel
num_workers = 3

# Amount of samples to load at a time
# batch_size = 8
batch_size = 8

# Number of times to go through the data during training
num_epochs = 200

device = torch.device("cuda:0")

loss_class_weights = torch.FloatTensor(loss_class_weights).to(device)



def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    val_acc_list = []
    val_loss_list = []



    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # count = 0
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
            top_1_error = 0

            # Iterate over data.
            batch_count = 0
            # check1 = time.time()
            for inputs, labels in dataloaders[phase]:
                # count += 1
                batch_count += batch_size
                if batch_count % 1000 == 0:
                    print(f'{batch_count}/{len(dataloaders[phase].dataset)} images done', end='\r')

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        # check2 = time.time()
                        # time_left = (check2 - check1) / batch_count * (len(dataloaders[phase].dataset) - batch_count)
                        # Track current batch
                        # print(f'time left = {time_left}, {batch_count}/{len(dataloaders[phase].dataset)} images done', end='\r')
                        # print(f'{batch_count}/{len(dataloaders[phase].dataset)} images done', end='\r')

                # print(f'{batch_size*count}/{dn}', end='\r')

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'val':
                val_loss_list.append(epoch_loss)
                val_acc_list.append(epoch_acc)
                metric_csv = np.array([val_loss_list, val_acc_list])
                np.savetxt("vgg_training_metrics.csv", metric_csv, delimiter=",")
                del metric_csv

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'model/vgg_test_best.pth')

                if (epoch + 1) % 25 == 0:
                    torch.save(model.state_dict(), f'model/vgg_test_epoch{epoch+1}.pth')

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print(f'Epoch took {time.time() - start:.2f} sec')
        print()


def freeze_layers(model, freeze_n=16):
    frozen = 0
    for param in model.parameters():
        frozen += 1
        if frozen > freeze_n:
            break
        param.requires_grad = False

if __name__ == '__main__':
    # torch.cuda.empty_cache()
    # model = models.resnet50()
    # model = conv_alexnet.ConvAlex(num_classes)

    # model = vgg.vgg16_bn(num_classes=num_classes)
    model = conv_vgg.ConvVGG(num_classes=num_classes)
    # weights_path = 'best_vgg/vgg_test_best.pth'
    # model.load_state_dict(torch.load(weights_path), strict=True)
    freeze_layers(model, 12)


    # TRAIN FURTHER
    # weights_path = 'best_vgg/vgg_test_best.pth'
    # model.load_state_dict(torch.load(weights_path), strict=True)
    # freeze_layers(model, 16)


    # model.new_classifier.requires_grad = True
    # model.new_classifier.weight.requires_grad = True
    print(model)
    # for param in model.parameters():
    #     print(param.size(), param.requires_grad)

    # c = 0
    # for param in model.parameters():
        # c+=1
    # print(c)

    # c= 0
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            # c += 1
            print("\t",name)
    # print(c)

    # quit()



    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # quit()


    criterion = nn.CrossEntropyLoss(weight=loss_class_weights)
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # original one
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # dataset = datasets.FakeData(size=1000, transform=transforms.ToTensor())
    # dataset = ImageFolder

    datasets_dict = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(datasets_dict[x], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) for x in ['train', 'val']}

    # print(datasets_dict['train'].class_to_idx)
    # quit()

    model.to('cuda')

    dn = len(datasets_dict['train'])

    train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
    torch.save(model.state_dict(), 'model/vgg_test_weights.pth')
    torch.save(optimizer.state_dict(), 'sgd_state.pth')


    # for epoch in range(num_epochs):
    #     print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    #     print('-' * 10)
    #     count = 0
    #     running_loss = 0.0
    #     running_corrects = 0
    #
    #     breaker = 0
    #     for data, target in dataloaders_dict['train']:
    #         ###
    #         breaker += 1
    #         ###
    #
    #         count += 1
    #         data = data.to('cuda', non_blocking=True)
    #         target = target.to('cuda', non_blocking=True)
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = criterion(output, target)
    #         loss.backward()
    #         optimizer.step()
    #         print(f'{batch_size*count}/{dn}', end='\r')
    #
    #         # statistics
    #         _, preds = torch.max(output, 1)
    #         running_loss += float(loss.item() * data.size(0))
    #         running_corrects += torch.sum(preds == target.data)
    #
    #         ###
    #         # if breaker == 1000:
    #         #     print(preds)
    #         #     break
    #         ###
    #
    #     epoch_loss = running_loss / dn
    #     epoch_acc = float(running_corrects.double() / dn)
    #
    #     print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    #
    # print('Done')















#
# # Amount of processes to run parallel
# num_workers = 2
#
# # Output dimension for the classifier
# num_classes = 10
#
# # Amount of samples to load at a time
# batch_size = 32
#
# # Number of times to go through the data during training
# num_epochs = 15
#
# # Transformations for the images
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#
#     'val': transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
#
# def set_parameter_requires_grad(model, freeze_n=0):
#     freeze = 0
#     for param in model.parameters():
#         param.requires_grad = False
#         freeze += 1
#         if freeze == freeze_n:
#             break
#
# def initialize_model(num_classes, freeze_n=0):
#     alex = alexnet.alexnet(pretrained=True)
#     # model_ft = models.alexnet(pretrained=use_pretrained)
#     set_parameter_requires_grad(alex, freeze_n)
#     num_ftrs = model_ft.classifier[6].in_features
#     model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
#     input_size = 224
#
#     return model_ft, input_size
#
#
#
# # def set_parameter_requires_grad(model, feature_extracting):
# #     if feature_extracting:
# #         c = 0
# #         for param in model.parameters():
# #             c += 1
# #             param.requires_grad = False
# #             if c == 8:
# #                 break
#
#
# def main():
#     alex = alexnet.alexnet(pretrained=True)
#     print(alex)
#     set_parameter_requires_grad(alex, freeze_n=0)
#     # criterion = nn.CrossEntropyLoss()
#     # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#
# if __name__ == '__main__':
#     main()
