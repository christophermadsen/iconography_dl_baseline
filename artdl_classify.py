""" Using the trained ArtDL convolutional model for making classifications and
extracting the class activation maps.
"""

### Imports ###
import os, sys
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch_mods.dataset_with_paths import ImageFolderWithPaths as ImageFolder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from architecture import resnet50
from preprocessing.squarepad import SquarePad

### Global variables
# Path variables
test_set_path = '../DataFolders/test_folder'
info_path = '../DEVKitArt/info.csv'
class_txt = 'sets/classes.txt'

# Numeric variables
num_classes = 10

# Torch variables
cuda0 = torch.device('cuda:0')

### Main
## Preparing the data
# Loading the names of the 10 classes in the ArtDL model
with open(class_txt, 'r') as f:
    classes = [l.strip() for l in f.readlines()]

# Loading the targets dataframe
info = pd.read_csv(info_path)

# Initiating df for model outputs
output_df = pd.DataFrame(columns=['item'] + classes)

# Defining the transform at time of loading data
inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Defining dataset and dataloader (for just the test set)
test_set = ImageFolder(test_set_path, transform=inference_transform)
dataloader = DataLoader(test_set)

## Preparing the model
# Defining the ArtDL convolutional model
model = resnet50.Net(num_classes)

# Loading the state dictionary for the model
weights = torch.load('artdl_model/res50.pth', map_location=cuda0)

# Loading the weights into the model
model.load_state_dict(weights, strict=True)

# Set the model to use cuda and in evaluation mode
model.cuda()
model.eval()

# iterate over data, batch_size should be 1 !!!
counter = 0
data_len = len(test_set)
for data, item in dataloader:
    counter += 1
    if counter % 100 == 0:
        print(f'Doing {counter}/{data_len}', end='\r')
    data = data.cuda()
    output = model(data)
    output = torch.sigmoid(output[0])
    output = output.data.cpu().numpy()
    output[np.where(output != np.max(output))] = 0
    output[np.where(output == np.max(output))] = 1
    output = list(output.astype(int))
    output_df.loc[len(output_df)] = [item[0]] + output

output_df.to_csv('evaluation_files/art_dl_test.csv', index=False)
print('Done')
