""" Using the trained VGG convolutional model for making classifications
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
from architecture import conv_vgg
from preprocessing.squarepad import SquarePad

### Global variables
# Path variables
test_set_path = '../DataFolders/test_folder'
info_path = '../DEVKitArt/info.csv'
class_txt = 'sets/vgg_classes.txt'

input == input('Classify with strategy 1, 2 or 3? (1/2/3) ')
# First strategy
if input == '1':
    weights_path = 'best_vgg_1st_strategy/vgg_test_best_1st_strategy.pth'
    output_path = 'best_vgg_2nd_strategy/vgg_test_final.csv'
# Second strategy
elif input == '2':
    weights_path = 'best_vgg_2nd_strategy/vgg_test_best_2nd_strategy.pth'
    output_path = 'best_vgg_2nd_strategy/vgg_test_final.csv'
# Third strategy
elif input == '3':
    weights_path = 'best_vgg_3rd_strategy/vgg_test_best.pth'
    output_path = 'best_vgg_3rd_strategy/vgg_test_final.csv'
else:
    print('Try again with a valid input.')
    quit()

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
    SquarePad(),
    transforms.Resize((224, 224)),
])

# Defining dataset and dataloader (for just the test set)
test_set = ImageFolder(test_set_path, transform=inference_transform)
dataloader = DataLoader(test_set)

## Preparing the model
# Defining the ArtDL convolutional model
model = conv_vgg.ConvVGG(num_classes)

# Loading the weights into the model
model.load_state_dict(torch.load(weights_path, map_location=cuda0), strict=True)

# Set the model to use cuda and in evaluation mode
model.cuda()
model.eval()

# iterate over data, make classifications and save as csv
counter = 0
data_len = len(test_set)
for data, item in dataloader:
    counter += 1
    if counter % 100 == 0:
        print(f'Doing {counter}/{data_len}', end='\r')
    data = data.cuda()
    output = model(data)
    del data

    # Transform output and add result to dataframe
    output = torch.softmax(output, 1)[0]
    output = output.data.cpu().numpy()
    output[np.where(output != np.max(output))] = 0
    output[np.where(output == np.max(output))] = 1
    output = list(output.astype(int))
    output_df.loc[len(output_df)] = [item[0]] + output

# Save results as CSV
output_df.to_csv(output_path, index=False)
print('Done')
