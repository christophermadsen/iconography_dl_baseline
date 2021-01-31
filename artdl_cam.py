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
from matplotlib import cm

### Global variables
# Path variables
test_set_path = '../DataFolders/test_folder'
info_path = '../DEVKitArt/info.csv'
class_txt = 'sets/classes.txt'
predictions_path = 'evaluation_files/art_dl_test.csv'
cams_path = 'evaluation_files/ArtDL_cams'

# Numeric variables
num_classes = 10

# Torch variables
cuda0 = torch.device('cuda:0')

### Main
## Preparing the data
# Loading the names of the 10 classes in the ArtDL model
with open(class_txt, 'r') as f:
    classes = [l.strip() for l in f.readlines()]

# Making folders for the CAMs
# Comment this out if already run previously, cannot be done twice.
os.mkdir(f'{cams_path}/truth')
os.mkdir(f'{cams_path}/correct')
os.mkdir(f'{cams_path}/wrong')

for c in classes:
    os.mkdir(f'{cams_path}/truth/{c}')
    os.mkdir(f'{cams_path}/correct/{c}')
    os.mkdir(f'{cams_path}/wrong/{c}')
    os.mkdir(f'{cams_path}/truth/{c}_overlaps')
    os.mkdir(f'{cams_path}/correct/{c}_overlaps')
    os.mkdir(f'{cams_path}/wrong/{c}_overlaps')

# Loading the target and prediction dataframes
info = pd.read_csv(info_path)
predictions = pd.read_csv(predictions_path)
pred_keys = pd.Series(predictions.index, index=predictions.item).to_dict()

# Preparing column names for target df
cols = ['item'] + classes

# Target df has same items and columns as prediction df
target = info[cols]
target = target.loc[target['item'].isin(list(predictions['item']))]
target = target.reset_index(drop=True)
target_keys = pd.Series(target.index, index=target.item).to_dict()

# Saving dataframes.
predictions.to_csv(cams_path+'/predictions.csv', index=False)
target.to_csv(cams_path+'/target.csv', index=False)

# Column no longer needed, the key dicts maps indices and img name.
predictions = predictions.drop('item', axis=1)
target = target.drop('item', axis=1)

# Defining the transform at time of loading data
inference_transform = transforms.Compose([
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Colormap to have a sequential grayscale while retaining a visually pleasing
# heatmap in RGB for the CAM.
CMRmap = cm.get_cmap('jet') # returns RGBA

# Defining dataset and dataloader (for just the test set)
# test_set = ImageFolder(test_set_path, transform=inference_transform)
test_set = ImageFolder(test_set_path, transform=transforms.ToTensor())
dataloader = DataLoader(test_set)

## Preparing the model
# Defining the ArtDL convolutional model
model = resnet50.CAM(num_classes)

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
    original_image = data
    img_name = item[0]
    counter += 1

    print(f'Doing {counter}/{data_len}', end='\r')
    img_size = data.size()[-2], data.size()[-1]
    data = inference_transform(data)
    data = data.cuda()
    cams = model(data)
    high_res_cams = F.interpolate(cams,
        img_size, mode='bicubic', align_corners=False)
    cams = cams.squeeze()
    high_res_cams = high_res_cams.squeeze()

    # min-max normalization
    cams /= F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5
    high_res_cams /= F.adaptive_max_pool2d(high_res_cams, (1, 1)) + 1e-5

    # torch to numpy
    cams = cams.data.cpu().numpy()
    high_res_cams = high_res_cams.data.cpu().numpy()

    index = pred_keys[img_name]
    y_index = target_keys[img_name]
    prediction = predictions.iloc[index].to_numpy()
    y = target.iloc[y_index].to_numpy()

    # Original image Tensor to PIL
    og_image_pil = transforms.functional.to_pil_image(original_image.squeeze(0))

    # Prediction index
    id = prediction.nonzero()[0][0]
    y_id = y.nonzero()[0][0]
    class_name = classes[id]
    true_class = classes[y_id]

    # Getting the prediction CAM and changing mapping colors. Converting to
    # Tensor, fixing dimensions and converting to PIL.
    hrc = high_res_cams[id]
    hrc = CMRmap(hrc)
    hrc = hrc[:, :, :3] # Remove 4th dim from RGBA
    hrc = torch.from_numpy(hrc)
    hrc = hrc.permute(2, 0, 1)
    hrc = transforms.functional.to_pil_image(hrc)

    # Overlay of CAM and original image
    overlayed = Image.blend(og_image_pil, hrc, 0.5)

    # Logic for placing in correct folders.
    if (y == prediction).all():
        hrc.save(f'{cams_path}/correct/{class_name}/{img_name}_cam.jpg')
        overlayed.save(f'{cams_path}/correct/{class_name}_overlaps/{img_name}_cam_ol.jpg')
    else:
        hrc.save(f'{cams_path}/wrong/{class_name}/{img_name}_cam.jpg')
        overlayed.save(f'{cams_path}/wrong/{class_name}_overlaps/{img_name}_cam_ol.jpg')

        # If the prediction is wrong, also get the CAM for the ground truth.
        hrc = high_res_cams[y.nonzero()[0][0]]
        hrc = CMRmap(hrc)
        hrc = hrc[:, :, :3] # Remove 4th dim from RGBA
        hrc = torch.from_numpy(hrc)
        hrc = hrc.permute(2, 0, 1)
        hrc = transforms.functional.to_pil_image(hrc)
        hrc.save(f'{cams_path}/truth/{true_class}/{img_name}_pred_{class_name}.jpg')
        overlayed = Image.blend(og_image_pil, hrc, 0.5)
        overlayed.save(f'{cams_path}/truth/{true_class}_overlaps/{img_name}_pred_{class_name}.jpg')

print('\nDone')
