""" Small function for getting the means and standard deviations of each channel
for a batched tensor image dataset.
"""

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

all_img_path = '../../DEVKitArt/JPEGImages/'
img_folder_path = '../../DEVKitArt/'
data_path = ''

def get_mean_std(dataloader, data_len):
    print(f'0/{data_len}')
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
        if num_batches % 1000 == 0:
            print(f"Progress: {num_batches}/{data_len}", end='\r')

    # VAR[X] = E[X**2] - E[X]**2
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean ** 2) ** 0.5
    return mean, std

dataset = ImageFolder(root=data_path, transform=transforms.ToTensor())
loader = DataLoader(dataset=dataset, batch_size=1)
mean, std = get_mean_std(loader, len(dataset))

print(f"Mean values of data set: {mean}\nStd of data set: {std}")
