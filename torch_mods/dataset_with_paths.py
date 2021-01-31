import os
from torchvision.datasets import ImageFolder
import torch

class ImageFolderWithPaths(ImageFolder):
    """Mods ImageFolder to return (data, (path)) instead of (data, target)
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # make a new tuple that includes original and the path
        file_name = path.replace('\\','/').split('/')[-1]
        item_name = os.path.splitext(file_name)[0]

        # 2nd index is path instead of 'target'
        return original_tuple[0], item_name
