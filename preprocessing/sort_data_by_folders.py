""" Small script for copying images into folders by their class name, useful for
working with Torch Dataloaders.
"""

from shutil import copy
import os
from collections import Counter
import pandas as pd

# Global variables
all_img_path = '../../DEVKitArt/JPEGImages/'
train_list = open('../sets/train.txt', 'r').read().splitlines()
val_list = open('../sets/val.txt', 'r').read().splitlines()
desti = '../../DataFolders/data_by_class/'
info_path = '../../DEVKitArt/info.csv'
class_txt = '../sets/classes.txt'

# Load class names
with open(class_txt, 'r') as f:
    classes = [l.strip() for l in f.readlines()]

# Get targets for files 
cols = ['item'] + classes
info = pd.read_csv(info_path)
target = info[cols]
target_keys = pd.Series(target.index, index=target.item).to_dict()
target = target.drop('item', axis=1)
cols = cols[1:]

# Copy training files
n = len(train_list)
counter = 0
visited_names = []
for fname in train_list:
    ind = target_keys[fname]
    y = target.iloc[ind].to_numpy()
    y = cols[y.nonzero()[0][0]]
    if fname+'.jpg' in visited_names:
        for i in range(1000):
            new_name = fname+f'_{i}.jpg'
            if new_name not in visited_names:
                copy(all_img_path+fname+'.jpg', f'{desti}train/{y}/{new_name}')
                visited_names.append(new_name)
                break
    else:
        copy(all_img_path+fname+'.jpg', f'{desti}train/{y}/{fname}.jpg')
        visited_names.append(f'{fname}.jpg')

    counter += 1
    if counter % 1000 == 0:
        print(f'Doing {counter}/{n}', end='\r')

# Copy validation files
for fname in val_list:
    ind = target_keys[fname]
    y = target.iloc[ind].to_numpy()
    y = cols[y.nonzero()[0][0]]
    copy(all_img_path+fname+'.jpg', f'{desti}val/{y}/{fname}.jpg')
