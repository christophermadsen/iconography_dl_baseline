""" Small script for copying images from a folder to another,
useful when restructuring for the torch dataset classes.
"""

from shutil import copy
import os
from collections import Counter

# Global variables
all_img_path = '../../DEVKitArt/JPEGImages/'

train = '../../DataFolders/train_folder/train'
val = '../../DataFolders/val_folder/val'
test = '../../DataFolders/test_folder/test'

train_list = open('../sets/train.txt', 'r').read().splitlines()
val_list = open('../sets/val.txt', 'r').read().splitlines()
test_list = open('../sets/test.txt', 'r').read().splitlines()

# Count of all files (duplicates too)
n = len(train_list) + len(val_list) + len(test_list)

def copy_all_no_duplicates():
    counter = 0
    for fname in train_list:
        copy(all_img_path+fname+'.jpg', train)
        counter += 1
        if counter % 1000 == 0:
            print(f'{counter}/{n}', end='\r')
    for fname in val_list:
        copy(all_img_path+fname+'.jpg', val)
        counter += 1
        if counter % 1000 == 0:
            print(f'{counter}/{n}', end='\r')
    for fname in test_list:
        copy(all_img_path+fname+'.jpg', test)
        counter += 1
        if counter % 1000 == 0:
            print(f'{counter}/{n}', end='\r')

copy_all_no_duplicates()
