""" Makes folders with the name of the ArtDL classes. Useful for torch
Dataloaders.
"""

import os

class_txt = '../sets/classes.txt'

input = input('Make folder for expanded data folders or without duplicates? A: (1 / 2)')

if input == '1':
    print('Making folders for expanded data set')
    folder_name = 'data_by_class'
else:
    print('Making folders for data set without duplicates')
    folder_name = 'data_by_class_no_dup'

with open(class_txt, 'r') as f:
    classes = [l.strip() for l in f.readlines()]

for c in classes:
    os.mkdir(f'../../DataFolders/{folder_name}/train/{c}')
    os.mkdir(f'../../DataFolders/{folder_name}/val/{c}')
