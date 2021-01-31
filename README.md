# An Explainable Deep Learning Baseline for Iconography Research in Artworks
## Author: Christopher Buch Madsen

This repository contains the codebase for the thesis "An Explainable Deep Learning Baseline for Iconography Research in Artworks" written by Christopher Buch Madsen while attending the Bachelor of AI at the University of Amsterdam. The thesis was delivered 29 January, 2021.

### Overview of how to run the code.

#### Requirements
    Python >= 3.9
    PyTorch >= 1.7.1+cu110
    Numpy
    Pandas
    Matplotlib
    Sklearn (>= 0.24.1)
    ipynb (for examples)
    
#### Setup
1. Clone this repository.
2. Download the ArtDL data set at: <br/>
http://www.artdl.org/ <br/>
Unzip the files and place the "DEVKitArt" folder in the folder prior to this repository.
3. Create a folder named "DataFolders" at the same destination as DEVKitArt <br/>
The directory tree should at this point look like: <br/>
| <br/>
|───DEVKitArt <br/>
| <br/>
|───DataFolders <br/>
| <br/>
|───main    <--- the project repository <br/>
| <br/>
...
4. Run the following python files from main/preprocessing consecutively:
    - seclude_img.py
    - make_class_folders.py
    - sort_data_by_folders.py
    - sort_data_by_folder_no_dup.py
This will set up the data in sorted folders, necessary for the project, the directory tree should now be:
| <br/>
|───DEVKitArt <br/>
| <br/>
|───DataFolders <br/>
|   |───data_by_class <br/>
|   |───data_by_class_no_dup <br/>
|   |───test_folder <br/>
|       |───test <br/>
|   |───train_folder <br/>
|       |───train <br/>
|   |───val_folder <br/>
|       |───val <br/>
|───main    <--- the project repository <br/>
| <br/>
...
5. Fourth item
