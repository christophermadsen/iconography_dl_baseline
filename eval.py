import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import classification_report

info_path = '../DEVKitArt/info.csv'
input = input('Get metrics from VGG16 model 1/2/3 or ArtDL model? A: (VGG1 / VGG2 / VGG3 / ART) ')
class_txt = 'sets/vgg_classes.txt'

# for Resnet50
if input == 'ART':
    output_path = 'evaluation_files/art_dl_test.csv'
    class_txt = 'sets/classes.txt'

# for vgg16
elif input == 'VGG1':
    output_path = 'best_vgg_1st_strategy/vgg_test_final_1st_strategy.csv'

elif input == 'VGG2':
    output_path = 'best_vgg_2nd_strategy/vgg_test_final_2nd_strategy.csv'

elif input == 'VGG3':
    output_path = 'best_vgg_3rd_strategy/vgg_test_final.csv'

else:
    print('Try again with a valid input.')
    quit()

info = pd.read_csv(info_path)
output = pd.read_csv(output_path)

with open(class_txt, 'r') as f:
    classes = [l.strip() for l in f.readlines()]

cols = ['item'] + classes

# ----------------------------------------------

# Target has same items and columns as output
target = info[cols]
target = target.loc[target['item'].isin(list(output['item']))]

# Make items the indices of target and output dataframes
output = output.set_index('item')
target = target.set_index('item')

# Getting metrics
# M_0,0 = TN    M_1,0 = FN    M_1,1 = TP    M_0,1 = FP   # matrix coordinates+
confusion_matrix = multilabel_confusion_matrix(target, output)
report = classification_report(target, output)

print()
print('Metrics report for all classes')
print(report)
print()
print('Classes for reference')
print(classes)
print()
