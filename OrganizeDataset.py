import os
import os.path
import shutil
import csv
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def save_img(x, folder):
    img = x['filename']
    old_image_path = os.path.join(folder, img)
    new_image_path = os.path.join(x['folder_name'], img)
    if not os.path.exists(new_image_path):
        shutil.copy(old_image_path, new_image_path)

def organize_dataset(classes_to_keep, old_folder, train_ratio):
    folder_path = '../data/classif_' + classes_to_keep[0] + '_' + classes_to_keep[1]
    df = pd.read_csv('../data/train_info.csv')
    
    # créer le nouveau dossier rangé par style
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    images = [f for f in os.listdir(old_folder) if os.path.isfile(os.path.join(old_folder, f)) and f.endswith('.jpg')]
    print(len(images))
    
    # only keep the images that are in the folder to order
    df = df[df['filename'].isin(images)]
    
    # simplify style name
    df['style'] = [str(styl).strip().replace(' ', '_').lower() for styl in df['style']]
    
    # only keep the styles defined by class1 / class2
    df = df[df['style'].isin(classes_to_keep)].reset_index(drop = True)
    
    # create folder name in function of train/test split
    df['nb_tot'] = df.groupby(['style'])['filename'].transform('count')
    df['ind'] = df.groupby(['style']).cumcount()
    df['train_valid'] = [folder_path + '/train/' if float(ind)/float(tot) < train_ratio else folder_path +'/valid/' for (ind,tot) in zip(df['ind'],df['nb_tot']) ]
    df['folder_name'] = df['train_valid'] + df['style'] 
  
    for subfolder in list(df.folder_name.unique()):
        print(subfolder)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
            
    df.apply(lambda x : save_img(x, old_folder), axis = 1)
    return df
