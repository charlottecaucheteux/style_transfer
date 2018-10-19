#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:54:16 2018

@author: mathildelavacquery
"""


import argparse
# import urllib
from class_dataloader import *
from features_generator import *
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Train/Run a ResNet/VGG Art Style classifier')
    parser.add_argument('-o', '--is_organised', type = str, nargs = '?', help = 'Whether the training directory is already organized or not', default = 'True')
    parser.add_argument('-c', '--class_names', type = list, nargs = '?', help = 'Name of the raw data directory', default = ['impressionism', 'realism'])
    parser.add_argument('-rd','--raw_data_dir', type = str, nargs= '?', help = 'Name of the raw data directory', default = 'None')
    parser.add_argument('-dd', '--data_dir', type = str, help = 'Name of the training directory')
    parser.add_argument('-sc', '--computeScalingFromScratch', type = str, nargs = '?', help = 'True if we need to compute the scaling parameters from scratch', default = 'False')
    parser.add_argument('-m', '--model', type = str, nargs = '?', help = 'model to use, either resnet or vgg16', default = 'resnet')
    parser.add_argument('-gf', '--generate_features', type = str, nargs = '?', help ='True to generate, False to use the already generated features', default = 'False')
    args = parser.parse_args()
    

    # download the raw data from kaggle competition - need to be registered and to approave project
    #urllib.urlretrieve("https://www.kaggle.com/c/painter-by-numbers/download/train_1.zip", filename= "")
    #urllib.urlretrieve("https://www.kaggle.com/c/painter-by-numbers/download/train_2.zip", filename= "")
    
    loader = Dataloader(args.is_organised, args.class_names, args.raw_data_dir, args.data_dir)
    if loader.is_organised == "False":
        print('ok')
        df1 = loader.organise_dataset(0.8, 1)
        df2 = loader.organise_dataset(0.8, 2)
        df = df1.append(pd.DataFrame(data = df2), ignore_index=True)
        print('Train, valid repartition: ', df.groupby(['train_valid', 'style'])['filename'].count())

    # Part 1 - dataloader
    dataloader, data_sizes, class_names = loader.getDataLoader(loader.data_dir, args.computeScalingFromScratch)
    
    # Part 2 - Features Generator
    output_dir = args.data_dir + '/' args.model
    if args.generate_features == 'True':
        fg = FeaturesGenerator(args.model, dataloader, output_dir, use_gpu = False)
        fg.generate()
        fg.plotPCA(class_names)
   
    conv_feat_train = load_array(output_dir +'/conv_feat_train.bc')
    labels_train = load_array(output_dir+'/labels_train.bc')
    conv_feat_val = load_array(output_dir+'/conv_feat_valid.bc')
    labels_val = load_array(output_dir+'/labels_valid.bc')



    # Part 3 - train the model 

    




   
        
