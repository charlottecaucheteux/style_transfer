#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:54:16 2018

@author: charlotte.caucheteux
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models,transforms,datasets
import torch
import bcolz
import time


class FeaturesGenerator(object):
    
    def __init__(self, model_type, dataloader, output_dir, use_gpu = False):
        self.model_type = model_type
        self.dataloader = dataloader #dictionnaire avec train et valid dedans 
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.model = None
        self.features = None
        self.labels = None

    def get_preconvfeat_model(self):
        
        if self.model_type == 'resnet':
            model_resnet = torchvision.models.resnet18(pretrained=True)
            if self.use_gpu:
                model_resnet = model_resnet.cuda()
            for param in model_resnet.parameters():
                param.requires_grad = False
                    
            # Parameters of newly constructed modules have requires_grad=True by default
            num_ftrs = model_resnet.fc.in_features
            model_resnet.fc = nn.Linear(512, 2)
            model_features_block = nn.Sequential(*list(model_resnet.children())[:-1])
         
        if self.model_type == 'vgg16':
            model_vgg = models.vgg16(pretrained=True)
            if self.use_gpu:
                model_vgg = model_vgg.cuda()

            for param in model_vgg.parameters():
                param.requires_grad = False
                model_vgg.classifier._modules['6'] = nn.Linear(4096, 127)
            model_features_block = model_vgg.features
         
        
        self.model = model_features_block
        return(model_features_block)
        
    def compute_preconvfeat(self, model, dataset, save_dir_features, save_dir_labels, save_batch = 10):
        conv_features = []
        labels_list   = []
        i = 0
        print(len(dataset))
        for data in dataset:
            i = i + 1
            print(i)
            try : 
                inputs, labels = data
                if self.use_gpu:
                    inputs , labels = Variable(inputs.cuda()),Variable(labels.cuda())
                else:
                    inputs , labels = Variable(inputs),Variable(labels)
                x = model(inputs)
                print(x)
                #pdb.set_trace()
                conv_features.extend(x.data.cpu().numpy())
                labels_list.extend(labels.data.cpu().numpy())
                
                if i % save_batch == 0:
                    save_array(save_dir_features, conv_features)
                    save_array(save_dir_labels, labels_list)

            except (OSError, ValueError, IOError):
                print(str(i) + " image is not used ! " )
             
        conv_features = np.concatenate([[feat] for feat in conv_features])
        save_array(save_dir_features, conv_features)
        save_array(save_dir_labels, labels_list)   
            
        return (conv_features, labels_list) 
  
    def generate(self):
        model = self.get_preconvfeat_model()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        for phase in ['valid', 'train']:
            dataset = self.dataloader[phase]
            save_dir_features = os.path.join(self.output_dir, 'conv_feat_'+phase+'.bc')
            save_dir_labels = os.path.join(self.output_dir, 'labels_'+phase+'.bc')
            features, labels = self.compute_preconvfeat(model,dataset, save_dir_features, save_dir_labels)
        self.features_train, self.labels_train = features
    
    
    def plotPCA(self):
        X = np.array([x.flatten() for x in self.features])
        y = np.array(self.labels)
        label_names = self.dataloader.class_names
        
        pca = PCA(n_components=2)
        X_pca = pca.fit(X).transform(X)
        
        colors = ['navy', 'turquoise']
        
        plt.figure(figsize=(8, 8))
    #    for color, i, target_name in zip(colors, [0, 1], iris.target_names):
        for color, i, target_name in zip(colors, [0, 1], label_names):
            plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                        color=color, lw=2, label=target_name, s = 1)
    
        plt.title("PCA")
        plt.legend()
        plt.show()




if __name__ == 'main':
    output_dir = 'data/classif_impressionism_realism/resnet_test'
    fg = FeaturesGenerator('resnet', dataloaders, output_dir, use_gpu = True)
    fg.generate()
    fg.plotPCA()
