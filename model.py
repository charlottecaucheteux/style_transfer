#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 18:48:01 2018

@author: charlotte.caucheteux
"""

class Model(object):
  
  def __init__(self, model_type, use_gpu = True):
      self.model_type = model_type
      self.use_gpu = use_gpu
      self.model = None

  def load_and_tune(self):
      if self.model_type == 'vgg':
          model_vgg = models.vgg16(pretrained=True)
          if self.use_gpu:
              print("using gpu version for the model")
              model_vgg = model_vgg.cuda()
              
          for param in model_vgg.parameters():
              param.requires_grad = False
              model_vgg.classifier._modules['6'] = nn.Linear(4096, 127)
          self.model = model_vgg
      
      if self.model_type == 'resnet':
          model_resnet = model_resnet.to(device)
          for param in model_resnet.parameters():
              param.requires_grad = False

          # Parameters of newly constructed modules have requires_grad=True by default
          num_ftrs = model_resnet.fc.in_features
          model_resnet.fc = nn.Linear(512, 2)
          self.model = model_resnet

  @property
  def features_block(self):
      if self.model_type == 'vgg':
          return(self.model.features)
      if self.model_type == 'resnet':
          model_resnet_conv = nn.Sequential(*list(self.model.children())[:-1])
          return(model_resnet_conv)

  @property    
  def classif_block(self):
      if self.model_type == 'vgg':
          return(self.model.classifier)
      if self.model_type == 'resnet':
          return(self.model.fc)

if __name__ == '__main__':
    m = Model('vgg')
    m.load_and_tune()
    fb = m.features_block
    classifier = m.classif_block