#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:54:33 2018

@author: charlotte.caucheteux
"""

class Trainer(object):
    
    def __init__(self, input_path, size_dic, model_type = 'vgg'):
        self.model_type = model_type
        self.input_path = input_path
        self.size_dic = size_dic
        self.losses = None
        self.conv_feat_train = None
        self.labels_train = None
        self.conv_feat_val = None
        self.labels_val = None
        self.model = None
      
    #TODOOOO
    def init_model(self):
        if self.model_type == 'vgg':
            kjh
            
    def data_gen(self, conv_feat,labels,batch_size=64,shuffle=True):
        labels = np.array(labels)
        if shuffle:
            index = np.random.permutation(len(conv_feat))
            conv_feat = conv_feat[index]
            labels = labels[index]
        for idx in range(0,len(conv_feat),batch_size):
            yield(conv_feat[idx:idx+batch_size],labels[idx:idx+batch_size])
        
    def load_features(self):
        self.conv_feat_train = load_array(data_dir+'/resnet/conv_feat_train.bc')
        self.labels_train = load_array(data_dir+'/resnet/labels_train.bc')
        self.conv_feat_val = load_array(data_dir+'/resnet/conv_feat_val.bc')
        self.labels_val = load_array(data_dir+'/resnet/labels_val.bc')
    
    
    def train(self, model, size_dic, conv_feat=None,labels=None,epochs=1,optimizer=None,train=True,shuffle=True):
              #model, size_dic, conv_feat=None,labels=None,epochs=1,optimizer=None,train=True,shuffle=True):
        self.load_features()
        
        losses = {}
        accuracies = {}
        for phase in ['train', 'valid']:
          losses[phase] = []
          accuracies[phase] = []
            
        for epoch in range(epochs):
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                batches = data_gen(conv_feat=conv_feat[phase],labels=labels[phase],shuffle=shuffle)
                total = 0
                running_loss = 0.0
                running_corrects = 0
                for inputs,classes in batches:
                    if use_gpu:
                        inputs , classes = torch.from_numpy(inputs).cuda(), torch.from_numpy(classes).cuda()
                    else:
                        inputs , classes = torch.from_numpy(inputs), torch.from_numpy(classes)
    
                    inputs = inputs.view(inputs.size(0), -1)
                    outputs = model(inputs) #contain weights
                    loss = criterion(outputs, classes)           
                    #if train:
                    if phase == 'train':
                        if optimizer is None:
                            raise ValueError('Pass optimizer for train mode')
                        optimizer = optimizer
                        optimizer.zero_grad()
                        ##loss.backward() compute the gradient automatically for all the network 
                        loss.backward() ##gradient
                        optimizer.step() ##step of the optimizer
                    _,preds = torch.max(outputs.data,1)
                    # statistics
                    running_loss += loss.data.item()
                    running_corrects += torch.sum(preds == classes.data)
    
                epoch_loss = running_loss / size_dic[phase]
                epoch_acc = running_corrects.data.item() / size_dic[phase]
                print(phase + ' - Loss: {:.4f} Acc: {:.4f}'.format(
                             epoch_loss, epoch_acc))
                losses[phase].append(epoch_loss)
                accuracies[phase].append(epoch_acc)
        self.losses = losses
        self.accuracies = accuracies
    return(losses, accuracies)
    
    def plot_losses(self):
        dic = self.accuracies
        title = 'Loss curves rgd the number of epochs'
        yt = dic['train']
        yv = dic['valid']
        x = range(len(yt))
        plt.plot(x, yt, 'r', label = 'train')
        plt.plot(x, yv, 'b', label = 'valid')
        plt.title(title)
        plt.legend()
        plt.show()        

    