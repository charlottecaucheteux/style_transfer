from torchvision import transforms,datasets
import os
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

def getDataLoader(data_dir, batch_size=4):

    rgb_mean, rgb_std = getScaleParameters(data_dir)
    print(rgb_mean, rgb_std)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)
        ]),
    }


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=6)
                      for x in ['train', 'valid']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    return dataloaders


def getScaleParameters(data_dir):
    try:
        params = pickle.load("scaleParams.P")
        return params
    except:
        print("Computing the parameters")
        params = computeScaleParameters(data_dir)
        pickle.dump(params, "scaleParams.P")
        return params

def computeScaleParameters(data_dir):
    train_dir = os.path.join(data_dir, "train")
    n_pixels_glob = 0
    
    r_full_sum=0
    g_full_sum=0
    b_full_sum=0

    r_full_sumsquared=0
    g_full_sumsquared=0
    b_full_sumsquared=0

    for class_folders in os.listdir(train_dir):
        class_folders_path = os.path.join(train_dir, class_folders)
        for imgName in os.listdir(class_folders_path):
            image = Image.open(os.path.join(class_folders_path, imgName))
            width, height = image.size
            pixel_values = list(image.getdata())

            size = width*height

            r_values = [pixel_values[i][0] for i in range (size)]
            g_values = [pixel_values[i][1] for i in range (size)]
            b_values = [pixel_values[i][2] for i in range (size)]

            r_values_sqr = [pixel_values[i][0]*2 for i in range (size)]
            g_values_sqr = [pixel_values[i][1]*2 for i in range (size)]
            b_values_sqr = [pixel_values[i][2]*2 for i in range (size)]

            n_pixels_glob += width*height

            r_full_sum += sum(r_values)
            g_full_sum += sum(g_values)
            b_full_sum += sum(b_values)

            r_full_sumsquared += sum(r_values_sqr)
            g_full_sumsquared += sum(g_values_sqr)
            b_full_sumsquared += sum(b_values_sqr)


    r_full_mean = r_full_sum/n_pixels_glob
    g_full_mean = g_full_sum/n_pixels_glob
    b_full_mean = b_full_sum/n_pixels_glob

    r_full_std = r_full_sumsquared/n_pixels_glob - r_full_mean**2
    g_full_std = g_full_sumsquared/n_pixels_glob - g_full_mean**2
    b_full_std = b_full_sumsquared/n_pixels_glob - b_full_mean**2

    return [r_full_mean, g_full_mean, b_full_mean]/255, [r_full_std, g_full_std, b_full_std]/255

# data_dir = 'D:/HEC/Cours/dpx/artLearningProject/dataset/'
# getDataLoader(data_dir)

