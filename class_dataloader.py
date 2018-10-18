from torchvision import transforms,datasets
import os
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

def save_img(x, folder):
    img = x['filename']
    old_image_path = os.path.join(folder, img)
    new_image_path = os.path.join(x['folder_name'], img)
    if not os.path.exists(new_image_path):
        shutil.copy(old_image_path, new_image_path)

class Dataloader(nn.Module):
    def __init__(self, is_organised, num_class, class_names, raw_data_dir, data_dir):
        self.is_organised = is_organised
        self.num_class = num_class
        self.class_names = class_names
        self.raw_data_dir = raw_data_dir
        self.data_dir = data_dir

    # Dealing with the raw data
    def organize_dataset(self, train_ratio):
        df = pd.read_csv(self.raw_data_dir + '/train_info.csv')
        if self.is_organised not True:
            folder_path = self.data_dir + '/classif_' + self.class_names[0] + '_' + self.class_names[1]

            # créer le nouveau dossier rangé par style
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            images = [f for f in os.listdir(self.raw_data_dir) if os.path.isfile(os.path.join(self.raw_data_dir, f)) and f.endswith('.jpg')]
            print(len(images))

            # only keep the images that are in the folder to order
            df = df[df['filename'].isin(images)]

            # simplify style name
            df['style'] = [str(styl).strip().replace(' ', '_').lower() for styl in df['style']]

            # only keep the styles defined by class1 / class2
            df = df[df['style'].isin(self.class_names)].reset_index(drop = True)

            # create folder name in function of train/test split
            df['nb_tot'] = df.groupby(['style'])['filename'].transform('count')
            df['ind'] = df.groupby(['style']).cumcount()
            df['train_valid'] = [folder_path + '/train/' if float(ind)/float(tot) < train_ratio else folder_path +'/valid/' for (ind,tot) in zip(df['ind'],df['nb_tot']) ]
            df['folder_name'] = df['train_valid'] + df['style']

            for subfolder in list(df.folder_name.unique()):
                print(subfolder)
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)

            df.apply(lambda x : save_img(x, self.raw_data_dir), axis = 1)
            self.is_organised = True
        return df

    # Retrieving or computing the scaling parameters as two lists (mean, std) of 3 values (R, G, B)
    def computeScaleParameters(self):
        train_dir = os.path.join(self.data_dir, "train")
        n_pixels_glob = 0

        r_full_sum=0
        g_full_sum=0
        b_full_sum=0

        r_full_sumsquared=0
        g_full_sumsquared=0
        b_full_sumsquared=0
        # Reading all images, summing the values of the pixel and the value squared
        for class_folders in os.listdir(train_dir):
            if class_folders != '.DS_Store':
                class_folders_path = os.path.join(train_dir, class_folders)
                for imgName in os.listdir(class_folders_path):
                    try:
                        image = Image.open(os.path.join(class_folders_path, imgName))
                        width, height = image.size
                        pixel_values = list(image.getdata())


                        size = width*height
                        try:
                            r_values = [pixel_values[i][0] for i in range (size)]
                            g_values = [pixel_values[i][1] for i in range (size)]
                            b_values = [pixel_values[i][2] for i in range (size)]

                            r_values_sqr = [pixel_values[i][0]*2 for i in range (size)]
                            g_values_sqr = [pixel_values[i][1]*2 for i in range (size)]
                            b_values_sqr = [pixel_values[i][2]*2 for i in range (size)]


                            r_full_sum += sum(r_values)
                            g_full_sum += sum(g_values)
                            b_full_sum += sum(b_values)

                            r_full_sumsquared += sum(r_values_sqr)
                            g_full_sumsquared += sum(g_values_sqr)
                            b_full_sumsquared += sum(b_values_sqr)

                            n_pixels_glob += width*height
                        except:
                            pass
                    except:
                        pass
        # Computing mean and std from previous sums
        r_full_mean = r_full_sum/n_pixels_glob
        g_full_mean = g_full_sum/n_pixels_glob
        b_full_mean = b_full_sum/n_pixels_glob

        r_full_std = r_full_sumsquared/n_pixels_glob - r_full_mean**2
        g_full_std = g_full_sumsquared/n_pixels_glob - g_full_mean**2
        b_full_std = b_full_sumsquared/n_pixels_glob - b_full_mean**2

        return [r_full_mean/255, g_full_mean/255, b_full_mean/255], [r_full_std/255, g_full_std/255, b_full_std/255]

    # Check if the RGB scaling parameters have been computed
    def getScaleParameters(self):
        try:
            # Trying to retrieve
            with open('scaleParams.P', 'rb') as input:
                return(pickle.load(input))
        except:
            # If not present, computing.
            print("Computing the parameters")
            train_dir = os.path.join(self.data_dir, "train")
            params = computeScaleParameters(self.data_dir)
            # Writing for future use
            with open('scaleParams.P', 'wb') as output:
                pickle.dump(params, output)
            return params

    # DataLoader
    # input: directory of data organized with train/classes and valid/classes
    # output: dataloaders, data_sizes, class_names used later in computation. Data transformation is done on the fly.

    def getDataLoader(self, batch_size=4):

        # Computing scaling parameters
        rgb_mean, rgb_std = getScaleParameters(self.data_dir)
        print("Normalizing the images with the following parameters for mean and std")
        print(rgb_mean, rgb_std)

        # For the data augmentation
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

        # Creating ImageFolder object
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'valid']}


        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                     shuffle=True, num_workers=6)
                          for x in ['train', 'valid']}

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
        class_names = image_datasets['train'].classes


        # Returning dataloader and metadata only
        return dataloaders, data_sizes, class_names


# data_dir = '/content/drive/My Drive/DeepLearningProject/data/classif_style1/'
# getDataLoader(data_dir)
