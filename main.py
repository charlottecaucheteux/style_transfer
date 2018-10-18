
import argparse
# import urllib
from class_dataloader import *




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Train/Run a ResNet/VGG Art Style classifier')
    parser.add_argument('-o', '--is_organised', type = str, nargs = '?', help = 'Whether the training directory is already organized or not', default = 'False')
    parser.add_argument('-c', '--class_names', type = list, nargs = '?', help = 'Name of the raw data directory', default = ['impressionism', 'realism'])
    parser.add_argument('-rd','--raw_data_dir', type = str, help = 'Name of the raw data directory')
    parser.add_argument('-dd', '--data_dir', type = str, help = 'Name of the training directory')
    args = parser.parse_args()
    

    # download the raw data from kaggle competition - need to be registered and to approave project
    #urllib.urlretrieve("https://www.kaggle.com/c/painter-by-numbers/download/train_1.zip", filename= "")
    #urllib.urlretrieve("https://www.kaggle.com/c/painter-by-numbers/download/train_2.zip", filename= "")
    
    loader = Dataloader(args.is_organised, args.class_names, args.raw_data_dir, args.data_dir)
    # print(loader.data_dir)
    if loader.is_organised == "False":
        print('ok')
        df1 = loader.organise_dataset(0.8, 1)
        df2 = loader.organise_dataset(0.8, 2)
        df = df1.append(pd.DataFrame(data = df2), ignore_index=True)
        print('Train, valid repartition: ', df.groupby(['train_valid', 'style'])['filename'].count())

    dataloader, data_sizes, class_names = loader.getDataLoader(loader.data_dir)

    print(dataloader)


