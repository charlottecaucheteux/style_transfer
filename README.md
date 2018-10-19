# Style Classification


### 1. Use our drive (data and functions are already downloaded) : 
https://drive.google.com/drive/folders/1TxWlhFs9hKlKEzAbL6Mw6PHLIWjCF7-w?usp=sharing

### 2. You can run directly the notebook : Notebook art classification project, in the drive 

## OR 

### 1. Download the data from these URL:
https://www.kaggle.com/c/painter-by-numbers/download/train_1.zip

https://www.kaggle.com/c/painter-by-numbers/download/train_2.zip

https://www.kaggle.com/c/painter-by-numbers/data/train_info.csv

ATTENTION : need to have a kaggle account and to approave the project

### 2. Unzip the files in the directory raw_data, organisation of the directory should be:

```bash
├── raw_data
│   ├── train_1
│   ├── train_2
│   └── train_info.csv
├── data
│   ├── classif_class1_class2
│   │   ├── train
│   │   └── valid
│   ├── resnet
│   └── vgg16
├── main.py
├── class_dataloader.py
├── features_generator.py
├── model.py
├── trainer.py
├── utils.py
└── scaleParams.P
```
 The data folder will be generated automatically by the DataLoader.
 The file scaleParams.P will be generated automatically by the Dataloader.
 The folders resnet and vgg16 will be generated automatically by the FeaturesGenerator.

### 3. For the first resnet training, run:
```bash
python main.py -o False -rd raw_data -dd data -m resnet -sc True -gf True 
```
### 4. Then, when the features are already ccomputed, just run:
```bash
python main.py -dd data
```
### 4. All parameters available:
```bash
'-gpu', '--use_gpu'       Whether to use GPU or not, default = False
'-o', '--is_organised'    Whether the directory is already organised or not, default = True
'-c', '--class_names'     Painting styles to classify, default = ['impressionism', 'realism']
'-rd','--raw_data_dir'    Name of the raw_data directory where the dataset has been unziped, default = 'None'
'-dd', '--data_dir'       Name of the data directory, compulsory
'-sc', '--computeScalingFromScratch'
'-m', '--model'
'-gf', '--generate_features'
```







