# Style Classification

This project aims at classify paintings according to their style. We reached 68% of accuracy on a test set of 20% of the images while trying to distinguish impressionism from realism.

We used a deep learning approach, with transfer learning from a VGG19 and a residual neural network (ResNet18).

## Getting Started

### Run on the cloud

Use the [drive](https://drive.google.com/drive/folders/1TxWlhFs9hKlKEzAbL6Mw6PHLIWjCF7-w?usp=sharing) directly

Run the Notebook art classification project.

### Run locally

#### The data
Download the data from these URL:
- [train1](https://www.kaggle.com/c/painter-by-numbers/download/train_1.zip)
- [train2](https://www.kaggle.com/c/painter-by-numbers/download/train_2.zip)
- [train_info](https://www.kaggle.com/c/painter-by-numbers/data/train_info.csv)

(You need to have a kaggle account and to approve the project)

Unzip the files in the directory raw_data, organisation of the directory should be:

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

#### The training:
```bash
# compute features
python main.py -o False -rd raw_data -dd data -m resnet -sc True -gf True 
# train
python main.py -dd data
```
Parameters available:
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

## Acknowledgments

- [Recognizing Art Style Automatically in painting with deep learning (Adrian Lecoutre, Benjamin Negrevergne, Florian Yger, 2017)](http://www.lamsade.dauphine.fr/~bnegrevergne/webpage/documents/2017_rasta.pdf)

- [Kaggle](www.kaggle.com) competition






