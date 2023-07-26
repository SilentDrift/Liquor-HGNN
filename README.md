# Liquor-HGNN
This repository is the official implementation of the paper 'Liquor-HGNN: introducing new head-loss based edge weights for water distribution networks'.

## Requirements
```
pip install torch --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Datasets

BattleDIM Dataset can be downloaded here: [(https://drive.google.com/drive/folders/1aNVGmHvBOA0zytIC4F8DWWbNPNBykimf?usp=drive_link)
please create a dataset folder in the repository and load the data from there


## Training and Evaluation


Create predicted demands:
```
python expected_demands.py
```

Train and evaluate model
```
python train.py

optional arguments:                                                                                                                      
  -h, --help           show this help message and exit                                                                                   
  --layer LAYER        options: 'GATConv', 'SAGEConv', 'GraphConv', 'LEConv' (default: 'GATConv')                                                             
  --epochs N           (default: 50)                                                                                                                  
  --hidden-channels N  (default: 64)                                                                                                                  
  --batch-size N       (default: 512)                                                                                                                  
  --lr lr              initial learning rate for optimizer e.g.: 1e-4 | 'auto' (default: 'auto')                                                          
  --num-layer N        (default: 10)                                                                                                                  
  --seed N             (default: 0)                                                                                                                  
  --no-cuda            (default: False)
```
Evaluate pretrained model
```
python eval.py Path

positional arguments:
  Path                 path to pretrained weights, e.g. 'models/pretrained/GATConv.statedict'

optional arguments:                                                                                                                      
  -h, --help           show this help message and exit                                                                                   
  --layer LAYER        options: 'GATConv', 'SAGEConv', 'GraphConv', 'LEConv' (default: 'GATConv')                                                                                                                  
  --hidden-channels N  (default: 64)                                                                                                                  
  --batch-size N       (default: 512)                                                      
  --num-layer N        (default: 10)                                                                                                                  
  --seed N             (default: 0)                                                                                                                  
  --no-cuda            (default: False)
```


## Results

To get the economic score for the BattleDIM competition dataset visit: https://github.com/KIOS-Research/BattLeDIM

Our model achieves the following performance:

GATConv:

```
Total score: € 282785.1
True Positives: 11
False Positives: 51
False Negatives: 12
```
```
    Team Name & True Positive Rate (%) & False Positives & Economic Score
    
     Liquor-HGNN & 47.83 & 51 & 282,785 
     Tongji-Team & 56.52 & 3 & 264,873 
     Under Pressure & 65.22 & 4 & 260,562
     IRI & 43.47 & 1 & 210,772
     Leakbusters & 47.83 & 7 & 195,490
     Tsinghua & 47.83 & 5 & 167,981
     UNIFE & 43.47 & 4 & 127,626
```
