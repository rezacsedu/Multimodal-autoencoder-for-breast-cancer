## Multimodal Autoencoders for Cancer Genomics
Implementation of our paper titled "Prognostically Relevant Subtypes and Survival Prediction for Breast Cancer Based on Multimodal Genomics Data" submitted to IEEE Access journal, August 2019.

In this implementation, Multimodal Autoencoders(MAE) is used to predict the clinical status of breast cancer patients based on their genomics data. This MAE model is trained with breast cancer genomics data from The Cancer Genome Atlas Breast Invasive Carcinoma (TCGA-BRCA). 

### Predicted clinical status
1. Breast cancer subtypes which is determined by the estrogen receptor (ER), progesterone receptor (PGR), and HER2/neu status
2. Survival rate (0-1, with 1 being the best chance of survival)

### Requirements
* Python 3
* TensorFlow
* Keras. 

### Download and create the dataset
* Open the terminal
* Clone the repo using `git clone https://github.com/rezacsedu/MultimodalAE-BreastCancer.git`
* Run the dataset creation program `python3 main_download.py -d DATASET_IDX`.

| DATASET_IDX |                      Data Types                      | Space Requirements (GB) |
|------------:|:-----------------------------------------------------|:-----------------------:|
|           1 | DNA Methylation                                      |          148            |
|           2 | Gene Expression                                      |          9              |
|           3 | miRNA Expression                                     |          0.24           |
|           4 | Gene Expression + miRNA Expression                   |          10             |
|           5 | DNA Methylation + Gene Expression + miRNA Expression |          162            |

### Train the neural networks
* Open the terminal
* Run the neural networks program `python3 main_run.py <options>`, with the below supported options: 

|               Option               |   Values   |                                                                                                                                                                                                                                                                                                                                                                                                              Details                                                                                                                                                                                            | Required |
|-----------------------------------:|:-----------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------:|
| -p PLATFORM<br>--platform PLATFORM | int [1-2]  | [1] Tensorflow, [2] Theano                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |    yes   |
|             -t TYPE<br>--type TYPE | int [1-2]  | [1] Breast cancer type classification<br>[2] Survival rate regression                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |    yes   |
|    -d DATASET<br>--dataset DATASET | int [1-15] | [1] DNA Methylation GPL8490<br>[2] DNA Methylation GPL16304<br>[3] Gene Expression Count<br>[4] Gene Expression FPKM<br>[5] Gene Expression FPKM-UQ<br>[6] miRNA Expression<br>[7] Gene Expression Count + miRNA Expression<br>[8] Gene Expression FPKM + miRNA Expression<br>[9] Gene Expression FPKM-UQ + miRNA Expression<br>[10] DNA Met GPL8490 + Gene Count + miRNA<br>[11] DNA Met GPL16304 + Gene Count + miRNA<br>[12] DNA Met GPL8490 + Gene FPKM + miRNA<br>[13] DNA Met GPL16304 + Gene FPKM + miRNA<br>[14] DNA Met GPL8490 + Gene FPKM-UQ + miRNA<br>[15] DNA Met GPL16304 + Gene FPKM-UQ + miRNA |    yes   |
|              --pretrain_epoch PRE_EPOCH | int        | Pre-training epoch. Default = 100                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |    no    |
|          --train_epoch TRAIN_EPOCH | int        | Training epoch. Default = 100                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |    no    |
|                      --batch BATCH | int        | Batch size for pre-training and training. Default = 10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |    no    |
|                    --pre_lr PRE_LR | int        | Pre-training learning rate. Default = 0.01                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |    no    |
|                --train_lr TRAIN_LR | int        | Training learning rate. Default = 0.1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |    no    |
|                  --dropout DROPOUT | int        | Dropout rate. Default = 0.2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |    no    |
|                          --pca PCA | int [1-2]  | [1] Use PCA<br>[2] Don't use PCA<br>Default = [2] Don't use                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |    no    |
|              --optimizer OPTIMIZER | int [1-3]  | [1] Stochastic gradient descent<br>[2] RMSProp<br>[3] Adam<br>Default = [1] Stochastic gradient descent                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |    no    |

### Example
If we want to perform breast cancer subtype classification based on the dime sion reduced DNA methylation dataset using PCA on TensorFlow platform, one can issue the following command from the terminal: 
`python3 main_run.py --platform 1 --type 1 --dataset 1 --batch 10 --pretrain_epoch 5 --train_epoch 5 --pca 1 --optimizer 3`

In the preceding command, we define:
-- 10 as the batch size
-- 5 as the number of pretraining epoch
-- 5 is the fine tuning epoch
-- 3 is the idx for the Adam optimizer. 

### #Sample execution: 
Cancer Type Classification with DNA Methylation Platform GPL8490 (TensorFlow)

    ER Status Prediction
    -----------------------------
    [START] Pre-training step:
    >> Epoch 1 finished     RBM Reconstruction error 522.190925
    >> Epoch 2 finished     RBM Reconstruction error 497.765570
    >> Epoch 3 finished     RBM Reconstruction error 492.680869
    >> Epoch 4 finished     RBM Reconstruction error 494.515497
    >> Epoch 5 finished     RBM Reconstruction error 468.050771
    >> Epoch 1 finished     RBM Reconstruction error 2680.144531
    >> Epoch 2 finished     RBM Reconstruction error 2672.767578
    >> Epoch 3 finished     RBM Reconstruction error 2691.162842
    >> Epoch 4 finished     RBM Reconstruction error 2597.989502
    >> Epoch 5 finished     RBM Reconstruction error 2758.419678
    [END] Pre-training step
    
    [START] Fine tuning step:
    >> Epoch 0 finished     ANN training loss 0.610027
    >> Epoch 1 finished     ANN training loss 0.594821
    >> Epoch 2 finished     ANN training loss 0.568818
    >> Epoch 3 finished     ANN training loss 0.564796
    >> Epoch 4 finished     ANN training loss 0.558171
    [END] Fine tuning step
    
    Accuracy: 0.786260
    Precision: 0.6182040673620418
    Recall: 0.7862595419847328
    F1-score: 0.6921772036275853

    PGR Status Prediction
    ---------------------------------
    [START] Pre-training step:
    >> Epoch 1 finished     RBM Reconstruction error 422.876587
    >> Epoch 2 finished     RBM Reconstruction error 393.641800
    >> Epoch 3 finished     RBM Reconstruction error 377.866021
    >> Epoch 4 finished     RBM Reconstruction error 368.311999
    >> Epoch 5 finished     RBM Reconstruction error 380.356941
    >> Epoch 1 finished     RBM Reconstruction error 2793.383789
    >> Epoch 2 finished     RBM Reconstruction error 2742.516602
    >> Epoch 3 finished     RBM Reconstruction error 2704.654785
    >> Epoch 4 finished     RBM Reconstruction error 2839.105469
    >> Epoch 5 finished     RBM Reconstruction error 2749.048584
    [END] Pre-training step
    
    [START] Fine tuning step:
    >> Epoch 0 finished     ANN training loss 0.921267
    >> Epoch 1 finished     ANN training loss 0.662474
    >> Epoch 2 finished     ANN training loss 0.674687
    >> Epoch 3 finished     ANN training loss 0.669110
    >> Epoch 4 finished     ANN training loss 0.739354
    [END] Fine tuning step
    
    Accuracy: 0.694656
    Precision: 0.48254763708408605
    Recall: 0.6946564885496184
    F1-score: 0.5694931572794169

    HER2 Status Prediction
    ------------------------------
    [START] Pre-training step:
    >> Epoch 1 finished     RBM Reconstruction error 309.675462
    >> Epoch 2 finished     RBM Reconstruction error 302.142036
    >> Epoch 3 finished     RBM Reconstruction error 294.692107
    >> Epoch 4 finished     RBM Reconstruction error 290.237393
    >> Epoch 5 finished     RBM Reconstruction error 289.501104
    >> Epoch 1 finished     RBM Reconstruction error 1846.207275
    >> Epoch 2 finished     RBM Reconstruction error 1806.483032
    >> Epoch 3 finished     RBM Reconstruction error 1898.162720
    >> Epoch 4 finished     RBM Reconstruction error 1902.564453
    >> Epoch 5 finished     RBM Reconstruction error 1867.702637
    [END] Pre-training step
    
    [START] Fine tuning step:
    >> Epoch 0 finished     ANN training loss 1.010514
    >> Epoch 1 finished     ANN training loss 0.988286
    >> Epoch 2 finished     ANN training loss 0.995581
    >> Epoch 3 finished     ANN training loss 0.987776
    >> Epoch 4 finished     ANN training loss 0.986907
    [END] Fine tuning step
    
    Accuracy: 0.613043
    Accuracy: 0.612809
    Precision: 0.37582230623818524
    Recall: 0.6130434782608696

#### Special note ###
If you already have the processed datasets without running the `main_download.py`, please add `MAIN_MDBN_TCGA_BRCA = "main_datasets_folder"` on the first line of these two files:

* /mdbn_tcga_brca/Tensorflow/dataset_location.py
* /mdbn_tcga_brca/Theano/dataset_location.py

with `main_datasets_folder` being the main folder of your datasets.
