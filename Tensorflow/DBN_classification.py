from __future__ import print_function, division
import numpy as np
import pandas as pd
import sys
import timeit
from itertools import cycle

np.random.seed(123456789)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import itertools

from tf_models import SupervisedDBNClassification
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dataset_location import *

def print_and_plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return cm


def load_data(dataset, pca=2):
    """ The load dataset function
    
    This function covers for singular dataset
    (either DNA Methylation, Gene Expression, or miRNA Expression)
    for ER, PGR, and HER2 status prediction.
    Input is .npy file location in string format.
    Output is in numpy.ndarray format.
    """

    # Initialize list of dataset files' name
    temp_input = []
    temp_label = []

    # Input list of dataset files' name
    if dataset == 1:        # Methylation Platform GPL8490  (27578 cpg sites)
        temp_input.extend((INPUT_MET_TYPE_ER,INPUT_MET_TYPE_PGR,INPUT_MET_TYPE_HER2))
        temp_label.extend((LABELS_MET_TYPE_ER,LABELS_MET_TYPE_PGR,LABELS_MET_TYPE_HER2))
    elif dataset == 2:      # Methylation Platform GPL16304 (485577 cpg sites)
        temp_input.extend((INPUT_METLONG_TYPE_ER,INPUT_METLONG_TYPE_PGR,INPUT_METLONG_TYPE_HER2))
        temp_label.extend((LABELS_METLONG_TYPE_ER,LABELS_METLONG_TYPE_PGR,LABELS_METLONG_TYPE_HER2))
    elif (dataset == 3) or (dataset == 4) or (dataset == 5):    # Gene
        temp_label.extend((LABELS_GEN_TYPE_ER,LABELS_GEN_TYPE_PGR,LABELS_GEN_TYPE_HER2))
        if dataset == 3:    # Gene Count
            temp_input.extend((INPUT_GEN_TYPE_ER_COUNT,INPUT_GEN_TYPE_PGR_COUNT,INPUT_GEN_TYPE_HER2_COUNT))
        elif dataset == 4:  # Gene FPKM
            temp_input.extend((INPUT_GEN_TYPE_ER_FPKM,INPUT_GEN_TYPE_PGR_FPKM,INPUT_GEN_TYPE_HER2_FPKM))
        elif dataset == 5:  # Gene FPKM-UQ
            temp_input.extend((INPUT_GEN_TYPE_ER_FPKMUQ,INPUT_GEN_TYPE_PGR_FPKMUQ,INPUT_GEN_TYPE_HER2_FPKMUQ))
    elif dataset == 6:      # miRNA
        temp_input.extend((INPUT_MIR_TYPE_ER,INPUT_MIR_TYPE_PGR,INPUT_MIR_TYPE_HER2))
        temp_label.extend((LABELS_MIR_TYPE_ER,LABELS_MIR_TYPE_PGR,LABELS_MIR_TYPE_HER2))

    
    min_max_scaler = MinMaxScaler()     # Initialize normalization function
    rval = []                           # Initialize list of outputs

    # Iterate 3 times, each for ER, PGR, and HER2
    for i in range(3):
        # Load the dataset as 'numpy.ndarray'
        try:
            input_set = np.load(temp_input[i])
            label_set = np.load(temp_label[i])
        except Exception as e:
            sys.exit("Change your choice of features because the data is not available")

        # feature selection by PCA
        if pca == 1:
            pca0 = PCA(n_components=600)
            input_set = pca0.fit_transform(input_set)
        
        # normalize input
        input_set = min_max_scaler.fit_transform(input_set)

        rval.extend((input_set, label_set))

    return rval


def test_DBN(finetune_lr=0.1,
    pretraining_epochs=100,
    pretrain_lr=0.01,
    training_epochs=100,
    dataset=6, batch_size=10,
    layers=[1000, 1000, 1000],
    dropout=0.4, pca=2, optimizer=1):
    
    # Title
    temp_title = ["DNA Methylation Platform GPL8490",
                  "DNA Methylation Platform GPL16304",
                  "Gene Expression HTSeq Count",
                  "Gene Expression HTSeq FPKM",
                  "Gene Expression HTSeq FPKM-UQ",
                  "miRNA Expression"]
    print("\nCancer Type Classification with " + temp_title[dataset-1] + " (Tensorflow)\n")
    
    # Load datasets
    datasets = load_data(dataset, pca)
    
    temp_str = ["ER", "PGR", "HER2"]

    # Iterate for ER, PGR, and HER2
    for protein in range(3):
        # start timer
        start = timeit.default_timer()
        
        # Title
        print("\n" + temp_str[protein] + " Status Prediction\n")

        # Split dataset into training set and test set
        X_train, X_test, Y_train, Y_test = train_test_split(datasets[protein*2], datasets[(protein*2)+1], test_size=0.25, random_state=100)

        # Training
        classifier = SupervisedDBNClassification(hidden_layers_structure=layers,
                                                 learning_rate_rbm=pretrain_lr,
                                                 learning_rate=finetune_lr,
                                                 n_epochs_rbm=pretraining_epochs,
                                                 n_iter_backprop=training_epochs,
                                                 batch_size=batch_size,
                                                 activation_function='relu',
                                                 dropout_p=0.2,
                                                 l2_regularization=1.)
        classifier.fit(X_train, Y_train)

        # Compute the prediction accuracy 
        Y_pred = classifier.predict(X_test)
        Y_pred_train = classifier.predict(X_train)
        print('Accuracy: %f' % accuracy_score(Y_test, Y_pred))
        print('tAccuracy: %f' % accuracy_score(Y_train, Y_pred_train))

        # Compute the precision, recall and f1 score of the classification
        p, r, f, s = precision_recall_fscore_support(Y_test, Y_pred, average='weighted')
        print('Precision:', p)
        print('Recall:', r)
        print('F1-score:', f)
        p, r, f, s = precision_recall_fscore_support(Y_train, Y_pred_train, average='weighted')
        print('tPrecision:', p)
        print('tRecall:', r)
        print('tF1-score:', f)

        
        # Plot non-normalized confusion matrix
        cnf_matrix = confusion_matrix(Y_test, Y_pred)
        if protein == 2:
            class_names = ['Positive','Negative','Indeterminate','Equivocal']
        else:
            class_names = ['Positive','Negative','Indeterminate']
        
        plt.figure()
        print_and_plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='Confusion matrix, without normalization')
        plt.show()
        
        
        # stop timer and show result
        stop = timeit.default_timer()
        print(temp_str[protein] + " Status Prediction is done in " + str(stop-start) + "s")

