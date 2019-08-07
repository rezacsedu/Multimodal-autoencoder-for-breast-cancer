from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from tf_models import SupervisedDBNRegression

import sys
import numpy

from dataset_location import *


def load_data(dataset, pca=2):
    """ The load dataset function
    
    This function covers for singular dataset
    (either DNA Methylation, Gene Expression, or miRNA Expression)
    for survival rate prediction.
    Input is .npy file location in string format.
    """

    # Input list of dataset files' name
    if dataset == 1:                                            # Methylation Platform GPL8490  (27578 cpg sites)
        temp_input = INPUT_MET_SURVIVAL
        temp_label = LABELS_MET_SURVIVAL
    elif dataset == 2:                                          # Methylation Platform GPL16304 (485577 cpg sites)
        temp_input = INPUT_METLONG_SURVIVAL
        temp_label = LABELS_METLONG_SURVIVAL
    elif (dataset == 3) or (dataset == 4) or (dataset == 5):    # Gene
        temp_label = LABELS_GEN_SURVIVAL
        if dataset == 3:                                        # Gene Count
            temp_input = INPUT_GEN_SURVIVAL_COUNT
        elif dataset == 4:                                      # Gene FPKM
            temp_input = INPUT_GEN_TYPE_SURVIVAL_FPKM
        elif dataset == 5:                                      # Gene FPKM-UQ
            temp_input = INPUT_GEN_TYPE_SURVIVAL_FPKMUQ
    elif dataset == 6:                                          # miRNA
        temp_input = INPUT_MIR_SURVIVAL
        temp_label = LABELS_MIR_SURVIVAL

    
    # Load the dataset as 'numpy.ndarray'
    try:
        input_set = numpy.load(temp_input)
        label_set = numpy.load(temp_label)
    except Exception as e:
        sys.exit("Change your choice of features because the data is not available")

    # feature selection by PCA
    if pca == 1:
        pca0 = PCA(n_components=600)
        input_set = pca0.fit_transform(input_set)

    # normalize input
    min_max_scaler = MinMaxScaler()
    input_set = min_max_scaler.fit_transform(input_set)

    return input_set, label_set


def test_DBN(finetune_lr=0.1,
    pretraining_epochs=100,
    pretrain_lr=0.01,
    training_epochs=100,
    dataset=6, batch_size=10,
    layers=[1000, 1000, 1000],
    dropout=0.2,
    pca=2,
    optimizer=1):
	
    # title
    temp_title = ["DNA Methylation Platform GPL8490",
                  "DNA Methylation Platform GPL16304",
                  "Gene Expression HTSeq Count",
                  "Gene Expression HTSeq FPKM",
                  "Gene Expression HTSeq FPKM-UQ",
                  "miRNA Expression"]
    print("\nSurvival Rate Regression with " + temp_title[dataset-1] + " (Tensorflow)\n")
    
    # Loading dataset
    X, Y = load_data(dataset, pca)

    # Splitting data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)

    # Training
    regressor = SupervisedDBNRegression(hidden_layers_structure=layers,
                                    	learning_rate_rbm=pretrain_lr,
                                    	learning_rate=finetune_lr,
                                    	n_epochs_rbm=pretraining_epochs,
                                    	n_iter_backprop=training_epochs,
                                    	batch_size=batch_size,
                                    	activation_function='relu',
                                        dropout_p=dropout)
    regressor.fit(X_train, Y_train)

    # Test
    Y_pred = regressor.predict(X_test)
    Y_pred = numpy.transpose(Y_pred)[0]
    Y_pred_train = regressor.predict(X_train)
    Y_pred_train = numpy.transpose(Y_pred_train)[0]
    print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))
    print('Done. \ntraining R-squared: %f\nMSE: %f' % (r2_score(Y_train, Y_pred_train), mean_squared_error(Y_train, Y_pred_train)))
