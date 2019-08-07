# THIS IS THE SECOND PART OF TWO MAIN PROGRAMS IN THIS PROJECT.
# THE OTHER ONE BEING THE MAIN_DOWNLOAD.PY
# HERE YOU RUN THE TRAINING PROGRAM OF THE MULTIMODAL DEEP BELIEF NETWORK (MDBN)
# YOU WILL DESIGN YOUR OWN NETWORK

import sys
import os
import timeit
import argparse

DATASET = 6
PRETRAIN_EPOCH = 100
TRAIN_EPOCH = 100
BATCH_SIZE = 10
PRETRAIN_LR = 0.01
TRAIN_LR = 0.1
LAYERS = []
LAYERS_MET = []
LAYERS_GEN = []
LAYERS_MIR = []
LAYERS_TOT = []
DROPOUT = 0.2
PCA = 2
OPTIMIZER = 1

def main():
	global DATASET
	global PRETRAIN_EPOCH
	global TRAIN_EPOCH
	global BATCH_SIZE
	global PRETRAIN_LR
	global TRAIN_LR
	global LAYERS
	global DROPOUT
	global PCA
	global OPTIMIZER

	print("Welcome to mDBN breast cancer status prediction!")
	print("All training data by TCGA BRCA\n")

	parser = argparse.ArgumentParser()
	requiredArgs = parser.add_argument_group('required arguments')
	requiredArgs.add_argument("-p", "--platform", type=int, help="Platform to be used [1-2]", required=True)
	requiredArgs.add_argument("-t", "--type", type=int, help="Types of prediction [1-2]", required=True)
	requiredArgs.add_argument("-d", "--dataset", type=int, help="Dataset of TCGA BRCA to be used [1-15]", required=True)
	parser.add_argument("--pretrain_epoch", type=int, help="Pretrain for specified epochs")
	parser.add_argument("--train_epoch", type=int, help="Train for specified epochs")
	parser.add_argument("--batch_size", type=int, help="Pretraining and training batch size")
	parser.add_argument("--pretrain_lr", type=int, help="Pretraining learning rate")
	parser.add_argument("--train_lr", type=int, help="Training learning rate")
	parser.add_argument("--dropout", type=int, help="Dropout rate")
	parser.add_argument("--pca", type=int, help="PCA usage [1-2]")
	parser.add_argument("--optimizer", type=int, help="Type of optimizer to be used [1-3]")
	args = parser.parse_args()
	platform = int(args.platform)
	prediction = int(args.type)
	DATASET = int(args.dataset)

	if args.pretrain_epoch:
		PRETRAIN_EPOCH = int(args.pretrain_epoch)
	if args.train_epoch:
		TRAIN_EPOCH = int(args.train_epoch)
	if args.batch_size:
		BATCH_SIZE = int(args.batch_size)
	if args.pretrain_lr:
		PRETRAIN_LR = int(args.pretrain_lr)
	if args.train_lr:
		TRAIN_LR = int(args.train_lr)
	if args.dropout:
		DROPOUT = int(args.dropout)
	if args.pca:
		PCA = int(args.pca)
	if args.optimizer:
		OPTIMIZER = int(args.optimizer)


	######################
	####### LAYERS #######
	######################
	print("Neural Network Layers")
	if (DATASET >= 1) and (DATASET <= 6):
		try:
			n_layers = int(input("Number of hidden layers [default = 3]: "))
		except Exception as e:
			n_layers = 3
		
		for i in range(n_layers):
			try:
				temp = int(input("Layer " + str(i) + " size [default = 1000]: "))
			except Exception as e:
				temp = 1000
			
			LAYERS.append(temp)

	elif (DATASET >= 7) and (DATASET <= 15):
		if (DATASET >= 10) and (DATASET <= 15):
			try:
				n_layers = int(input("Number of hidden layers for DNA-Methylation's DBN [default = 3]: "))
			except Exception as e:
				n_layers = 3

			for i in range(n_layers):
				try:
					temp = int(input("Layer " + str(i) + " size [default = 1000]: "))
				except Exception as e:
					temp = 1000

				LAYERS_MET.append(temp)

		try:
			n_layers = int(input("Number of hidden layers for Gene-Expression's DBN [default = 3]: "))
		except Exception as e:
			n_layers = 3
		
		for i in range(n_layers):
			try:
				temp = int(input("Layer " + str(i) + " size [default = 1000]: "))
			except Exception as e:
				temp = 1000
			
			LAYERS_GEN.append(temp)

		try:
			n_layers = int(input("Number of hidden layers for miRNA-Expression's DBN [default = 3]: "))
		except Exception as e:
			n_layers = 3
		
		for i in range(n_layers):
			try:
				temp = int(input("Layer " + str(i) + " size [default = 1000]: "))
			except Exception as e:
				temp = 1000
			
			LAYERS_MIR.append(temp)

		try:
			n_layers = int(input("Number of hidden layers for combined DBN [default = 3]: "))
		except Exception as e:
			n_layers = 3
		
		for i in range(n_layers):
			try:
				temp = int(input("Layer " + str(i) + " size [default = 500]: "))
			except Exception as e:
				temp = 500
			
			LAYERS_TOT.append(temp)



	######################
	#### OPEN PROGRAM ####
	######################
	start = timeit.default_timer()
	program_path = os.path.dirname(os.path.realpath(__file__))
	if platform == 1:										# 1. Tensorflow
		sys.path.insert(0, program_path + '/Tensorflow')
		if prediction == 1: 								# 1.1. Tensorflow Classification
			if (DATASET >= 1) and (DATASET <= 6):			# 1.1.1. Tensorflow Classification DBN
				from DBN_classification import test_DBN
				test_DBN(dataset = DATASET,
						 pretraining_epochs = PRETRAIN_EPOCH,
						 training_epochs = TRAIN_EPOCH,
						 pretrain_lr = PRETRAIN_LR,
						 finetune_lr = TRAIN_LR,
						 batch_size = BATCH_SIZE,
						 layers=LAYERS,
						 dropout=DROPOUT,
						 pca=PCA,
						 optimizer=OPTIMIZER)

			elif (DATASET >= 7) and (DATASET <= 15):		# 1.1.2 Tensorflow Classification mDBN
				from mDBN_classification import test_mDBN
				test_mDBN(dataset = DATASET,
						  pretraining_epochs = PRETRAIN_EPOCH,
						  training_epochs = TRAIN_EPOCH,
						  pretrain_lr = PRETRAIN_LR,
						  finetune_lr = TRAIN_LR,
						  batch_size = BATCH_SIZE,
						  layers_met=LAYERS_MET,
						  layers_gen=LAYERS_GEN,
						  layers_mir=LAYERS_MIR,
						  layers_tot=LAYERS_TOT,
						  dropout=DROPOUT,
						  pca=PCA,
						  optimizer=OPTIMIZER)

		elif prediction == 2:								# 1.2. Tensorflow Regression
			if (DATASET >= 1) and (DATASET <= 6):			# 1.2.1. Tensorflow Regression DBN
				from DBN_regression import test_DBN
				test_DBN(dataset = DATASET,
						 pretraining_epochs = PRETRAIN_EPOCH,
						 training_epochs = TRAIN_EPOCH,
						 pretrain_lr = PRETRAIN_LR,
						 finetune_lr = TRAIN_LR,
						 batch_size = BATCH_SIZE,
						 layers=LAYERS,
						 dropout=DROPOUT,
						 pca=PCA,
						 optimizer=OPTIMIZER)

			elif (DATASET >= 7) and (DATASET <= 15):		# 1.2.2. Tensorflow Regression mDBN
				from mDBN_regression import test_mDBN
				test_mDBN(dataset = DATASET,
						  pretraining_epochs = PRETRAIN_EPOCH,
						  training_epochs = TRAIN_EPOCH,
						  pretrain_lr = PRETRAIN_LR,
						  finetune_lr = TRAIN_LR,
						  batch_size = BATCH_SIZE,
						  layers_met=LAYERS_MET,
						  layers_gen=LAYERS_GEN,
						  layers_mir=LAYERS_MIR,
						  layers_tot=LAYERS_TOT,
						  dropout=DROPOUT,
						  pca=PCA,
						  optimizer=OPTIMIZER)

	elif platform == 2:										# 2. Theano
		sys.path.insert(0, program_path + '/Theano')
		if prediction == 1:									# 2.1. Theano Classification
			if (DATASET >= 1) and (DATASET <= 6):			# 2.1.1. Theano Classification DBN
				from DBN_classification import test_DBN
				test_DBN(dataset = DATASET,
						 pretraining_epochs = PRETRAIN_EPOCH,
						 training_epochs = TRAIN_EPOCH,
						 pretrain_lr = PRETRAIN_LR,
						 finetune_lr = TRAIN_LR,
						 batch_size = BATCH_SIZE,
						 layers=LAYERS,
						 dropout=DROPOUT,
						 pca=PCA,
						 optimizer=OPTIMIZER)

			elif (DATASET >= 7) and (DATASET <= 15):		# 2.1.2. Theano Classification mDBN
				from mDBN_classification import test_mDBN
				test_mDBN(dataset = DATASET,
						  pretraining_epochs = PRETRAIN_EPOCH,
						  training_epochs = TRAIN_EPOCH,
						  pretrain_lr = PRETRAIN_LR,
						  finetune_lr = TRAIN_LR,
						  batch_size = BATCH_SIZE,
						  layers_met=LAYERS_MET,
						  layers_gen=LAYERS_GEN,
						  layers_mir=LAYERS_MIR,
						  layers_tot=LAYERS_TOT,
						  dropout=DROPOUT,
						  pca=PCA,
						  optimizer=OPTIMIZER)

		elif prediction == 2:								# 2.2. Theano Regression
			if (DATASET >= 1) and (DATASET <= 6):			# 2.2.1. Theano Regression DBN
				from DBN_regression import test_DBN
				test_DBN(dataset = DATASET,
						 pretraining_epochs = PRETRAIN_EPOCH,
						 training_epochs = TRAIN_EPOCH,
						 pretrain_lr = PRETRAIN_LR,
						 finetune_lr = TRAIN_LR,
						 batch_size = BATCH_SIZE,
						 layers=LAYERS,
						 dropout=DROPOUT,
						 pca=PCA,
						 optimizer=OPTIMIZER)

			elif (DATASET >= 7) and (DATASET <= 15):		# 2.2.2. Theano Regression mDBN
				from mDBN_regression import test_mDBN
				test_mDBN(dataset = DATASET,
						  pretraining_epochs = PRETRAIN_EPOCH,
						  training_epochs = TRAIN_EPOCH,
						  pretrain_lr = PRETRAIN_LR,
						  finetune_lr = TRAIN_LR,
						  batch_size = BATCH_SIZE,
						  layers_met=LAYERS_MET,
						  layers_gen=LAYERS_GEN,
						  layers_mir=LAYERS_MIR,
						  layers_tot=LAYERS_TOT,
						  dropout=DROPOUT,
						  pca=PCA,
						  optimizer=OPTIMIZER)

	stop = timeit.default_timer()
	print("\nOverall the program run for: " + str(stop-start) + "s")


if __name__ == '__main__':
    main()