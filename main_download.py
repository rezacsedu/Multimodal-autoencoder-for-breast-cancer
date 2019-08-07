# THIS IS THE FIRST PART OF TWO MAIN PROGRAMS IN THIS PROJECT.
# THE OTHER ONE BEING THE MAIN_RUN.PY
# HERE YOU PICK THE TYPE OF DATASET YOU WANT TO USE AND CHOSE THE TARGET FOLDER

import sys
import os
import argparse

DATASET = 5					# dataset to download initialization
MAIN_LOCATION = "/home"		# target folder initialization

def main():
	global DATASET
	global MAIN_LOCATION
	print("Welcome to mDBN breast cancer status prediction!")
	print("All training data by TCGA BRCA\n")
	
	# 1. Set type of dataset to download
	parser = argparse.ArgumentParser()
	requiredArgs = parser.add_argument_group('required arguments')
	requiredArgs.add_argument("-d", "--dataset", type=int, help="Dataset of TCGA BRCA to be downloaded [1-5]", required=True)
	args = parser.parse_args()
	DATASET = int(args.dataset)

	# 2. Set target folder
	try:
		MAIN_LOCATION = raw_input("On which folder do you want to download these [default = /home]: ")
	except Exception as e:
		MAIN_LOCATION = "/home"
	if not MAIN_LOCATION:
		MAIN_LOCATION = "/home"

	# Run create_dataset on /Preprocessing/preprocessing_main.py
	# Send the type of dataset to download and the target folder
	program_path = os.path.dirname(os.path.realpath(__file__))
	sys.path.insert(0, program_path + '/Preprocessing')
	from preprocessing_main import create_dataset
	create_dataset(dataset=DATASET, location=MAIN_LOCATION)


if __name__ == '__main__':
    main()