import os
import sys
import subprocess
from shutil import copyfile

def create_dataset(dataset=3, location="/home"):
	global DATASET
	DATASET = dataset
	
	#########################
	### SET MAIN LOCATION ###
	#########################
	program_path = os.path.dirname(os.path.realpath(__file__))

	with open(program_path + "/folder_location.py") as f:
		lines = f.readlines()

	with open(program_path + "/folder_location.py", "w") as f:
		lines.insert(0, "MAIN_MDBN_TCGA_BRCA = \"" + location + "/\"\n\n")
		f.write("".join(lines))

	if not os.path.isdir(location):
		os.makedirs(location)
	
	from folder_location import *


	#########################
	#### DOWNLOAD DATASET ###
	#########################
	# 1. Clinical
	if not os.path.isdir(DATASET_CLINICAL):
		os.makedirs(DATASET_CLINICAL)
	os.chdir(DATASET_CLINICAL)

	bashCommand = program_path + "/gdc-client download -m " + program_path + "/gdc_manifest_cli_20180225.txt"
	print("\nDownloading clinical file ...\n")
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	print(output)

	# 2. DNA methylation
	if (dataset==1) or (dataset==5):
		if not os.path.isdir(DATASET_METHYLATION):
			os.makedirs(DATASET_METHYLATION)
		os.chdir(DATASET_METHYLATION)

		bashCommand = program_path + "/gdc-client download -m " + program_path + "/gdc_manifest_met_20180225.txt"
		print("\nDownloading DNA methylation file ...\n")
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()
		print(output)

	# 3. Gene Expression
	if (dataset==2) or (dataset==4) or (dataset==5):
		if not os.path.isdir(DATASET_GENE):
			os.makedirs(DATASET_GENE)
		os.chdir(DATASET_GENE)

		bashCommand = program_path + "/gdc-client download -m " + program_path + "/gdc_manifest_gen_20180225.txt"
		print("\nDownloading gene expression file ...\n")
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()
		print(output)

	# 4. miRNA Expression
	if (dataset==3) or (dataset==4) or (dataset==5):
		if not os.path.isdir(DATASET_MIRNA):
			os.makedirs(DATASET_MIRNA)
		os.chdir(DATASET_MIRNA)

		bashCommand = program_path + "/gdc-client download -m " + program_path + "/gdc_manifest_mir_20180225.txt"
		print("\nDownloading miRNA expression file ...\n")
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()
		print(output)

	# back to program path
	os.chdir(program_path)
	

	############################
	##### CREATE MAIN META #####
	############################
	from preprocess_meta import *
	requests_meta()
	meta_per_case()
	file_amount()
	submitter_id_to_case_uuid()


	############################
	### CREATE CLINICAL META ###
	############################
	from preprocess_clinical import pathology_receptor
	if not os.path.isdir(TARGET_CLINICAL):
		os.makedirs(TARGET_CLINICAL)

	pathology_receptor()
	copyfile(program_path + "/survival_plot.tsv", TARGET_CLINICAL + "survival_plot.tsv")


	############################
	#### CREATE GENERAL META ###
	############################
	from preprocess_others import *
	# (DNA Methylation) or (DNA Methylation + Gene Expression + miRNA Expression)
	if (dataset==1) or (dataset==5):
		meta_methylation_list_files()
		meta_methylation_cpg()
		meta_long_methylation()
		meta_methylation_sample_type()
		meta_methylation_used_case()
		meta_methylation_cpg_index()


	############################
	###### CREATE DATASET ######
	############################
	from preprocess_packaging import *

	# 1. labels
	label_cancer_type(dataset=DATASET)
	label_survival(dataset=DATASET)

	# 2. DNA Methylation
	if (dataset==1) or (dataset==5):
		input_met_cancer_type()
		input_metlong_cancer_type()
		input_met_survival()
		input_metlong_survival()

	# 3. Gene Expression
	if (dataset==2) or (dataset==4) or (dataset==5):
		input_gen_cancer_type()
		input_gen_survival()

	# 4. miRNA Expression
	if (dataset==3) or (dataset==4) or (dataset==5):
		input_mir_cancer_type()
		input_mir_survival()

	# 5. Gene Expression + miRNA Expression
	if (dataset==4) or (dataset==5):
		input_gen_genmir_cancer_type()
		input_mir_genmir_cancer_type()
		input_gen_genmir_survival()
		input_mir_genmir_survival()

	# 6. DNA Methylation + Gene Expression + miRNA Expression
	if (dataset==5):
		input_met_metgenmir_cancer_type()
		input_gen_metgenmir_cancer_type()
		input_mir_metgenmir_cancer_type()
		input_metlong_metlonggenmir_cancer_type()
		input_gen_metlonggenmir_cancer_type()
		input_mir_metlonggenmir_cancer_type()
		input_met_metgenmir_survival()
		input_gen_metgenmir_survival()
		input_mir_metgenmir_survival()
		input_metlong_metlonggenmir_survival()
		input_gen_metlonggenmir_survival()
		input_mir_metlonggenmir_survival()



	############################
	###### CREATE DATASET ######
	############################
	# 1. Tensorflow folder
	with open(program_path + "/../Tensorflow/dataset_location.py") as f:
		lines = f.readlines()

	with open(program_path + "/../Tensorflow/dataset_location.py", "w") as f:
		lines.insert(0, "MAIN_MDBN_TCGA_BRCA = \"" + location + "/\"\n\n")
		f.write("".join(lines))

	# 2. Theano folder
	with open(program_path + "/../Theano/dataset_location.py") as f:
		lines = f.readlines()

	with open(program_path + "/../Theano/dataset_location.py", "w") as f:
		lines.insert(0, "MAIN_MDBN_TCGA_BRCA = \"" + location + "/\"\n\n")
		f.write("".join(lines))