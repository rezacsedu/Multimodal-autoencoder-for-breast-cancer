from folder_location import *

import os
import csv
import json
import yaml
import numpy as np
import timeit
import gzip
import numpy as np
import requests



# Find the row size (nr of cpg sites) for each methylation files
def meta_methylation_file_size():
	# 1234 methylation files
	# from 1095 cases (963 have 1 files, 125 have 2 files,  7 have 3 files)

	# load meta of methylation
	data_met = np.genfromtxt(TARGET_META_CSV + "methylation_beta_value.csv", dtype=str, delimiter=',', skip_header=0)
	
	# find where the case id column is located in your meta_clinicals.csv
	file_id_column, = np.where(data_met[0]=='file_id')[0]
	file_name_column, = np.where(data_met[0]=='file_name')[0]
	
	data_met = data_met[1:]

	temp_list = []
	
	# iterate for methylation file
	for i in range(len(data_met)):
		file_id = data_met[i,file_id_column]
		file_name = data_met[i,file_name_column]

		with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
			file = f.read().splitlines()

		# file without header
		rows = file[1:]

		temp_list.append(len(rows))
		print(len(rows))

	# remove duplicate rows
	print(list(set(temp_list)))

	# 342 files have 27578 rows
	# 892 files have 485577 rows
	# there are two different platforms that are used to compute the DNA methylation on TCGA BRCA patients:
	# 	1. Platform GPL8490 that uses 27578 cpg sites (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL8490)
	# 	2. Platform GPL16304 that uses 485577 cpg sites (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL16304)



# Create list of file id who uses NCBI Platform GPL8490 (short CPG sites) and NCBI Platform GPL16304 (long CPG sites)
def meta_methylation_list_files():
	# load meta of methylation
	data_met = np.genfromtxt(TARGET_META_CSV + "methylation_beta_value.csv", dtype=str, delimiter=',', skip_header=0)
	
	# find where the case id column is located in your meta_clinicals.csv
	file_id_column, = np.where(data_met[0]=='file_id')[0]
	file_name_column, = np.where(data_met[0]=='file_name')[0]
	
	data_met = data_met[1:]

	files_short = []	# list of file id with 27578 rows
	files_long = []		# list of file id with 485577 rows

	# iterate for methylation file
	for i in range(len(data_met)):
		file_id = data_met[i,file_id_column]
		file_name = data_met[i,file_name_column]

		with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
			temp = f.read().splitlines()

		# file without header
		rows = temp[1:]

		if len(rows) == 27578:
			files_short.append(file_id)
		elif len(rows) == 485577:
			files_long.append(file_id)

	# save as json
	if not(os.path.isdir(TARGET_METHYLATION)):
		os.makedirs(TARGET_METHYLATION)

	files_short_j = json.dumps(files_short, indent=2)
	with open(TARGET_METHYLATION + "files_short.json", "w") as text_file:
		text_file.write(files_short_j)
	print("files_short.json is created.")

	files_long_j = json.dumps(files_long, indent=2)
	with open(TARGET_METHYLATION + "files_long.json", "w") as text_file:
		text_file.write(files_long_j)
	print("files_long.json is created.")



# Create files with list of cpg sites in Platform GPL8490, Platform GPL16304, and the overlapping sites between those 2 platforms
def meta_methylation_cpg():
	# from all 1234 methylation files, TCGA BRCA uses two different platforms which correspond to different size of :
	# 	342 files have 27578 rows (27578 cpg sites)
	# 	892 files have 485577 rows (485577 cpg sites)
	# we will find the overlapping rows between the 27578-rows-file and the 485577-rows-file
	# the overlapping rows will be used

	cpg_long = []	# list of cpg from the 485577-rows-file
	cpg_short = []	# list of cpg from the 27578-rows-file

	#############################
	######## 1. CPG_LONG ########
	#############################
	# load meta of methylation and files_long
	data_met = np.genfromtxt(TARGET_META_CSV + "methylation_beta_value.csv", dtype=str, delimiter=',', skip_header=0)
	
	# find where the case id column is located in your meta_clinicals.csv
	file_id_column, = np.where(data_met[0]=='file_id')[0]
	file_name_column, = np.where(data_met[0]=='file_name')[0]
	
	data_met = data_met[1:]

	with open(TARGET_METHYLATION + "files_long.json") as file:
		list_file = yaml.safe_load(file)

	# randomly taking 1 file as sample file and load it
	file_id = list_file[np.random.randint(len(list_file))]
	file_name = data_met[np.where(data_met[:,file_id_column] == file_id),file_name_column][0,0]

	with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
		rows = f.read().splitlines()

	# iterate for every rows without the header
	comment_line = 1
	for row in range(comment_line, len(rows)):
		line = rows[row].split("\t")
		cpg_long.append(line[0])

	cpg_long.sort()
	
	# save as json
	cpg_long_j = json.dumps(cpg_long, indent=2)
	with open(TARGET_METHYLATION + "cpg_long.json", "w") as text_file:
		text_file.write(cpg_long_j)

	print("cpg_long.json is created.")
	    
	
	#############################
	######## 2. CPG_SHORT #######
	#############################
	# load meta of methylation and files_long
	with open(TARGET_METHYLATION + "files_short.json") as file:
		list_file = yaml.safe_load(file)

	# randomly taking 1 file as sample file and load it
	file_id = list_file[np.random.randint(len(list_file))]
	file_name = data_met[np.where(data_met[:,file_id_column] == file_id),file_name_column][0,0]

	with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
		rows = f.read().splitlines()

	# iterate for every rows without the header
	comment_line = 1
	for row in range(comment_line, len(rows)):
		line = rows[row].split("\t")
		cpg_short.append(line[0])

	cpg_short.sort()

	# save as json
	cpg_short_j = json.dumps(cpg_short, indent=2)
	with open(TARGET_METHYLATION + "cpg_short.json", "w") as text_file:
		text_file.write(cpg_short_j)

	print("cpg_short.json is created.")

	    
	#############################
	####### 3. CPG OVERLAP ######
	#############################
	# find intersection between cpg_long and cpg_short
	np_cpg_overlap = np.intersect1d(np.asarray(cpg_long),np.asarray(cpg_short))
	cpg_overlap = np_cpg_overlap.tolist()
	cpg_overlap.sort()

	# save as json
	cpg_overlap_j = json.dumps(cpg_overlap, indent=2)
	with open(TARGET_METHYLATION + "cpg.json", "w") as text_file:
		text_file.write(cpg_overlap_j)

	print("cpg.json is created.")

	# 25978 CPG sites out of the 27578 sites of Platform GPL8490 and 485577 of Platform GPL16304 are overlapping
	# which count for ~94% of all CPG sites  of Platform GPL8490

	# Based on this observation, we will create 2 different type of dataset for the Neural Network:
	#	1. methylation dataset which uses all files from Platform GPL8490 and Platform GPL8490
	#	   we will only use the overlapping 25978 CPG sites
	#	2. long methylation dataset which only use Platform GPL16304
	#	   we will use the whole 485577 CPG sites from Platform GPL16304
	#	   for this second dataset, we will first create the necessary meta files



# Create meta files for the long methylation dataset (from GPL8490)
def meta_long_methylation():
	################################
	### 1. META LONG METHYLATION ###
	################################
	# load meta of methylation
	data_met = np.genfromtxt(TARGET_META_CSV + "methylation_beta_value.csv", dtype=str, delimiter=',', skip_header=0)
	headers = data_met[0]
	
	# find where the case id column is located in your meta_clinicals.csv
	file_id_column, = np.where(data_met[0]=='file_id')[0]
	
	data_met = data_met[1:]

	# load files_long as list of files that want to be kept
	with open(TARGET_METHYLATION + "files_long.json") as file:
		kept_file_id_j = yaml.safe_load(file)
	
	kept_file_id = np.asarray(kept_file_id_j)

	# methylation_long_beta_value is the original methylation_beta_value with only files_long
	data_met_long = data_met[np.logical_or.reduce([data_met[:,file_id_column] == i for i in kept_file_id])]
	with open(TARGET_META_CSV + 'methylation_long_beta_value.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(headers)
		for row in data_met_long:
			writer.writerow(row)
	print("methylation_long_beta_value.csv is created.")


	#######################################
	### 2. FILE AMOUNT LONG METHYLATION ###
	#######################################
	# load file_amount and get list of cases
	file_amount = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
	cases = file_amount[:,0]
	
	# get list of case (with duplicates) from methylation_long_beta_value
	# find where the case id column is located in your meta_clinicals.csv
	case_id_column, = np.where(headers=='cases.0.case_id')[0]
	cases_met_long_all = data_met_long[:,case_id_column]
	
	# transform it as dictionary of cases_id to number of files
	unique, counts = np.unique(cases_met_long_all, return_counts=True)
	dict_cases_met_long_all = dict(zip(unique, counts))

	# save file_amount_met_long as csv
	with open(TARGET_META_CSV + 'file_amount_met_long.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(["Cases", "Long Methylation Beta Value"])
		for case in cases:
			if case in dict_cases_met_long_all:
				writer.writerow([case,dict_cases_met_long_all[case]])
			else:
				writer.writerow([case,0])
	print("file_amount_met_long.csv is created.")

	# save unique list of case from files_long for further use as json
	cases_met_long_all = np.unique(cases_met_long_all)
	cases_met_long_all_j = json.dumps(cases_met_long_all.tolist(), indent=2)
	with open(TARGET_METHYLATION + "cases_met_long_all.json", "w") as text_file:
		text_file.write(cases_met_long_all_j)
	print("cases_met_long_all.json is created.")



# Check cases that don't have tumor sample file
def meta_methylation_sample_type():
	#####################################
	### 1. NORMAL METHYLATION DATASET ###
	#####################################
	# out of 1234 files, 1104 files from Primary Tumor, 123 files from Solid Tissue Normal, 7 files from Metastatic
	# only include cases that at least have 1 tumor sample, because we do classification/regression only between tumor patients 

	# load meta of methylation and files_long
	file_amount = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
	data_met = np.genfromtxt(TARGET_META_CSV + "methylation_beta_value.csv", dtype=str, delimiter=',', skip_header=0)
	
	# find where the case id column is located in your meta_clinicals.csv
	sample_type_column, = np.where(data_met[0]=='cases.0.samples.0.sample_type')[0]
	case_id_column, = np.where(data_met[0]=='cases.0.case_id')[0]
	
	data_met = data_met[1:]

	# cek cases with 1 files
	'''temp_list = []

	for i in range(len(file_amount)):
		if file_amount[i,8] == "1":
			temp_list.append([file_amount[i,0],data_met[np.where(data_met[:,case_id_column] == file_amount[i,0]),sample_type_column][0,0]])

	print(temp_list)'''
	# all 963 files are from Primary Tumor. So, all cases will be included


	# cek cases with 2 files
	'''temp_list = []

	for i in range(len(file_amount)):
		if file_amount[i,8] == "2":
			temp_list.append(data_met[np.where(data_met[:,case_id_column] == file_amount[i,0]),sample_type_column][0])

	print(temp_list)'''
	# out of 125 cases =
	# 	118 cases are combination of Primary Tumor + Solid Tissue Normal
	# 	4 cases consist of 2 Primary Tumor files
	# 	3 cases are combination of Primary Tumor + Metastatic
	# all cases can be used
	# out of the 4 cases that consists of > 1 Primary Tumor Files, we will only used 1 of it


	# cek cases with 3 files
	'''temp_list = []

	for i in range(len(file_amount)):
		if file_amount[i,8] == "3":
			temp_list.append([file_amount[i,0],data_met[np.where(data_met[:,case_id_column] == file_amount[i,0]),sample_type_column][0]])

	print(temp_list)'''
	# out of 7 cases =
	#	2 cases consist of 3 Primary Tumor files
	#	4 cases are combination of Primary Tumor + Solid Tissue Normal + Metastatic
	#	1 cases are combination of 2 Primary Tumor files + 1 Solid Tissue Normal file
	# all cases can be used
	# out of the 3 cases that consists of > 1 Primary Tumor Files, we will only used 1 of it

	# so, all 1095 cases can be used


	#####################################
	#### 2. LONG METHYLATION DATASET ###
	#####################################
	# 892 files
	# from 789 cases (692 have 1 files, 91 have 2 files,  6 have 3 files)
	# out of 892 files, 791 files from Primary Tumor, 96 files from Solid Tissue Normal, 5 files from Metastatic
	# only include cases that at least have 1 tumor sample, because we do classification/regression only between tumor patients 

	# load meta of methylation and files_long
	file_amount_met_long = np.genfromtxt(TARGET_META_CSV + "file_amount_met_long.csv", dtype=str, delimiter=',', skip_header=1)
	data_met_long = np.genfromtxt(TARGET_META_CSV + "methylation_long_beta_value.csv", dtype=str, delimiter=',', skip_header=0)
	
	# cek cases with 1 files first
	cases_remove_met_long = np.empty((0,1))

	for i in range(len(file_amount_met_long)):
		if file_amount_met_long[i,1] == "1":
			sample_type = data_met_long[np.where(data_met_long[:,case_id_column] == file_amount_met_long[i,0]),sample_type_column][0,0]
			if sample_type != "Primary Tumor":
				case_removed = data_met_long[np.where(data_met_long[:,case_id_column] == file_amount_met_long[i,0]),case_id_column][0,0]
				cases_remove_met_long = np.append(cases_remove_met_long,case_removed)

	# save list of case that don't have tumor sample as json
	cases_remove_met_long_j = json.dumps(cases_remove_met_long.tolist(), indent=2)
	with open(TARGET_METHYLATION + "cases_remove_met_long.json", "w") as text_file:
		text_file.write(cases_remove_met_long_j)
	print("cases_remove_met_long.json is created.")

	# out of 692 files =
	#	685 are Primary Tumor
	#	6 are Solid Tissue Normal
	#	1 is Metastatic
	# so, only 685 can be used


	# check cases with 2 files first
	'''for i in range(len(file_amount_met_long)):
		if file_amount_met_long[i,1] == "2":
			sample_type = data_met_long[np.where(data_met_long[:,case_id_column] == file_amount_met_long[i,0]),sample_type_column][0]
			print(sample_type)'''

	# out of 91 cases =
	# 	86 cases are combination of Primary Tumor + Solid Tissue Normal
	# 	4 cases consist of 2 Primary Tumor files
	# 	1 cases are combination of Primary Tumor + Metastatic
	# all 91 cases can be used
	# out of the 4 cases that consists of > 1 Primary Tumor Files, we will only used 1 of it


	# check cases with 3 files first
	'''for i in range(len(file_amount_met_long)):
		if file_amount_met_long[i,1] == "3":
			sample_type = data_met_long[np.where(data_met_long[:,case_id_column] == file_amount_met_long[i,0]),sample_type_column][0]
			print(sample_type)'''

	# out of 6 cases =
	#	2 cases consist of 3 Primary Tumor files
	#	3 cases are combination of Primary Tumor + Solid Tissue Normal + Metastatic
	#	1 cases are combination of 2 Primary Tumor files + 1 Solid Tissue Normal file
	# all cases can be used
	# out of the 3 cases that consists of > 1 Primary Tumor Files, we will only used 1 of it

	# so, 782 cases out of 789 can be used



# List of final cases for methylation (which consist of at least 1 tumor sample) 
def meta_methylation_used_case():
	# load cases_met_long_all and cases_remove_met_long
	with open(TARGET_METHYLATION + "cases_met_long_all.json") as f:
		file = yaml.safe_load(f)
	cases_met_long_all = np.asarray(file)

	with open(TARGET_METHYLATION + "cases_remove_met_long.json") as f:
		file = yaml.safe_load(f)
	cases_remove_met_long = np.asarray(file)

	# cases_met_long = cases_met_long_all - cases_remove_met_long
	cases_met_long = np.setdiff1d(cases_met_long_all,cases_remove_met_long)

	# save as json
	cases_met_long_j = json.dumps(cases_met_long.tolist(), indent=2)
	with open(TARGET_METHYLATION + "cases_met_long.json", "w") as text_file:
		text_file.write(cases_met_long_j)
	print("cases_met_long.json is created.")
	


# Check if the main cpg list (cpg.json) actually overlap with all files
def meta_methylation_check_cpg():
	# load cpg.json as the base file
	with open(TARGET_METHYLATION + "cpg.json") as file:
		cpg = yaml.safe_load(file)
	np_cpg = np.asarray(cpg)			# list of used_cpg

	# load methylation_beta_value
	data_met = np.genfromtxt(TARGET_META_CSV + "methylation_beta_value.csv", dtype=str, delimiter=',', skip_header=0)
	# find where the case id column is located in your meta_clinicals.csv
	file_id_column, = np.where(data_met[0]=='file_id')[0]
	file_name_column, = np.where(data_met[0]=='file_name')[0]	
	data_met = data_met[1:]

	overlap_count = 0

	# iterate for the whole 1234 methylation files
	for file in data_met:
		file_id = file[file_id_column]
		file_name = file[file_name_column]
		
		with open(DATASET_METHYLATION + file_id + "/" +  file_name) as f:
			rows = f.read().splitlines()

		new_cpg = []

		comment_line = 1
		for row in range(comment_line, len(rows)):
			line = rows[row].split("\t")
			new_cpg.append(line[0])

		np_new_cpg = np.asarray(new_cpg)
		# find the overlapping cpg
		np_overlap = np.intersect1d(np_cpg,np_new_cpg)

		# if the cpgs in this file are all in the base file cpg.json
		if np.array_equal(np_overlap,np_cpg):
			overlap_count = overlap_count + 1

	# number of files comply to cpg.json out of all files
	print(str(overlap_count) + " out of " + str(len(data_met)) + " comply with the base file cpg.json")



# Index list of each cpg.json elements inside cpg_short.json and cpg_long.json 
def meta_methylation_cpg_index():
	with open(TARGET_METHYLATION + "cpg.json") as f:
		cpg = yaml.safe_load(f)

	with open(TARGET_METHYLATION + "cpg_long.json") as f:
		cpg_long = yaml.safe_load(f)

	with open(TARGET_METHYLATION + "cpg_short.json") as f:
		cpg_short = yaml.safe_load(f)

	cpg_in_cpg_short_idx = []
	cpg_in_cpg_long_idx = []

	
	# 1. cpg_in_cpg_short_idx
	for i in range(len(cpg_short)):
		if cpg_short[i] in cpg:
			cpg_in_cpg_short_idx.append(i)

	cpg_in_cpg_short_idx_j = json.dumps(cpg_in_cpg_short_idx, indent=2)
	with open(TARGET_METHYLATION + "cpg_in_cpg_short_idx.json", "w") as f:
		f.write(cpg_in_cpg_short_idx_j)

	print("cpg_in_cpg_short_idx.json is created")


	# 2. cpg_in_cpg_long_idx
	for i in range(len(cpg_long)):
		if cpg_long[i] in cpg:
			cpg_in_cpg_long_idx.append(i)

	cpg_in_cpg_long_idx_j = json.dumps(cpg_in_cpg_long_idx, indent=2)
	with open(TARGET_METHYLATION + "cpg_in_cpg_long_idx.json", "w") as f:
		f.write(cpg_in_cpg_long_idx_j)

	print("cpg_in_cpg_long_idx.json is created")



# Count table size, find cases without tumor-file
def meta_gene():
	# 3666 files: 1222 htsec-count files + 1222 htsec-FPKM files + 1222 htsec-FPKM-UQ files
	# from 1092 cases (973 have 3 files, 110 have 6 files,  7 have 9 files, 2 have 12 files)
	# hypothesis: each cases same amount of count, FPKM, FPKM-UQ files

	# load file_amount.csv and data_gene.csv
	file_amount = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
	data_gene = np.genfromtxt(TARGET_META_CSV + "gene_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)
	
	# find where the case id column is located in your meta_clinicals.csv
	file_id_column, = np.where(data_gene[0]=='file_id')[0]
	file_name_column, = np.where(data_gene[0]=='file_name')[0]
	case_id_column, = np.where(data_gene[0]=='cases.0.case_id')[0]
	workflow_column, = np.where(data_gene[0]=='analysis.workflow_type')[0]
	sample_type_column, = np.where(data_gene[0]=='cases.0.samples.0.sample_type')[0]
	
	data_gene = data_gene[1:]

	#############################
	######### TABLE SIZE ########
	#############################
	'''temp_list = []

	for file in data_gene:
		with gzip.open(DATASET_GENE + file[file_id_column] + "/" + file[file_name_column]) as f:
			temp = np.genfromtxt(f, dtype=str, delimiter='\t')

		temp_list.append(temp.shape[0])
		print(temp.shape[0])

	print(len(temp_list))
	print(list(set(temp_list)))'''

	# 2444 files (possibly htsec-FPKM, htsec-FPKM UQ) have 60483 rows
	# 1222 files (possibly htsec-count) have 60488 rows
	
	
	#############################
	######## SAMPLE TYPE ########
	#############################
	# only include cases that at least have 1 tumor sample, because we do classification/regression only between tumor patients 

	# cek cases with 3 files first
	'''temp_list = []

	for i in range(len(file_amount)):
		if file_amount[i,9] == "3":
			temp_list.append(data_gene[np.where((data_gene[:,case_id_column] == file_amount[i,0]) * (data_gene[:,workflow_column] == "HTSeq - Counts")),sample_type_column][0,0])

	print(temp_list)'''
	# 1 out of the 973 cases ('2b22db1d-54a1-4b9e-a86e-a174cf51d95c') only have normal sample. So this will be excluded.


	# cek cases with 6 files first
	'''temp_list = []

	for i in range(len(file_amount)):
		if file_amount[i,9] == "6":
			temp_list.append(data_gene[np.where((data_gene[:,case_id_column] == file_amount[i,0]) * (data_gene[:,workflow_column] == "HTSeq - Counts")),sample_type_column][0])

	print(temp_list)'''
	# out of 110 cases =
	# 	106 cases are combination of Primary Tumor + Solid Tissue Normal
	# 	4 cases are combination of Primary Tumor + Metastatic
	# so, all cases can be used, we only used the Primary Tumor files


	# cek cases with 9 files first
	temp_list = []

	for i in range(len(file_amount)):
		if file_amount[i,9] == "9":
			temp_list.append([file_amount[i,0],data_gene[np.where((data_gene[:,case_id_column] == file_amount[i,0]) * (data_gene[:,workflow_column] == "HTSeq - Counts")),sample_type_column][0]])

	print(temp_list)
	# out of 7 cases =
	#	3 cases consist of 3 Primary Tumor files
	#	3 cases are combination of Primary Tumor + Solid Tissue Normal + Metastatic
	#	1 cases are combination of 2 Primary Tumor files + 1 Solid Tissue Normal file
	# all cases can be used
	# out of the 4 cases that consists of > 1 Primary Tumor Files, we will only used 1 of it


	# cek cases with 12 files first
	'''temp_list = []

	for i in range(len(file_amount)):
		if file_amount[i,9] == "12":
			temp_list.append([file_amount[i,0],data_gene[np.where((data_gene[:,case_id_column] == file_amount[i,0]) * (data_gene[:,workflow_column] == "HTSeq - Counts")),sample_type_column][0]])

	print(temp_list)'''
	# out of 2 cases, both cases consist of 3 Primary Tumor files + 1 Solid Tissue Normal file
	# all cases can be used
	# out of the 2 cases that consists of > 1 Primary Tumor Files, we will only used 1 of it

	# so, total there is 1091 cases (1092 - 1) that can be used



# Count table size, find cases without tumor-file
def meta_mirna():
	# 1207 files
	# from 1079 cases (962 have 1 files, 108 have 2 files, 7 have 3 files, 2 have 4 files)

	# load file_amount.csv and data_gene.csv
	file_amount = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
	data_mir = np.genfromtxt(TARGET_META_CSV + "mirna_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)
	
	# find where the case id column is located in your meta_clinicals.csv
	file_id_column, = np.where(data_mir[0]=='file_id')[0]
	file_name_column, = np.where(data_mir[0]=='file_name')[0]
	case_id_column, = np.where(data_mir[0]=='cases.0.case_id')[0]
	sample_type_column, = np.where(data_mir[0]=='cases.0.samples.0.sample_type')[0]
	
	data_mir = data_mir[1:]

	#############################
	######### TABLE SIZE ########
	#############################
	'''temp_list = []

	for file in data_mir:
		file_id = file[file_id_column]
		file_name = file[file_name_column]

		temp = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t')

		temp_list.append(temp.shape[0])
		print(temp.shape[0])

	print(len(temp_list))
	print(list(set(temp_list)))'''

	# all 1207 files have 1882 rows
	
	
	#############################
	######## SAMPLE TYPE ########
	#############################
	# out of 1207 files, 1096 files from Primary Tumor, 104 files from Solid Tissue Normal, 7 files from Metastatic
	# only include cases that at least have 1 tumor sample, because we do classification/regression only between tumor patients 

	# cek cases with 1 files first
	'''temp_list = []

	for i in range(len(file_amount)):
		if file_amount[i,10] == "1":
			temp_list.append([file_amount[i,0],data_mir[np.where(data_mir[:,case_id_column] == file_amount[i,0]),sample_type_column][0,0]])

	print(temp_list)'''
	# 1 out of the 962 cases ('3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6') only have normal sample. So this will be excluded.


	# cek cases with 2 files first
	'''temp_list = []

	for i in range(len(file_amount)):
		if file_amount[i,10] == "2":
			temp_list.append(data_mir[np.where(data_mir[:,case_id_column] == file_amount[i,0]),sample_type_column][0])

	print(temp_list)'''
	# out of 108 cases =
	#	6 cases consist of 2 Primary Tumor files
	# 	97 cases are combination of Primary Tumor + Solid Tissue Normal
	# 	5 cases are combination of Primary Tumor + Metastatic
	# so, all cases can be used
	# out of the 6 cases that consists of > 1 Primary Tumor Files, we will only used 1 of it


	# cek cases with 3 files first
	'''temp_list = []

	for i in range(len(file_amount)):
		if file_amount[i,10] == "3":
			temp_list.append([file_amount[i,0],data_mir[np.where(data_mir[:,case_id_column] == file_amount[i,0]),sample_type_column][0]])

	print(temp_list)'''
	# out of 7 cases =
	#	3 cases consist of 3 Primary Tumor files
	#	2 cases are combination of Primary Tumor + Solid Tissue Normal + Metastatic
	#	2 cases are combination of 2 Primary Tumor files + 1 Solid Tissue Normal file
	# all cases can be used
	# out of the 5 cases that consists of > 1 Primary Tumor Files, we will only used 1 of it


	# cek cases with 4 files first
	temp_list = []

	for i in range(len(file_amount)):
		if file_amount[i,10] == "4":
			temp_list.append([file_amount[i,0],data_mir[np.where(data_mir[:,case_id_column] == file_amount[i,0]),sample_type_column][0]])

	print(temp_list)
	# out of 2 cases, both cases consist of 3 Primary Tumor files + 1 Solid Tissue Normal file
	# all cases can be used
	# out of the 2 cases that consists of > 1 Primary Tumor Files, we will only used 1 of it

	# so, total there is 1078 cases (1079 - 1) that can be used



if __name__ == '__main__':
	#meta_methylation_file_size()
    meta_methylation_list_files()
    meta_methylation_cpg()
    meta_long_methylation()
    meta_methylation_sample_type()
    meta_methylation_used_case()
    #meta_methylation_check_cpg()
    meta_methylation_cpg_index()
    
    #meta_gene()

    #meta_mirna()