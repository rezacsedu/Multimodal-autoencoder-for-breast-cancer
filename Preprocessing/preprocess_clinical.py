from folder_location import *

import os
import csv
import json
import yaml
import numpy as np
import timeit
import xml.etree.ElementTree as ET
from datetime import datetime



ns = {'brca': 'http://tcga.nci/bcr/xml/clinical/brca/2.7',
      'admin': 'http://tcga.nci/bcr/xml/administration/2.7',
      'shared': 'http://tcga.nci/bcr/xml/shared/2.7',
      'clin_shared': 'http://tcga.nci/bcr/xml/clinical/shared/2.7',
      'brca_shared': 'http://tcga.nci/bcr/xml/clinical/brca/shared/2.7',
      'shared_stage': 'http://tcga.nci/bcr/xml/clinical/shared/stage/2.7',
      'brca_nte': 'http://tcga.nci/bcr/xml/clinical/brca/shared/new_tumor_event/2.7/1.0',
      'nte': 'http://tcga.nci/bcr/xml/clinical/shared/new_tumor_event/2.7',
      'follow_up_v1.5': 'http://tcga.nci/bcr/xml/clinical/brca/followup/2.7/1.5',
      'follow_up_v2.1': 'http://tcga.nci/bcr/xml/clinical/brca/followup/2.7/2.1',
      'follow_up_v4.0': 'http://tcga.nci/bcr/xml/clinical/brca/followup/2.7/4.0',
      'rx': 'http://tcga.nci/bcr/xml/clinical/pharmaceutical/2.7',
      'rad': 'http://tcga.nci/bcr/xml/clinical/radiation/2.7'}



# List all prefix in patients' clinical XML files
def all_prefix():
    # Load meta for clinical data
    meta_clinicals = np.genfromtxt(TARGET_META_CSV + "clinical_supplement.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(meta_clinicals[0]=='file_id')[0]
    file_name_column, = np.where(meta_clinicals[0]=='file_name')[0]

    meta_clinicals = meta_clinicals[1:]
    
    # Empty list for prefix
    prefix = []

    # Iterate for every patient's clinical data
    for meta_clinical in meta_clinicals:
        file_id = meta_clinical[file_id_column]
        file_name = meta_clinical[file_name_column] 
        
        # parse the XML file
        tree = ET.parse(DATASET_CLINICAL + file_id + "/" + file_name)

        # Iterate for every line in a clinical XML file
        for elements in tree.iter():
            prefix.append((elements.tag.split("{", 1)[1]).split("}", 1)[0])

        # Remove duplicate prefixes
        prefix = list(set(prefix))

    print(prefix)

    # Resulting prefix:
    #   http://tcga.nci/bcr/xml/clinical/brca/shared/new_tumor_event/2.7/1.0
    #   http://tcga.nci/bcr/xml/clinical/shared/new_tumor_event/2.7
    #   http://tcga.nci/bcr/xml/shared/2.7
    #   http://tcga.nci/bcr/xml/clinical/brca/2.7
    #   http://tcga.nci/bcr/xml/clinical/shared/stage/2.7
    #   http://tcga.nci/bcr/xml/administration/2.7
    #   http://tcga.nci/bcr/xml/clinical/radiation/2.7
    #   http://tcga.nci/bcr/xml/clinical/shared/2.7
    #   http://tcga.nci/bcr/xml/clinical/brca/followup/2.7/2.1
    #   http://tcga.nci/bcr/xml/clinical/brca/followup/2.7/4.0
    #   http://tcga.nci/bcr/xml/clinical/brca/shared/2.7
    #   http://tcga.nci/bcr/xml/clinical/pharmaceutical/2.7
    #   http://tcga.nci/bcr/xml/clinical/brca/followup/2.7/1.5

    # we define the ns constant based on this list



def compare_elmt():
    # Load meta for clinical data
    meta_clinicals = np.genfromtxt(TARGET_META_CSV + "clinical_supplement.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(meta_clinicals[0]=='file_id')[0]
    file_name_column, = np.where(meta_clinicals[0]=='file_name')[0]

    meta_clinicals = meta_clinicals[1:]
    
    elmts_tag = []
    elmts_attrib = []
    elmts_text = []

    cc = 0

    '''start = 0
    end = 18

    for i in range(end-start):
        elmts_text.append([])'''

    for meta_clinical in meta_clinicals:
        file_id = meta_clinical[file_id_column]
        file_name = meta_clinical[file_name_column] 

        # parse the XML file and take the root
        tree = ET.parse(DATASET_CLINICAL + file_id + "/" + file_name)
        root = tree.getroot()

        temp_tag = []
        temp_attrib = []
        temp_text = []

        for child in root[1][102]:
            temp_text = []
            
            for e in child[11]:
                temp_text.append(e.text)
            
            elmts_text.append(temp_text)

    #compare_tag = elmts_tag[1:] == elmts_tag[:-1]
    #compare_attrib = elmts_attrib[1:] == elmts_attrib[:-1]
    #compare_text = elmts_text[1:] == elmts_text[:-1]

    #print("Elements' tag is equal = %r" % compare_tag)
    #print("Elements' attribute is equal = %r" % compare_attrib)
    #print("Elements' text is equal = %r" % compare_text)
    #print(elmts_tag)
    #print(elmts_attrib)
    print(elmts_text)
    #print(cc)



def compare_elmt_len():
    # Load meta for clinical data
    meta_clinicals = np.genfromtxt(TARGET_META_CSV + "clinical_supplement.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(meta_clinicals[0]=='file_id')[0]
    file_name_column, = np.where(meta_clinicals[0]=='file_name')[0]

    meta_clinicals = meta_clinicals[1:]
    
    elmts_len = []

    for meta_clinical in meta_clinicals:
        # find the file id and file name column in your clinical_supplement.csv
        file_id = meta_clinical[file_id_column]
        file_name = meta_clinical[file_name_column] 

        # parse the XML file and take the root
        tree = ET.parse(DATASET_CLINICAL + file_id + "/" + file_name)
        root = tree.getroot()

        temp_len = []

        if len(root[1][102]) > 0:
            for child in root[1][102]:
                for e in child[11]:
                    elmts_len.append(len(e))
                    
        if (len(root[1][103]) > 0):
            for child in root[1][103]:
                temp_len = []
                for e in child:
                    temp_len.append(len(e))

                elmts_len.append(temp_len)

        elmts_len.append(len(root[1][103]))

        for child in root[1][101]:
            if child.tag == "{http://tcga.nci/bcr/xml/clinical/brca/followup/2.7/4.0}follow_up":
                elmts_len.append(len(root[1][97]))

        os.chdir('..')

    compare_len = elmts_len[1:] == elmts_len[:-1]

    print("Root's len is equal = %r" % compare_len)
    print(elmts_len)
    print(meta_clinicals[607]['file_id'])
    print(meta_clinicals[359]['file_name'])



# Create a file of each patient's whole ids (drug id, radiation id, follow-up id)
def uuid():
    # Load meta for clinical data
    meta_clinicals = np.genfromtxt(TARGET_META_CSV + "clinical_supplement.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(meta_clinicals[0]=='file_id')[0]
    file_name_column, = np.where(meta_clinicals[0]=='file_name')[0]
    case_id_column, = np.where(meta_clinicals[0]=='cases.0.case_id')[0]

    meta_clinicals = meta_clinicals[1:]

    # Empty dictionary for the final data
    new_dict = {}

    # Iterate for each patient's clinical data and take the drugs', radiations', and follow-ups' id
    for meta_clinical in meta_clinicals:
        # find the file id and file name column in your clinical_supplement.csv
        file_id = meta_clinical[file_id_column]
        file_name = meta_clinical[file_name_column] 

        # parse the XML file and take the root
        tree = ET.parse(DATASET_CLINICAL + file_id + "/" + file_name)
        root = tree.getroot()

        # Empty dict per case id
        new_dict[meta_clinical[case_id_column]] = {'drugs':[], 'radiations':[], 'follow_ups_15':[], 'follow_ups_21':[], 'follow_ups_40':[]}

        # Drugs
        for child in root[1].find('rx:drugs', ns):
            new_dict[meta_clinical[case_id_column]]['drugs'].append(child.find('rx:bcr_drug_uuid', ns).text)

        # Radiations
        for child in root[1].find('rad:radiations', ns):
            new_dict[meta_clinical[case_id_column]]['radiations'].append(child.find('rad:bcr_radiation_uuid', ns).text)

        # Follow-up 1.5
        for child in root[1].find('brca:follow_ups', ns).findall('follow_up_v1.5:follow_up', ns):
            new_dict[meta_clinical[case_id_column]]['follow_ups_15'].append(child.find('clin_shared:bcr_followup_uuid', ns).text)

        # Follow-up 2.1
        for child in root[1].find('brca:follow_ups', ns).findall('follow_up_v2.1:follow_up', ns):
            new_dict[meta_clinical[case_id_column]]['follow_ups_21'].append(child.find('clin_shared:bcr_followup_uuid', ns).text)

        # Follow-up 4.0
        for child in root[1].find('brca:follow_ups', ns).findall('follow_up_v4.0:follow_up', ns):
            new_dict[meta_clinical[case_id_column]]['follow_ups_40'].append(child.find('clin_shared:bcr_followup_uuid', ns).text)

    
    # Create folder for clinical processed data
    if not os.path.isdir(TARGET_CLINICAL):
        os.makedirs(TARGET_CLINICAL)

    # Format and save as json
    new_dict_j = json.dumps(new_dict, indent=2)
    with open(TARGET_CLINICAL + "uuid.json", "w") as text_file:
        text_file.write(new_dict_j)

    print("uuid.json is created.")



# In TCGA, each events (first occurence, radiations, drugs, follow-ups) are treated separately at different time
# The time when these events finish are called form completion date
# So there are general completion date, radiation date, and etc
# Create a file for every patient's whole form completion date
def form_completion():
    # Load meta for clinical data
    meta_clinicals = np.genfromtxt(TARGET_META_CSV + "clinical_supplement.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(meta_clinicals[0]=='file_id')[0]
    file_name_column, = np.where(meta_clinicals[0]=='file_name')[0]
    case_id_column, = np.where(meta_clinicals[0]=='cases.0.case_id')[0]

    meta_clinicals = meta_clinicals[1:]

    #############################
    ##### 1. CREATE AS JSON #####
    #############################
    # Empty dictionary for the final data
    new_dict = {}

    # Iterate for each patient's clinical data and take the drugs', radiations', and follow-ups' id
    for meta_clinical in meta_clinicals:
        # find the file id and file name column in your clinical_supplement.csv
        file_id = meta_clinical[file_id_column]
        file_name = meta_clinical[file_name_column] 

        # parse the XML file and take the root
        tree = ET.parse(DATASET_CLINICAL + file_id + "/" + file_name)
        root = tree.getroot()

        # Empty dict per case id
        new_dict[meta_clinical[case_id_column]] = {'general':{}, 'drugs':{}, 'radiations':{}, 'follow_ups_15':{}, 'follow_ups_21':{}, 'follow_ups_40':{}}

        # General
        new_dict[meta_clinical[case_id_column]]['general']['day'] = root[1].find('clin_shared:day_of_form_completion', ns).text
        new_dict[meta_clinical[case_id_column]]['general']['month'] = root[1].find('clin_shared:month_of_form_completion', ns).text
        new_dict[meta_clinical[case_id_column]]['general']['year'] = root[1].find('clin_shared:year_of_form_completion', ns).text

        # Drugs
        for child in root[1].find('rx:drugs', ns):
            new_dict[meta_clinical[case_id_column]]['drugs'][child.find('rx:bcr_drug_uuid', ns).text] = {
            'day':child.find('clin_shared:day_of_form_completion', ns).text,
            'month':child.find('clin_shared:month_of_form_completion', ns).text,
            'year':child.find('clin_shared:year_of_form_completion', ns).text
            }

        # Radiations
        for child in root[1].find('rad:radiations', ns):
            new_dict[meta_clinical[case_id_column]]['radiations'][child.find('rad:bcr_radiation_uuid', ns).text] = {
            'day':child.find('clin_shared:day_of_form_completion', ns).text,
            'month':child.find('clin_shared:month_of_form_completion', ns).text,
            'year':child.find('clin_shared:year_of_form_completion', ns).text
            }

        # Follow-up 1.5
        for child in root[1].find('brca:follow_ups', ns).findall('follow_up_v1.5:follow_up', ns):
            new_dict[meta_clinical[case_id_column]]['follow_ups_15'][child.find('clin_shared:bcr_followup_uuid', ns).text] = {
            'day':child.find('clin_shared:day_of_form_completion', ns).text,
            'month':child.find('clin_shared:month_of_form_completion', ns).text,
            'year':child.find('clin_shared:year_of_form_completion', ns).text
            }

        # Follow-up 2.1
        for child in root[1].find('brca:follow_ups', ns).findall('follow_up_v2.1:follow_up', ns):
            new_dict[meta_clinical[case_id_column]]['follow_ups_21'][child.find('clin_shared:bcr_followup_uuid', ns).text] = {
            'day':child.find('clin_shared:day_of_form_completion', ns).text,
            'month':child.find('clin_shared:month_of_form_completion', ns).text,
            'year':child.find('clin_shared:year_of_form_completion', ns).text
            }

        # Follow-up 4.0
        for child in root[1].find('brca:follow_ups', ns).findall('follow_up_v4.0:follow_up', ns):
            new_dict[meta_clinical[case_id_column]]['follow_ups_40'][child.find('clin_shared:bcr_followup_uuid', ns).text] = {
            'day':child.find('clin_shared:day_of_form_completion', ns).text,
            'month':child.find('clin_shared:month_of_form_completion', ns).text,
            'year':child.find('clin_shared:year_of_form_completion', ns).text
            }

    # Save as json
    new_dict_j = json.dumps(new_dict, indent=2)
    with open(TARGET_CLINICAL + "form_completion.json", "w") as text_file:
        text_file.write(new_dict_j)

    print("form_completion.json is created.")


    #############################
    ####### 2. JSON TO CSV ######
    #############################
    # This file compare whether drugs/radiations/follow-ups data came the same as / before / after general completion date
    meta_clinicals_case_id = meta_clinicals[:,case_id_column]

    with open(TARGET_CLINICAL + "uuid.json") as file:
        uuid = yaml.safe_load(file)

    with open(TARGET_CLINICAL + "form_completion.json") as file:
        form_completion_dict = yaml.safe_load(file)

    # create target file
    with open(TARGET_CLINICAL + 'form_completion.csv', 'w') as csvfile:
        # init file and headers
        fieldnames = ['Case',
                      'Drugs', 'Drugs == General', 'Drugs < General', 'Drugs > General',
                      'Radiations', 'Radiations == General', 'Radiations < General', 'Radiations > General',
                      'Follow-Up 1.5', 'Follow-Up 1.5 == General', 'Follow-Up 1.5 < General', 'Follow-Up 1.5 > General',
                      'Follow-Up 2.1', 'Follow-Up 2.1 == General', 'Follow-Up 2.1 < General', 'Follow-Up 2.1 > General',
                      'Follow-Up 4.0', 'Follow-Up 4.0 == General', 'Follow-Up 4.0 < General', 'Follow-Up 4.0 > General']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # iterate per case
        for case in meta_clinicals_case_id:
            # Set general completion date
            date_general = datetime(int(form_completion_dict[case]['general']['year']), int(form_completion_dict[case]['general']['month']), int(form_completion_dict[case]['general']['day']))
            
            row = {}
            
            # case
            row['Case'] = case
            

            # number of drugs and comparison to general completion date
            row['Drugs'] = len(form_completion_dict[case]['drugs'])
            eq = 0
            le = 0
            mo = 0
            for drug in uuid[case]['drugs']:
                date_drug = datetime(int(form_completion_dict[case]['drugs'][drug]['year']), int(form_completion_dict[case]['drugs'][drug]['month']), int(form_completion_dict[case]['drugs'][drug]['day']))
                if date_drug == date_general:
                    eq = eq + 1
                elif date_drug < date_general:
                    le = le + 1
                elif date_drug > date_general:
                    mo = mo + 1

            row['Drugs == General'] = eq
            row['Drugs < General'] = le
            row['Drugs > General'] = mo


            # number of radiations and comparison to general completion date
            row['Radiations'] = len(form_completion_dict[case]['radiations'])
            eq = 0
            le = 0
            mo = 0
            for radiation in uuid[case]['radiations']:
                date_radiation = datetime(int(form_completion_dict[case]['radiations'][radiation]['year']), int(form_completion_dict[case]['radiations'][radiation]['month']), int(form_completion_dict[case]['radiations'][radiation]['day']))
                if date_radiation == date_general:
                    eq = eq + 1
                elif date_radiation < date_general:
                    le = le + 1
                elif date_radiation > date_general:
                    mo = mo + 1
            
            row['Radiations == General'] = eq
            row['Radiations < General'] = le
            row['Radiations > General'] = mo


            # number of follow-up 1.5 and comparison to general completion date
            row['Follow-Up 1.5'] = len(form_completion_dict[case]['follow_ups_15'])
            eq = 0
            le = 0
            mo = 0
            for follow_up_15 in uuid[case]['follow_ups_15']:
                date_follow_up_15 = datetime(int(form_completion_dict[case]['follow_ups_15'][follow_up_15]['year']), int(form_completion_dict[case]['follow_ups_15'][follow_up_15]['month']), int(form_completion_dict[case]['follow_ups_15'][follow_up_15]['day']))
                if date_follow_up_15 == date_general:
                    eq = eq + 1
                elif date_follow_up_15 < date_general:
                    le = le + 1
                elif date_follow_up_15 > date_general:
                    mo = mo + 1
            
            row['Follow-Up 1.5 == General'] = eq
            row['Follow-Up 1.5 < General'] = le
            row['Follow-Up 1.5 > General'] = mo


            # number of follow-up 2.1 and comparison to general completion date
            row['Follow-Up 2.1'] = len(form_completion_dict[case]['follow_ups_21'])
            eq = 0
            le = 0
            mo = 0
            for follow_up_21 in uuid[case]['follow_ups_21']:
                date_follow_up_21 = datetime(int(form_completion_dict[case]['follow_ups_21'][follow_up_21]['year']), int(form_completion_dict[case]['follow_ups_21'][follow_up_21]['month']), int(form_completion_dict[case]['follow_ups_21'][follow_up_21]['day']))
                if date_follow_up_21 == date_general:
                    eq = eq + 1
                elif date_follow_up_21 < date_general:
                    le = le + 1
                elif date_follow_up_21 > date_general:
                    mo = mo + 1
            
            row['Follow-Up 2.1 == General'] = eq
            row['Follow-Up 2.1 < General'] = le
            row['Follow-Up 2.1 > General'] = mo


            # number of follow-up 4.0 and comparison to general completion date
            row['Follow-Up 4.0'] = len(form_completion_dict[case]['follow_ups_40'])
            eq = 0
            le = 0
            mo = 0
            for follow_up_40 in uuid[case]['follow_ups_40']:
                date_follow_up_40 = datetime(int(form_completion_dict[case]['follow_ups_40'][follow_up_40]['year']), int(form_completion_dict[case]['follow_ups_40'][follow_up_40]['month']), int(form_completion_dict[case]['follow_ups_40'][follow_up_40]['day']))
                if date_follow_up_40 == date_general:
                    eq = eq + 1
                elif date_follow_up_40 < date_general:
                    le = le + 1
                elif date_follow_up_40 > date_general:
                    mo = mo + 1
            
            row['Follow-Up 4.0 == General'] = eq
            row['Follow-Up 4.0 < General'] = le
            row['Follow-Up 4.0 > General'] = mo

            writer.writerow(row)

    print("form_completion.csv is created.")

    # From the resulting form_completion.csv, we can see that there are no consistent chronological order between every events
    # E.g. sometimes patients' follow-ups happens before the general form completion date



# Create a file for every patient's general status (gender, race, ethniticity, and etc)
def general():
    # Load meta for clinical data
    meta_clinicals = np.genfromtxt(TARGET_META_CSV + "clinical_supplement.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(meta_clinicals[0]=='file_id')[0]
    file_name_column, = np.where(meta_clinicals[0]=='file_name')[0]
    case_id_column, = np.where(meta_clinicals[0]=='cases.0.case_id')[0]

    meta_clinicals = meta_clinicals[1:]

    #############################
    ##### 1. CREATE AS JSON #####
    #############################
    # Empty dictionary for the final data
    new_dict = {}

    # Iterate for each patient's clinical data and take the drugs', radiations', and follow-ups' id
    for meta_clinical in meta_clinicals:
        # find the file id and file name column in your clinical_supplement.csv
        file_id = meta_clinical[file_id_column]
        file_name = meta_clinical[file_name_column] 

        # parse the XML file and take the root
        tree = ET.parse(DATASET_CLINICAL + file_id + "/" + file_name)
        root = tree.getroot()

        new_dict[meta_clinical[case_id_column]] = {'vital_status':{}, 'gender':{}, 'race':{}, 'ethnicity':{}, 'menopause_status':{}, 'neoadjuvant_treatment':{}, 'result':{}}
        new_dict[meta_clinical[case_id_column]]['vital_status'] = root[1].find('clin_shared:vital_status', ns).text
        new_dict[meta_clinical[case_id_column]]['gender'] = root[1].find('shared:gender', ns).text
        new_dict[meta_clinical[case_id_column]]['race'] = root[1].find('clin_shared:race_list', ns).find('clin_shared:race', ns).text
        new_dict[meta_clinical[case_id_column]]['ethnicity'] = root[1].find('clin_shared:ethnicity', ns).text
        new_dict[meta_clinical[case_id_column]]['menopause_status'] = root[1].find('clin_shared:menopause_status', ns).text
        new_dict[meta_clinical[case_id_column]]['neoadjuvant_treatment'] = root[1].find('shared:history_of_neoadjuvant_treatment', ns).text
        new_dict[meta_clinical[case_id_column]]['result'] = root[1].find('clin_shared:person_neoplasm_cancer_status', ns).text


    # Save as json
    new_dict_j = json.dumps(new_dict, indent=2)
    with open(TARGET_CLINICAL + "general.json", "w") as text_file:
        text_file.write(new_dict_j)

    print("general.json is created.")


    #############################
    ####### 2. JSON TO CSV ######
    #############################
    meta_clinicals_case_id = meta_clinicals[:,case_id_column]

    with open(TARGET_CLINICAL + "general.json") as file:
        general = yaml.safe_load(file)

    with open(TARGET_CLINICAL + 'general.csv', 'w') as csvfile:
        fieldnames = ['Case', 'Vital Status', 'Gender', 'Race', 'Ethnicity', 'Menopause Status', 'Neoadjuvant Treatment', 'Result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for case in meta_clinicals_case_id:
            row = {}
            row['Case'] = case
            row['Vital Status'] = general[case]['vital_status']
            row['Gender'] = general[case]['gender']
            row['Race'] = general[case]['race']
            row['Ethnicity'] = general[case]['ethnicity']
            row['Menopause Status'] = general[case]['menopause_status']
            row['Neoadjuvant Treatment'] = general[case]['neoadjuvant_treatment']
            row['Result'] = general[case]['result']
            
            writer.writerow(row)

    print("general.csv is created.")



# Create a file for every patient's general pathological status (cancer site, histology, and etc)
def pathology_general():
    # Load meta for clinical data
    meta_clinicals = np.genfromtxt(TARGET_META_CSV + "clinical_supplement.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(meta_clinicals[0]=='file_id')[0]
    file_name_column, = np.where(meta_clinicals[0]=='file_name')[0]
    case_id_column, = np.where(meta_clinicals[0]=='cases.0.case_id')[0]

    meta_clinicals = meta_clinicals[1:]

    #############################
    ##### 1. CREATE AS JSON #####
    #############################
    # Empty dictionary for the final data
    new_dict = {}

    # Iterate for each patient's clinical data and take the drugs', radiations', and follow-ups' id
    for meta_clinical in meta_clinicals:
        # find the file id and file name column in your clinical_supplement.csv
        file_id = meta_clinical[file_id_column]
        file_name = meta_clinical[file_name_column] 


        # parse the XML file and take the root
        tree = ET.parse(DATASET_CLINICAL + file_id + "/" + file_name)
        root = tree.getroot()


        # Initialization for each case
        new_dict[meta_clinical[case_id_column]] = {'method':{}, 'prospective_collection':{}, 'retrospective_collection':{}, 'site':{}, 'specific_site':[], 'histological_type':{}, 'histological_type_icd_o_3':{}}


        # Surgical method
        method = root[1].find('clin_shared:initial_pathologic_diagnosis_method', ns).text
        other_method = root[1].find('clin_shared:init_pathology_dx_method_other', ns).text
        if method == "Other method, specify:":
            if other_method == "Wide local incision":
                new_dict[meta_clinical[case_id_column]]['method'] = "Excisional Biopsy"
            elif other_method == "Biopsy, NOS" or other_method == "biopsy, NOS" or other_method == "Biopsy not specified" or other_method == "Ultrasound-guided biopsy":
                new_dict[meta_clinical[case_id_column]]['method'] = "Biopsy, NOS"
            elif other_method == "stereotactic biopsy":
                new_dict[meta_clinical[case_id_column]]['method'] = "Stereotactic biopsy"
            elif other_method == "Skin biopsy" or other_method == "SKIN BIOPSY":
                new_dict[meta_clinical[case_id_column]]['method'] = "Skin biopsy"
            elif other_method == "Lumpectomy":
                new_dict[meta_clinical[case_id_column]]['method'] = "Lumpectomy"
            elif other_method == "Modified Radical Masectomy" or other_method == "Patey's Suregery" or other_method == "Patey's Surgery":
                new_dict[meta_clinical[case_id_column]]['method'] = "Mastectomy"
            elif other_method == "intraoperative examination":
                new_dict[meta_clinical[case_id_column]]['method'] = "Intraoperative examination"
            elif other_method == "Ultrasound-guided mammotome biopsy":
                new_dict[meta_clinical[case_id_column]]['method'] = "Mammotome biopsy"
            elif other_method == None:
                new_dict[meta_clinical[case_id_column]]['method'] = None
        else:
            new_dict[meta_clinical[case_id_column]]['method'] = method


        # Prospective collection
        new_dict[meta_clinical[case_id_column]]['prospective_collection'] = root[1].find('clin_shared:tissue_prospective_collection_indicator', ns).text
        new_dict[meta_clinical[case_id_column]]['retrospective_collection'] = root[1].find('clin_shared:tissue_retrospective_collection_indicator', ns).text


        # Cancer site
        if (root[1].find('brca:anatomic_neoplasm_subdivisions', ns)[0].text == "Right") or (root[1].find('brca:anatomic_neoplasm_subdivisions', ns)[0].text == "Right Upper Inner Quadrant") or (root[1].find('brca:anatomic_neoplasm_subdivisions', ns)[0].text == "Right Upper Outer Quadrant") or (root[1].find('brca:anatomic_neoplasm_subdivisions', ns)[0].text == "Right Lower Inner Quadrant") or (root[1].find('brca:anatomic_neoplasm_subdivisions', ns)[0].text == "Right Lower Outer Quadrant"):
            new_dict[meta_clinical[case_id_column]]['site'] = "Right"
        if (root[1].find('brca:anatomic_neoplasm_subdivisions', ns)[0].text == "Left") or (root[1].find('brca:anatomic_neoplasm_subdivisions', ns)[0].text == "Left Upper Inner Quadrant") or (root[1].find('brca:anatomic_neoplasm_subdivisions', ns)[0].text == "Left Upper Outer Quadrant") or (root[1].find('brca:anatomic_neoplasm_subdivisions', ns)[0].text == "Left Lower Inner Quadrant") or (root[1].find('brca:anatomic_neoplasm_subdivisions', ns)[0].text == "Left Lower Outer Quadrant"):
            new_dict[meta_clinical[case_id_column]]['site'] = "Left"


        # Specific site
        for child in root[1].find('brca:anatomic_neoplasm_subdivisions', ns):
            new_dict[meta_clinical[case_id_column]]['specific_site'].append(child.text)

        if "Left" in new_dict[meta_clinical[case_id_column]]['specific_site']:
            new_dict[meta_clinical[case_id_column]]['specific_site'].remove("Left")

        if "Left Lower Inner Quadrant" in new_dict[meta_clinical[case_id_column]]['specific_site']:
            new_dict[meta_clinical[case_id_column]]['specific_site'].remove("Left Lower Inner Quadrant")
            new_dict[meta_clinical[case_id_column]]['specific_site'].append("Lower Inner Quadrant")

        if "Left Lower Outer Quadrant" in new_dict[meta_clinical[case_id_column]]['specific_site']:
            new_dict[meta_clinical[case_id_column]]['specific_site'].remove("Left Lower Outer Quadrant")
            new_dict[meta_clinical[case_id_column]]['specific_site'].append("Lower Outer Quadrant")

        if "Left Upper Inner Quadrant" in new_dict[meta_clinical[case_id_column]]['specific_site']:
            new_dict[meta_clinical[case_id_column]]['specific_site'].remove("Left Upper Inner Quadrant")
            new_dict[meta_clinical[case_id_column]]['specific_site'].append("Upper Inner Quadrant")

        if "Left Upper Outer Quadrant" in new_dict[meta_clinical[case_id_column]]['specific_site']:
            new_dict[meta_clinical[case_id_column]]['specific_site'].remove("Left Upper Outer Quadrant")
            new_dict[meta_clinical[case_id_column]]['specific_site'].append("Upper Outer Quadrant")

        if "Right" in new_dict[meta_clinical[case_id_column]]['specific_site']:
            new_dict[meta_clinical[case_id_column]]['specific_site'].remove("Right")

        if "Right Lower Inner Quadrant" in new_dict[meta_clinical[case_id_column]]['specific_site']:
            new_dict[meta_clinical[case_id_column]]['specific_site'].remove("Right Lower Inner Quadrant")
            new_dict[meta_clinical[case_id_column]]['specific_site'].append("Lower Inner Quadrant")

        if "Right Lower Outer Quadrant" in new_dict[meta_clinical[case_id_column]]['specific_site']:
            new_dict[meta_clinical[case_id_column]]['specific_site'].remove("Right Lower Outer Quadrant")
            new_dict[meta_clinical[case_id_column]]['specific_site'].append("Lower Outer Quadrant")

        if "Right Upper Inner Quadrant" in new_dict[meta_clinical[case_id_column]]['specific_site']:
            new_dict[meta_clinical[case_id_column]]['specific_site'].remove("Right Upper Inner Quadrant")
            new_dict[meta_clinical[case_id_column]]['specific_site'].append("Upper Inner Quadrant")

        if "Right Upper Outer Quadrant" in new_dict[meta_clinical[case_id_column]]['specific_site']:
            new_dict[meta_clinical[case_id_column]]['specific_site'].remove("Right Upper Outer Quadrant")
            new_dict[meta_clinical[case_id_column]]['specific_site'].append("Upper Outer Quadrant")


        # Histology
        histology = root[1].find('shared:histological_type', ns).text
        other_histology = root[1].find('shared:histological_type_other', ns).text
        if histology == "Mixed Histology (please specify)":
            new_dict[meta_clinical[case_id_column]]['histological_type'] = "Infiltrating Ductal and Lobular Carcinoma"
        elif histology == "Other, specify":
            new_dict[meta_clinical[case_id_column]]['histological_type'] = None
        else:
            new_dict[meta_clinical[case_id_column]]['histological_type'] = histology


        # Histology (ICD-O-3)
        new_dict[meta_clinical[case_id_column]]['histological_type_icd_o_3'] = root[1].find('clin_shared:icd_o_3_histology', ns).text


    # Save as json
    new_dict_j = json.dumps(new_dict, indent=2)
    with open(TARGET_CLINICAL + "pathology_general.json", "w") as text_file:
        text_file.write(new_dict_j)

    print("pathology_general.json is created.")


    #############################
    ####### 2. JSON TO CSV ######
    #############################
    meta_clinicals_case_id = meta_clinicals[:,case_id_column]

    with open(TARGET_CLINICAL + "pathology_general.json") as file:
        pathology_general = yaml.safe_load(file)

    with open(TARGET_CLINICAL + 'pathology_general.csv', 'w') as csvfile:
        fieldnames = ['Case','Method', 'Prospective Collection', 'Retrospective Collection', 'Histological Type', 'Histological Type (ICD-O-3)', 'Site', 'Site (Upper Inner)', 'Site (Upper Outer)', 'Site (Lower Inner)', 'Site (Lower Outer)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for case in meta_clinicals_case_id:
            row = {}
            row['Case'] = case
            row['Method'] = pathology_general[case]['method']
            row['Prospective Collection'] = pathology_general[case]['prospective_collection']
            row['Retrospective Collection'] = pathology_general[case]['retrospective_collection']
            row['Histological Type'] = pathology_general[case]['histological_type']
            row['Histological Type (ICD-O-3)'] = pathology_general[case]['histological_type_icd_o_3']
            row['Site'] = pathology_general[case]['site']
            if "Upper Inner Quadrant" in pathology_general[case]['specific_site']:
                row['Site (Upper Inner)'] = "YES"
            if "Upper Outer Quadrant" in pathology_general[case]['specific_site']:
                row['Site (Upper Outer)'] = "YES"
            if "Lower Inner Quadrant" in pathology_general[case]['specific_site']:
                row['Site (Lower Inner)'] = "YES"
            if "Lower Outer Quadrant" in pathology_general[case]['specific_site']:
                row['Site (Lower Outer)'] = "YES"
            
            writer.writerow(row)

    print("pathology_general.csv is created.")



# Create a file for every patient's pathological receptor status (ER, PGR, and HER2/neu)
def pathology_receptor():
    # Load meta for clinical data
    meta_clinicals = np.genfromtxt(TARGET_META_CSV + "clinical_supplement.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(meta_clinicals[0]=='file_id')[0]
    file_name_column, = np.where(meta_clinicals[0]=='file_name')[0]
    case_id_column, = np.where(meta_clinicals[0]=='cases.0.case_id')[0]

    meta_clinicals = meta_clinicals[1:]

    #############################
    ##### 1. CREATE AS JSON #####
    #############################
    # Empty dictionary for the final data
    new_dict = {}

    # Iterate for each patient's clinical data and take the drugs', radiations', and follow-ups' id
    for meta_clinical in meta_clinicals:
        # find the file id and file name column in your clinical_supplement.csv
        file_id = meta_clinical[file_id_column]
        file_name = meta_clinical[file_name_column] 

        # parse the XML file and take the root
        tree = ET.parse(DATASET_CLINICAL + file_id + "/" + file_name)
        root = tree.getroot()

        new_dict[meta_clinical[case_id_column]] = {'er_percentage':{}, 'er_status':{}, 'pgr_percentage':{}, 'pgr_status':{}, 'her2_total_cell_count':{}, 'her2_percentage':{}, 'her2_ihc_status':{}, 'her2_fish_status':{}}

        # ER
        new_dict[meta_clinical[case_id_column]]['er_percentage'] = root[1].find('brca_shared:er_level_cell_percentage_category', ns).text
        new_dict[meta_clinical[case_id_column]]['er_status'] = root[1].find('brca_shared:breast_carcinoma_estrogen_receptor_status', ns).text

        # PGR
        new_dict[meta_clinical[case_id_column]]['pgr_percentage'] = root[1].find('brca_shared:progesterone_receptor_level_cell_percent_category', ns).text
        new_dict[meta_clinical[case_id_column]]['pgr_status'] = root[1].find('brca_shared:breast_carcinoma_progesterone_receptor_status', ns).text

        # HER2/neu
        new_dict[meta_clinical[case_id_column]]['her2_total_cell_count'] = root[1].find('brca_shared:her2_neu_and_centromere_17_copy_number_analysis_input_total_number_count', ns).text
        new_dict[meta_clinical[case_id_column]]['her2_percentage'] = root[1].find('brca_shared:her2_erbb_pos_finding_cell_percent_category', ns).text
        new_dict[meta_clinical[case_id_column]]['her2_ihc_status'] = root[1].find('brca_shared:lab_proc_her2_neu_immunohistochemistry_receptor_status', ns).text
        new_dict[meta_clinical[case_id_column]]['her2_fish_status'] = root[1].find('brca_shared:lab_procedure_her2_neu_in_situ_hybrid_outcome_type', ns).text


    # Save as json
    new_dict_j = json.dumps(new_dict, indent=2)
    with open(TARGET_CLINICAL + "pathology_receptor.json", "w") as text_file:
        text_file.write(new_dict_j)

    print("pathology_receptor.json is created.")


    #############################
    ####### 2. JSON TO CSV ######
    #############################
    meta_clinicals_case_id = meta_clinicals[:,case_id_column]

    with open(TARGET_CLINICAL + "pathology_receptor.json") as file:
        pathology_receptor = yaml.safe_load(file)

    with open(TARGET_CLINICAL + 'pathology_receptor.csv', 'w') as csvfile:
        fieldnames = ['Case', 'ER Percentage', 'ER Status', 'PGR Percentage', 'PGR Status', 'HER2 Total Cell Count', 'HER2 Percentage', 'HER2 IHC Status', 'HER2 FISH Status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for case in meta_clinicals_case_id:
            row = {}
            row['Case'] = case
            row['ER Percentage'] = pathology_receptor[case]['er_percentage']
            row['ER Status'] = pathology_receptor[case]['er_status']
            row['PGR Percentage'] = pathology_receptor[case]['pgr_percentage']
            row['PGR Status'] = pathology_receptor[case]['pgr_status']
            row['HER2 Total Cell Count'] = pathology_receptor[case]['her2_total_cell_count']
            row['HER2 Percentage'] = pathology_receptor[case]['her2_percentage']
            row['HER2 IHC Status'] = pathology_receptor[case]['her2_ihc_status']
            row['HER2 FISH Status'] = pathology_receptor[case]['her2_fish_status']
            
            writer.writerow(row)

    print("pathology_receptor.csv is created.")



# Create a file for every patient's pathological lymph status (amount of nodes examined, amount of positive nodes)
def pathology_lymph():
    # Load meta for clinical data
    meta_clinicals = np.genfromtxt(TARGET_META_CSV + "clinical_supplement.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(meta_clinicals[0]=='file_id')[0]
    file_name_column, = np.where(meta_clinicals[0]=='file_name')[0]
    case_id_column, = np.where(meta_clinicals[0]=='cases.0.case_id')[0]

    meta_clinicals = meta_clinicals[1:]

    #############################
    ##### 1. CREATE AS JSON #####
    #############################
    # Empty dictionary for the final data
    new_dict = {}

    # Iterate for each patient's clinical data and take the drugs', radiations', and follow-ups' id
    for meta_clinical in meta_clinicals:
        # find the file id and file name column in your clinical_supplement.csv
        file_id = meta_clinical[file_id_column]
        file_name = meta_clinical[file_name_column] 

        # parse the XML file and take the root
        tree = ET.parse(DATASET_CLINICAL + file_id + "/" + file_name)
        root = tree.getroot()

        new_dict[meta_clinical[case_id_column]] = {'amount_nodes_examined':{}, 'amount_nodes_positive_by_ihc':{}, 'amount_nodes_positive_by_he':{}}

        new_dict[meta_clinical[case_id_column]]['amount_nodes_examined'] = root[1].find('clin_shared:lymph_node_examined_count', ns).text
        new_dict[meta_clinical[case_id_column]]['amount_nodes_positive_by_ihc'] = root[1].find('clin_shared:number_of_lymphnodes_positive_by_ihc', ns).text
        new_dict[meta_clinical[case_id_column]]['amount_nodes_positive_by_he'] = root[1].find('clin_shared:number_of_lymphnodes_positive_by_he', ns).text


    # Save to json
    new_dict_j = json.dumps(new_dict, indent=2)
    with open(TARGET_CLINICAL + "pathology_lymph.json", "w") as text_file:
        text_file.write(new_dict_j)

    print("pathology_lymph.json is created.")


    #############################
    ####### 2. JSON TO CSV ######
    #############################
    meta_clinicals_case_id = meta_clinicals[:,case_id_column]

    with open(TARGET_CLINICAL + "pathology_lymph.json") as file:
        pathology_lymph = yaml.safe_load(file)

    with open(TARGET_CLINICAL + 'pathology_lymph.csv', 'w') as csvfile:
        fieldnames = ['Case', 'Amount of Nodes Examined', 'Amount of Positive Nodes by IHC', 'Amount of Positive Nodes by HE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for case in meta_clinicals_case_id:
            row = {}
            row['Case'] = case
            row['Amount of Nodes Examined'] = pathology_lymph[case]['amount_nodes_examined']
            row['Amount of Positive Nodes by IHC'] = pathology_lymph[case]['amount_nodes_positive_by_ihc']
            row['Amount of Positive Nodes by HE'] = pathology_lymph[case]['amount_nodes_positive_by_he']
            
            writer.writerow(row)

    print("pathology_lymph.csv is created.")



# Create a file for every patient's cancer stage (ajcc-based and tnm-based stage)
def pathology_stage():
    # Load meta for clinical data
    meta_clinicals = np.genfromtxt(TARGET_META_CSV + "clinical_supplement.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(meta_clinicals[0]=='file_id')[0]
    file_name_column, = np.where(meta_clinicals[0]=='file_name')[0]
    case_id_column, = np.where(meta_clinicals[0]=='cases.0.case_id')[0]

    meta_clinicals = meta_clinicals[1:]

    #############################
    ##### 1. CREATE AS JSON #####
    #############################
    # Empty dictionary for the final data
    new_dict = {}

    # Iterate for each patient's clinical data and take the drugs', radiations', and follow-ups' id
    for meta_clinical in meta_clinicals:
        # find the file id and file name column in your clinical_supplement.csv
        file_id = meta_clinical[file_id_column]
        file_name = meta_clinical[file_name_column] 

        # parse the XML file and take the root
        tree = ET.parse(DATASET_CLINICAL + file_id + "/" + file_name)
        root = tree.getroot()

        new_dict[meta_clinical[case_id_column]] = {'ajcc_stage_version':{}, 'ajcc_stage':{}, 'tnm_stage_t':{}, 'tnm_stage_n':{}, 'tnm_stage_m':{}}

        new_dict[meta_clinical[case_id_column]]['ajcc_stage_version'] = root[1].find('shared_stage:stage_event', ns).find('shared_stage:system_version', ns).text
        new_dict[meta_clinical[case_id_column]]['ajcc_stage'] = root[1].find('shared_stage:stage_event', ns).find('shared_stage:pathologic_stage', ns).text
        new_dict[meta_clinical[case_id_column]]['tnm_stage_t'] = root[1].find('shared_stage:stage_event', ns).find('shared_stage:tnm_categories', ns).find('shared_stage:pathologic_categories', ns).find('shared_stage:pathologic_T', ns).text
        new_dict[meta_clinical[case_id_column]]['tnm_stage_n'] = root[1].find('shared_stage:stage_event', ns).find('shared_stage:tnm_categories', ns).find('shared_stage:pathologic_categories', ns).find('shared_stage:pathologic_N', ns).text
        new_dict[meta_clinical[case_id_column]]['tnm_stage_m'] = root[1].find('shared_stage:stage_event', ns).find('shared_stage:tnm_categories', ns).find('shared_stage:pathologic_categories', ns).find('shared_stage:pathologic_M', ns).text


    # Save as json
    new_dict_j = json.dumps(new_dict, indent=2)
    with open(TARGET_CLINICAL + "pathology_stage.json", "w") as text_file:
        text_file.write(new_dict_j)

    print("pathology_stage.json is created.")


    #############################
    ####### 2. JSON TO CSV ######
    #############################
    meta_clinicals_case_id = meta_clinicals[:,case_id_column]

    with open(TARGET_CLINICAL + "pathology_stage.json") as file:
        pathology_stage = yaml.safe_load(file)

    with open(TARGET_CLINICAL + 'pathology_stage.csv', 'w') as csvfile:
        fieldnames = ['Case', 'AJCC Stage Version', 'AJCC Stage', 'TNM Stage T', 'TNM Stage N', 'TNM Stage M']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for case in meta_clinicals_case_id:
            row = {}
            row['Case'] = case
            row['AJCC Stage Version'] = pathology_stage[case]['ajcc_stage_version']
            row['AJCC Stage'] = pathology_stage[case]['ajcc_stage']
            row['TNM Stage T'] = pathology_stage[case]['tnm_stage_t']
            row['TNM Stage N'] = pathology_stage[case]['tnm_stage_n']
            row['TNM Stage M'] = pathology_stage[case]['tnm_stage_m']
            
            writer.writerow(row)

    print("pathology_stage.csv is created.")



# Create a file for every patient's surgery status (surgery type, margin status, and etc)
def surgery():
    # Load meta for clinical data
    meta_clinicals = np.genfromtxt(TARGET_META_CSV + "clinical_supplement.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(meta_clinicals[0]=='file_id')[0]
    file_name_column, = np.where(meta_clinicals[0]=='file_name')[0]
    case_id_column, = np.where(meta_clinicals[0]=='cases.0.case_id')[0]

    meta_clinicals = meta_clinicals[1:]

    #############################
    ##### 1. CREATE AS JSON #####
    #############################
    # Empty dictionary for the final data
    new_dict = {}

    # Iterate for each patient's clinical data and take the drugs', radiations', and follow-ups' id
    for meta_clinical in meta_clinicals:
        # find the file id and file name column in your clinical_supplement.csv
        file_id = meta_clinical[file_id_column]
        file_name = meta_clinical[file_name_column] 

        # parse the XML file and take the root
        tree = ET.parse(DATASET_CLINICAL + file_id + "/" + file_name)
        root = tree.getroot()

        new_dict[meta_clinical[case_id_column]] = {'surgery':{}, 'reexcision_surgery':{}, 'margin_status':{}, 'reexcision_margin_status':{}}

        # Surgery
        if (root[1].find("brca:breast_carcinoma_surgical_procedure_name", ns).text == "Other") or (root[1].find("brca:breast_carcinoma_surgical_procedure_name", ns).text == None):
            if root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Wide Excision" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Wide local excision" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Wide Local Excision" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "wide local excision" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "breast conserving therapy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "partial left mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Partial mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Partial Mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "partial mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "needle localized segmental mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Right segmental mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "right segmental mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Segmental Mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Segmental mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "segmental mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "left segmental mastectomy with axillary sentinel lymph node biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "left segmental mastectomy with sentinel node dissection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "segmental mastectomy with right sentinel lymph node biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Segmental mastectomy with sentinel axillary lymph node biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Segmental mastectomy with sentinel lymph node biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "segmental mastectomy with sentinel lymph node biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "segmental mastectomy with sentinel lymph node biopsy and wire localization" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "segmental mastectomy with sentinel lymph node dissection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "segmental mastectomy with sentinel lymph node excision and biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Segmental mastectomy with sentinel node biopsy":
                new_dict[meta_clinical[case_id_column]]['surgery'] = "Lumpectomy"
            elif root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "needle directed right breast biopsy with left segmental mastectomy with axillary lymph node dissection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Left segmental mastectomy with axillary lymph node dissection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Left segmental mastectomy with left axillary lymph node dissection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "localized segmental mastectomy with axillary node dissection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Right segmental mastectomy with sentinel lymph node and axillary lymph node biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "right segmental mastectomy witrh axillary node dissection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "segmental mastectomy with axillary dissection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "segmental mastectomy with axillary lymph node dissection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "segmental mastectomy with excision of mass on chest wall and axillary node dissection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Segmental mastectomy with left axillary lymph node dissection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Left segmental mastectomy with level 1 and level 2 axillary dissection":
                new_dict[meta_clinical[case_id_column]]['surgery'] = "Lumpectomy with Axillary Lymph Node Dissection"
            elif root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "TOTAL MASTECTOMY" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Total Mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Total mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "total mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Bilateral Mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "L Total Mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Nipple Sparing Total Mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Bilateral skin sparing Mastectomy and Bilateral breast reconstruction with tissue expanders." or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Wide local excision and simple mastectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "bilateral total mastectomies with right sentinel node biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Right Total Mastectomy and Sentinel Node Biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "right total mastectomy with sentinel lymph node dissection right reconstruction with TRAMP" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "total mastectomy and sentinel node biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Total mastectomy with left sentinel lymph node biospy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Total mastectomy with sentinel lymoh node biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Total mastectomy with sentinel lymph node biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "total mastectomy with sentinel lymph node biopsy with tissue reconstruction" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Total mastectomy with retromammary lymph node excision":
                new_dict[meta_clinical[case_id_column]]['surgery'] = "Simple Mastectomy"
            elif root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Modified Radical Masectomy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Modified radical mastectomy with left breast biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Right total mastectomy with lymph node left axillary lymph node excision" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "total mastectomy with rigth axillary lymph node and sentinel lymph node dissection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Patey's Surgery" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Pateys surgery":
                new_dict[meta_clinical[case_id_column]]['surgery'] = "Modified Radical Mastectomy"
            elif root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "SKIN SPARING RADICAL MASTECTOMY":
                new_dict[meta_clinical[case_id_column]]['surgery'] = "Radical Mastectomy"
            elif root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Surgical resection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Surgical Resection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "surgical resection" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Excision" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "EXCISION WITH NEEDLE WIRE LOCALIZATION" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Reexc of biopsy site for gross/micro residual disease":
                new_dict[meta_clinical[case_id_column]]['surgery'] = "Surgery NOS"
            elif root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Fine Needle aspiration biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Fine needle aspiration biopsy":
                new_dict[meta_clinical[case_id_column]]['surgery'] = "Fine Needle Aspiration Biopsy"
            elif root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Excisional biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "excisional biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Excisional biospy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "Wide re-excisional biopsy" or \
                root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == "biopsy":
                new_dict[meta_clinical[case_id_column]]['surgery'] = "Excisional Biopsy"
            elif root[1].find("brca:surgical_procedure_purpose_other_text", ns).text == None:
                new_dict[meta_clinical[case_id_column]]['surgery'] = None
        else:
            new_dict[meta_clinical[case_id_column]]['surgery'] = root[1].find("brca:breast_carcinoma_surgical_procedure_name", ns).text


        # Reexcision surgery
        if root[1].find("brca:breast_carcinoma_primary_surgical_procedure_name", ns).text == "Other":
            if root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "Right Breast reexcision" or \
                root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "Right Breast Reexcision" or \
                root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "Left breast reexcision" or \
                root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "Reexcision" or \
                root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "Re-Excision of Superior Margin" or \
                root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "inner-upper margin re-excision" or \
                root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "Re-excision of the inferior margin" or \
                root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "Margin resection" or \
                root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "Additional resection/ margins (taken after first margin positive)" or \
                root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "surgical resection" or \
                root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "Reexcision of segmental mastectomy" or \
                root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "Re-excision of original lumpectomy site" or \
                root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "Skin excision":
                new_dict[meta_clinical[case_id_column]]['reexcision_surgery'] = "Reexcision NOS"
            elif root[1].find("brca:breast_neoplasm_other_surgical_procedure_descriptive_text", ns).text == "Mastectomy":
                new_dict[meta_clinical[case_id_column]]['reexcision_surgery'] = "Mastectomy NOS"
        else:
            new_dict[meta_clinical[case_id_column]]['reexcision_surgery'] = root[1].find("brca:breast_carcinoma_primary_surgical_procedure_name", ns).text


        # Margin status
        new_dict[meta_clinical[case_id_column]]['margin_status'] = root[1].find('clin_shared:margin_status', ns).text
        new_dict[meta_clinical[case_id_column]]['reexcision_margin_status'] = root[1].find('brca:breast_cancer_surgery_margin_status', ns).text


    # Save as json
    new_dict_j = json.dumps(new_dict, indent=2)
    with open(TARGET_CLINICAL + "surgery.json", "w") as text_file:
        text_file.write(new_dict_j)

    print("surgery.json is created.")


    #############################
    ####### 2. JSON TO CSV ######
    #############################
    meta_clinicals_case_id = meta_clinicals[:,case_id_column]

    with open(TARGET_CLINICAL + "surgery.json") as file:
        surgery = yaml.safe_load(file)

    with open(TARGET_CLINICAL + 'surgery.csv', 'w') as csvfile:
        fieldnames = ['Case', 'Surgery', 'Reexcision Surgery', 'Margin Status', 'Reexcision Margin Status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for case in meta_clinicals_case_id:
            row = {}
            row['Case'] = case
            row['Surgery'] = surgery[case]['surgery']
            row['Reexcision Surgery'] = surgery[case]['reexcision_surgery']
            row['Margin Status'] = surgery[case]['margin_status']
            row['Reexcision Margin Status'] = surgery[case]['reexcision_margin_status']
            
            writer.writerow(row)

    print("surgery.csv is created.")



# Create a file for every patient's drugs information (drugs name, therapy type, and etc)
def drugs():
    # Load meta for clinical data
    meta_clinicals = np.genfromtxt(TARGET_META_CSV + "clinical_supplement.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(meta_clinicals[0]=='file_id')[0]
    file_name_column, = np.where(meta_clinicals[0]=='file_name')[0]
    case_id_column, = np.where(meta_clinicals[0]=='cases.0.case_id')[0]

    meta_clinicals = meta_clinicals[1:]

    #############################
    ##### 1. CREATE AS JSON #####
    #############################
    # Empty dictionary for the final data
    new_dict = {}

    # Iterate for each patient's clinical data and take the drugs', radiations', and follow-ups' id
    for meta_clinical in meta_clinicals:
        # find the file id and file name column in your clinical_supplement.csv
        file_id = meta_clinical[file_id_column]
        file_name = meta_clinical[file_name_column] 

        # parse the XML file and take the root
        tree = ET.parse(DATASET_CLINICAL + file_id + "/" + file_name)
        root = tree.getroot()

        new_dict[meta_clinical[case_id_column]] = {}

        for drug in root[1].find("rx:drugs", ns):
            new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text] = {'name':{}, 'therapy_type':{}, 'regimen_indication':{}, 'response':{}}

            # Drugs name
            name = drug.find("rx:drug_name", ns).text
            if name=="abraxane" or name=="Abraxane" or name=="Albumin-Bound Paclitaxel" or name=="Paclitaxel (Protein-Bound)":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Abraxane/Albumin-bound-Paclitaxel/Protein-bound-Paclitaxel"
            elif name=="AE-37":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "AE-37"
            elif name=="Aloxi":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Aloxi"
            elif name=="Anastrazole" or name=="Anastrozole" or name=="ANASTROZOLE" or name=="ANASTROZOLE (ARIMIDEX)" or name=="anastrozolum" or name=="arimidex" or name=="Arimidex" or name=="ARIMIDEX" or name=="ARIMIDEX (ANASTROZOLE)" or name=="Arimidex (Anastrozole)":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Anastrozole/Arimidex"
            elif name=="Avastin" or name=="avastin" or name=="Bevacizumab" or name=="BEVACIZUMAB" or name=="BEVACIZUMAB (AVASTIN)/PLACEBO PROVIDED BY STUDY" or name=="Bevacizumab or Placebo":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Bevacizumab/Avastin"
            elif name=="Capecetabine" or name=="CAPECITABINE" or name=="Xeloda" or name=="XELODA" or name=="Xeloda (Capecitabine)":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Capecitabine/Xeloda"
            elif name=="CARBOPLATIN" or name=="Carboplatin":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Carboplatin"
            elif name=="Chemo, NOS":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Chemo, NOS"
            elif name=="Cisplatin":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Cisplatin"
            elif name=="clodronate" or name=="Clodronate" or name=="clodronic acid":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Clodronic acid"
            elif name=="Cyclophasphamide" or name=="Cyclophospamide" or name=="cyclophosphamid" or name=="cyclophosphamide" or name=="Cyclophosphamide" or name=="CYCLOPHOSPHAMIDE" or name=="cyclophosphamidum" or name=="Cyclophosphane" or name=="Cyotxan" or name=="Cytoxan" or name=="cytoxan" or name=="CYTOXAN" or name=="Cytoxen":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Cyclophosphamide/Cytoxan"
            elif name=="Denosumab" or name=="Xgeva":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Denosumab/Xgeva"
            elif name=="Docetaxel" or name=="docetaxel" or name=="DOCETAXEL" or name=="Doxetaxel" or name=="TAXOTERE" or name=="Taxotere" or name=="taxotere":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Docetaxel/Taxotere"
            elif name=="adriamicin" or name=="ADRIAMYCIN" or name=="adriamycin" or name=="Adriamycin" or name=="Adriamyicin" or name=="Adrimycin" or name=="Doxorubicin" or name=="doxorubicin" or name=="DOXORUBICIN" or name=="doxorubicin HCL" or name=="doxorubicine" or name=="Doxorubicinum":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Doxorubicin/Adriamycin"
            elif name=="Doxil" or name=="Doxorubicin Liposome":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Doxorubicin-Liposome/Doxil"
            elif name=="Epirubicin" or name=="Epirubicoin":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Epirubicin"
            elif name=="Everolimus":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Everolimus"
            elif name=="aromasin" or name=="Aromasin" or name=="Aromasin (Exemestane)" or name=="AROMASIN (EXEMESTANE)" or name=="aromatase exemestane" or name=="EXEMESTANE" or name=="Exemestane" or name=="EXEMESTANE (AROMASIN)":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Exemestane/Aromasin"
            elif name=="5 fluorouracil" or name=="5-Flourouracil" or name=="5-Fluorouracil" or name=="5-FU" or name=="FLOUROURACIL" or name=="fluorouracil" or name=="Fluorouracil" or name=="FLUOROURACIL":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Fluorouracil"
            elif name=="Faslodex" or name=="faslodex" or name=="Fulvestrant" or name=="FULVESTRANT" or name=="Fulvestrant (Faslodex)":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Fulvestrant/Faslodex"
            elif name=="gemcitabine" or name=="Gemcitabine" or name=="GEMZAR" or name=="Gemzar":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Gemcitabine/Gemzar"
            elif name=="Goserelin" or name=="Zoladex":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Goserelin/Zoladex"
            elif name=="Hormone, NOS":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Hormone, NOS"
            elif name=="Ibandronate":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Ibandronate"
            elif name=="Ifosfamide":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Ifosfamide"
            elif name=="Ixabepilone":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Ixabepilone"
            elif name=="Lapatinib" or name=="lapatinib":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Lapatinib/Tykerb"
            elif name=="FEMARA" or name=="femara" or name=="Femara" or name=="Femara (Letrozole)" or name=="Letrozol" or name=="letrozole" or name=="LETROZOLE" or name=="Letrozole" or name=="LETROZOLE (FEMARA)" or name=="Letrozole (Femara)" or name=="letrozolum":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Letrozole/Femara"
            elif name=="Leuprolide" or name=="LEUPROLIDE ACETATE (LUPRON)" or name=="Leuprorelin" or name=="Lupron" or name=="LUPRON":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Leuprorelin/Lupron"
            elif name=="Megace":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Megace"
            elif name=="MESNA-1" or name=="MESNA-2":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Mesna"
            elif name=="Metformin":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Metformin"
            elif name=="METHOTREXATE" or name=="Methotrexate" or name=="metotreksat":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Methotrexate"
            elif name=="Mitomycin":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Mitomycin C"
            elif name=="Mitoxantrone":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Mitoxantrone"
            elif name=="NEULASTA" or name=="Neulasta":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Neulasta"
            elif name=="E-75":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Neuvax/E-75"
            elif name=="Not otherwise specified":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "NOS"
            elif name=="paclitaxel" or name=="PACLITAXEL" or name=="Paclitaxel" or name=="paclitaxelum" or name=="Taxane" or name=="TAXOL" or name=="Taxol" or name=="taxol":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Paclitaxel/Taxol"
            elif name=="Pamidronate" or name=="Pamidronic acid":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Pamidronate"
            elif name=="Pemetrexed":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Pemetrexed"
            elif name=="Poly E":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Poly E"
            elif name=="Prednisone":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Prednisone"
            elif name=="Rituximab":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Rituximab"
            elif name=="Nolvadex" or name=="nolvadex" or name=="tamoxifen" or name=="TAMOXIFEN" or name=="Tamoxifen" or name=="TAMOXIFEN (NOVADEX)" or name=="tamoxifen citrate" or name=="tamoxiphene":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Tamoxifen/Nolvadex"
            elif name=="Tesetaxel":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Tesetaxel"
            elif name=="Fareston":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Toremifene/Fareston"
            elif name=="herceptin" or name=="HERCEPTIN" or name=="Herceptin" or name=="trastuzumab" or name=="Trastuzumab" or name=="TRASTUZUMAB" or name=="Trustuzumab":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Trastuzumab/Herceptin"
            elif name=="Triptorelin" or name=="triptorelin":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Triptorelin"
            elif name=="Vinblastine":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Vinblastine"
            elif name=="Vincristine":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Vincristine"
            elif name=="NAVELBINE" or name=="Navelbine" or name=="Vinorelbine":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Vinorelbine/Navelbine"
            elif name=="VP-16":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "VP-16"
            elif name=="Yondelis":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Trabectedin/Yondelis"
            elif name=="ZOLEDRONIC ACID" or name=="Zoledronic Acid" or name=="Zoledronic acid" or name=="zoledronic acid" or name=="Zometa":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Zoledronic-acid/Zometa"
            elif name=="ac" or name=="adriamycin+cuclophosphamide" or name=="adriamycin+cyclophosphamid" or name=="adriamycin+cyclophosphamide" or name=="adrimicin+cyclophosphamide" or name=="adrimycin+cyclophosphamide" or name=="doxorubicin+ cyclophosphamide" or name=="doxorubicin+cyclophosphamid" or name=="doxorubicine+cyclophosphamide":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Doxorubicin/Adriamycin + Cyclophosphamide/Cytoxan (AC)"
            elif name=="Adriamycin, cytoxan, avastin":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Doxorubicin/Adriamycin + Cyclophosphamide/Cytoxan (AC) + Bevacizumab/Avastin"
            elif name=="doxorubicine cyclophosphamide tamoxifen" or name=="doxorubicine+cyclophosphamide+tamoxifen":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Doxorubicin/Adriamycin + Cyclophosphamide/Cytoxan (AC) + Tamoxifen/Nolvadex"
            elif name=="taxol+adriamycin+cyclophosphamide+herceptin":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Doxorubicin/Adriamycin + Cyclophosphamide/Cytoxan (AC) + Paclitaxel/Taxol + Trastuzumab/Herceptin"
            elif name=="cyclophosphamide+methotrexatum+fluorouracillum" or name=="methotrexate+5 fluorouracil+cyclophosphamide":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Cyclophosphamide/Cytoxan + Methotrexate + Fluorouracil (CMF)"
            elif name=="tamoxiphen+anastrazolum" or name=="tamoxiphene+anastrozolum":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Tamoxifen/Nolvadex + Anastrozole/Arimidex"
            elif name=="tamoxiphene+leuporeline+gosereline":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Tamoxifen/Nolvadex + Leuprorelin/Lupron + Goserelin/Zoladex"
            elif name=="TCH":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Docetaxel/Taxotere + Carboplatin + Trastuzumab/Herceptin (TCH)"
            elif name=="Cytoxan and Taxotere" or name=="Taxotere/Cytoxan" or name=="tc" or name=="TC":
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = "Docetaxel/Taxotere + Cyclophosphamide/Cytoxan (TC)"
            elif name==None:
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['name'] = None


            # Therapy type
            t1 = drug.find("rx:therapy_types", ns).find("rx:therapy_type", ns).text
            t2 = drug.find("rx:therapy_types", ns).find("rx:therapy_type_notes", ns).text
            if t1 == "Other, specify in notes":
                if t2=="ancillary" or t2=="Bisphosphonate" or t2=="biphosphonate" or t2=="BISPHOSPHONATE" or t2=="Bisphosphonate therapy" or t2=="clinical trial - bisphosphonates as adjuvant therapy" or t2=="Bone metastases" or t2=="Given to induce menopause":
                    new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['therapy_type'] = "Ancillary"
                elif t2=="Aromatase Inhibitor":
                    new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['therapy_type'] = "Hormone Therapy"
                elif t2==None or t2=="Phase III Clinical Trial" or t2=="Phase III clinical trial":
                    new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['therapy_type'] = None
            else:
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['therapy_type'] = t1


            # Regimen Indication
            r1 = drug.find("clin_shared:regimen_indication", ns).text
            r2 = drug.find("clin_shared:regimen_indication_notes", ns).text
            if r1 == "OTHER, SPECIFY IN NOTES":
                if r2=="Patient has oesteoporosis (Prevention of further bone loss" or r2=="Maintenance (for osteopenia)" or r2=="Maintenance therapy":
                    new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['regimen_indication'] = "PALLIATIVE"
                elif r2==None or r2=="Given to induce menopause" or r2=="Estrogen receptor antagonist in metastatic breast cancer":
                    new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['regimen_indication'] = None
                elif r2=="Neo-Adjuvant" or r2=="Neo-adjuvant" or r2=="neoadjuvant":
                    new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['regimen_indication'] = "NEO-ADJUVANT"
                elif r2=="Cancer Vaccine Trial" or r2=="Preventative":
                    new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['regimen_indication'] = "PREVENTIVE"
            else:
                new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['regimen_indication'] = r1


            # Response
            new_dict[meta_clinical[case_id_column]][drug.find("rx:bcr_drug_uuid", ns).text]['response'] = drug.find("clin_shared:measure_of_response", ns).text


    # Save as json
    new_dict_j = json.dumps(new_dict, indent=2)
    with open(TARGET_CLINICAL + "drugs.json", "w") as text_file:
        text_file.write(new_dict_j)

    print(TARGET_CLINICAL + "drugs.json is created.")


    #############################
    ####### 2. JSON TO CSV ######
    #############################
    meta_clinicals_case_id = meta_clinicals[:,case_id_column]

    with open(TARGET_CLINICAL + "uuid.json") as file:
        uuid = yaml.safe_load(file)

    with open(TARGET_CLINICAL + "drugs.json") as file:
        drugs = yaml.safe_load(file)

    with open(TARGET_CLINICAL + 'drugs.csv', 'w') as csvfile:
        fieldnames = ['Case', 'Drug UUID', 'Drug Name', 'Therapy Type', 'Regimen Indication', 'Response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for case in meta_clinicals_case_id:
            for drug in uuid[case]['drugs']:
                row = {}
                row['Case'] = case
                row['Drug UUID'] = drug
                row['Drug Name'] = drugs[case][drug]['name']
                row['Therapy Type'] = drugs[case][drug]['therapy_type']
                row['Regimen Indication'] = drugs[case][drug]['regimen_indication']
                row['Response'] = drugs[case][drug]['response']

                writer.writerow(row)

    print("drugs.csv is created.")

    # from all 2408 drugs treatment, only 645 have response status



# Create a file for every patient's radiations information (type, treatment site, response, and etc)
def radiations():
    # Load meta for clinical data
    meta_clinicals = np.genfromtxt(TARGET_META_CSV + "clinical_supplement.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(meta_clinicals[0]=='file_id')[0]
    file_name_column, = np.where(meta_clinicals[0]=='file_name')[0]
    case_id_column, = np.where(meta_clinicals[0]=='cases.0.case_id')[0]

    meta_clinicals = meta_clinicals[1:]

    #############################
    ##### 1. CREATE AS JSON #####
    #############################
    # Empty dictionary for the final data
    new_dict = {}

    # Iterate for each patient's clinical data and take the drugs', radiations', and follow-ups' id
    for meta_clinical in meta_clinicals:
        # find the file id and file name column in your clinical_supplement.csv
        file_id = meta_clinical[file_id_column]
        file_name = meta_clinical[file_name_column] 

        # parse the XML file and take the root
        tree = ET.parse(DATASET_CLINICAL + file_id + "/" + file_name)
        root = tree.getroot()

        new_dict[meta_clinical[case_id_column]] = {}

        for radiation in root[1].find("rad:radiations", ns):
            new_dict[meta_clinical[case_id_column]][radiation.find("rad:bcr_radiation_uuid", ns).text] = {'type':{}, 'treatment_site':{}, 'regimen_indication':{}, 'response':{}}

            new_dict[meta_clinical[case_id_column]][radiation.find("rad:bcr_radiation_uuid", ns).text]['type'] = radiation.find("rad:radiation_type", ns).text
            new_dict[meta_clinical[case_id_column]][radiation.find("rad:bcr_radiation_uuid", ns).text]['treatment_site'] = radiation.find("rad:anatomic_treatment_site", ns).text
            new_dict[meta_clinical[case_id_column]][radiation.find("rad:bcr_radiation_uuid", ns).text]['regimen_indication'] = radiation.find("clin_shared:regimen_indication", ns).text
            new_dict[meta_clinical[case_id_column]][radiation.find("rad:bcr_radiation_uuid", ns).text]['response'] = radiation.find("clin_shared:measure_of_response", ns).text


    # Save as json
    new_dict_j = json.dumps(new_dict, indent=2)
    with open(TARGET_CLINICAL + "radiations.json", "w") as text_file:
        text_file.write(new_dict_j)

    print("radiations.json is created.")


    #############################
    ####### 2. JSON TO CSV ######
    #############################
    meta_clinicals_case_id = meta_clinicals[:,case_id_column]

    with open(TARGET_CLINICAL + "uuid.json") as file:
        uuid = yaml.safe_load(file)

    with open(TARGET_CLINICAL + "radiations.json") as file:
        radiations = yaml.safe_load(file)

    with open(TARGET_CLINICAL + 'radiations.csv', 'w') as csvfile:
        fieldnames = ['Case', 'Radiation UUID', 'Radiation Type', 'Treatment Site', 'Regimen Indication', 'Response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for case in meta_clinicals_case_id:
            for radiation in uuid[case]['radiations']:
                row = {}
                row['Case'] = case
                row['Radiation UUID'] = radiation
                row['Radiation Type'] = radiations[case][radiation]['type']
                row['Treatment Site'] = radiations[case][radiation]['treatment_site']
                row['Regimen Indication'] = radiations[case][radiation]['regimen_indication']
                row['Response'] = radiations[case][radiation]['response']

                writer.writerow(row)

    print("radiations.csv is created.")

    # from all 619 drugs treatment, only 261 have response status



if __name__ == '__main__':
    all_prefix()
    #compare_elmt()
    #compare_elmt_len()
    uuid()
    form_completion()
    general()
    pathology_general()
    pathology_receptor()
    pathology_lymph()
    pathology_stage()
    surgery()
    drugs()
    radiations()