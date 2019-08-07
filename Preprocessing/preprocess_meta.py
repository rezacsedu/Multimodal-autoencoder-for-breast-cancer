from folder_location import *

import os
import csv
import json
import yaml
import numpy as np
import timeit
import requests



# request payload for cases meta
PAYLOAD_CASES = {
    "filters":{
        "op":"=",
        "content":{
          "field":"project.project_id",
          "value":"TCGA-BRCA"
        }
    },
    "format":"json",
    "fields":"case_id,submitter_id",
    "sort":"case_id:asc",
    "size":"2000"
}

# request payload for files meta
PAYLOAD_FILES = {
    "filters":{
        "op":"=",
        "content":{
            "field":"cases.project.project_id",
            "value":"TCGA-BRCA"
        }
    },
    "format":"json",
    "fields":"cases.case_id,cases.submitter_id,file_id,file_name,data_type,experimental_strategy,cases.samples.sample_type,analysis.workflow_type,cases.samples.portions.analytes.submitter_id",
    "sort":"file_id:asc",
    "size":"30000"
}

PAYLOAD_EACH_DATA_TYPE = {
    "filters":{
        "op":"and",
        "content":[
            {
                "op":"=",
                "content":{
                    "field":"cases.project.project_id",
                    "value":"TCGA-BRCA"
                }
            },
            {
                "op":"=",
                "content":{
                    "field":"data_type",
                    "value":"Aligned Reads"
                }
            }
        ]
    },
    "format":"json",
    "fields":"cases.case_id,cases.submitter_id,file_id,file_name,data_type,experimental_strategy,cases.samples.sample_type,analysis.workflow_type,cases.samples.portions.analytes.submitter_id",
    "sort":"file_id:asc",
    "size":"5000"
}

DATA_FORMATS = (
    "json",
    "csv"
)

DATA_TYPES = (
    "Aligned Reads",
    "Annotated Somatic Mutation",
    "Raw Simple Somatic Mutation",
    "Aggregated Somatic Mutation",
    "Masked Somatic Mutation",
    "Copy Number Segment",
    "Masked Copy Number Segment",
    "Methylation Beta Value",
    "Gene Expression Quantification",
    "miRNA Expression Quantification",
    "Isoform Expression Quantification",
    "Clinical Supplement",
    "Biospecimen Supplement"
)



# Download meta file from GDC API (https://api.gdc.cancer.gov/)
# There are 3 types of meta file that will be downloaded:
#   1. meta cases
#   2. meta files
#   3. meta for each data type (Aligned Reads, Annotated Somatic Mutation, and etc.)
def requests_meta():
    # request meta in several data format
    for format in DATA_FORMATS:
        # Create folder for the download target
        # there are 2 data format that
        if format=="json":
            meta_folder = TARGET_META_JSON
        elif format=="csv":
            meta_folder = TARGET_META_CSV
        
        if not os.path.isdir(meta_folder):
            os.makedirs(meta_folder)


        #############################
        ####### 1. META CASES #######
        #############################
        endpt = 'https://api.gdc.cancer.gov/cases/'
        PAYLOAD_CASES["format"] = format

        response = requests.post(endpt, json=PAYLOAD_CASES)
        if format == "json":
            responseJ = json.dumps(response.json(), indent=2)

        dest = (meta_folder + "cases." + format)
        with open(dest, "w") as text_file:
            if format == "json":
                text_file.write(responseJ)
            else:
                text_file.write(response.text)

        print(dest + " is created.")


        #############################
        ####### 2. META FILES #######
        #############################
        endpt = 'https://api.gdc.cancer.gov/files/'
        PAYLOAD_FILES["format"] = format

        response = requests.post(endpt, json=PAYLOAD_FILES)
        if format == "json":
            responseJ = json.dumps(response.json(), indent=2)

        dest = (meta_folder + "files." + format)
        with open(dest, "w") as text_file:
            if format == "json":
                text_file.write(responseJ)
            else:
                text_file.write(response.text)

        print(dest + " is created.")


        #############################
        ### 3. META EACH DATA TYPE ##
        ############################# 
        endpt = 'https://api.gdc.cancer.gov/files/'
        PAYLOAD_EACH_DATA_TYPE["format"] = format

        # request meta for each data type
        for data_type in DATA_TYPES:
            PAYLOAD_EACH_DATA_TYPE["filters"]["content"][1]["content"]["value"] = data_type
            response = requests.post(endpt, json=PAYLOAD_EACH_DATA_TYPE)
            if format == "json":
                responseJ = json.dumps(response.json(), indent=2)

            dest = (meta_folder + data_type.lower().replace(' ', '_') + "." + format)
            with open(dest, "w") as text_file:
                if format == "json":
                    text_file.write(responseJ)
                else:
                    text_file.write(response.text)

            print(dest + " is created.")



# Combine meta of all files based on case UUID
def meta_per_case(): 
    # Get list of cases based on its id
    cases = np.genfromtxt(TARGET_META_CSV + "cases.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your cases.csv
    case_id_column, = np.where(cases[0]=='case_id')[0]
    
    cases = cases[1:]
    
    cases_id = cases[:,case_id_column]

    # Create dictionary for the final file
    new_dict = {}

    for case_id in cases_id:
        new_dict[case_id] = {}
        for data_type in DATA_TYPES:
            new_dict[case_id][data_type] = []

    
    for data_type in DATA_TYPES:
        with open(TARGET_META_JSON + data_type.lower().replace(' ', '_') + ".json") as file:
            temp_source = yaml.safe_load(file)

        meta_per_type = temp_source["data"]["hits"]

        for i in range(len(meta_per_type)):
            file_id = meta_per_type[i]["file_id"]
            file_name = meta_per_type[i]["file_name"]
            data_type = meta_per_type[i]["data_type"]
            if "experimental_strategy" in meta_per_type[i]:
                experimental_strategy = meta_per_type[i]["experimental_strategy"]
            else:
                experimental_strategy = ""
            if "analysis" in meta_per_type[i]:
                workflow = meta_per_type[i]["analysis"]["workflow_type"]
            else:
                workflow = ""

            for j in range(len(meta_per_type[i]["cases"])):
                new_dict[meta_per_type[i]["cases"][j]["case_id"]][data_type].append({"file_id":file_id, "file_name":file_name})
                
                if experimental_strategy:
                    new_dict[meta_per_type[i]["cases"][j]["case_id"]][data_type][-1]["experimental_strategy"] = experimental_strategy

                if workflow:
                    new_dict[meta_per_type[i]["cases"][j]["case_id"]][data_type][-1]["workflow"] = workflow
                
                if "samples" in meta_per_type[i]["cases"][j]:
                    new_dict[meta_per_type[i]["cases"][j]["case_id"]][data_type][-1]["samples"] = []
                    
                    for k in range(len(meta_per_type[i]["cases"][j]["samples"])):
                        sample_type = meta_per_type[i]["cases"][j]["samples"][k]["sample_type"]
                        new_dict[meta_per_type[i]["cases"][j]["case_id"]][data_type][-1]["samples"].append({"sample_type":sample_type, "analytes":[]})

                        for l in range(len(meta_per_type[i]["cases"][j]["samples"][k]["portions"])):
                            for m in range(len(meta_per_type[i]["cases"][j]["samples"][k]["portions"][l]["analytes"])):
                                analytes = meta_per_type[i]["cases"][j]["samples"][k]["portions"][l]["analytes"][m]["submitter_id"]
                                new_dict[meta_per_type[i]["cases"][j]["case_id"]][data_type][-1]["samples"][-1]["analytes"].append(analytes)

    new_dict_j = json.dumps(new_dict, indent=2)
    with open(TARGET_META_JSON + "meta.json", "w") as text_file:
        text_file.write(new_dict_j)

    print("meta.json is created.")



# Count the amount of files per case based on the data type
def file_amount():
    # Create list of cases id
    cases = np.genfromtxt(TARGET_META_CSV + "cases.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your cases.csv
    case_id_column, = np.where(cases[0]=='case_id')[0]

    cases = cases[1:]
    
    cases_id = cases[:,case_id_column]

    # Load meta.json
    with open(TARGET_META_JSON + "meta.json") as file:
        meta = yaml.safe_load(file)
    

    # create file_amount.csv
    with open(TARGET_META_CSV + 'file_amount.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=(("Cases",) + DATA_TYPES))

        writer.writeheader()
        for case_id in cases_id:
            row = {}
            row["Cases"] = case_id
            
            for type in DATA_TYPES:
                row[type] = len(meta[case_id][type])

            writer.writerow(row)

    print("file_amount.csv is created.")



# Create dictionary that translate from patients' submitter id to their case uuid
def submitter_id_to_case_uuid():
    # Load cases.csv
    cases = np.genfromtxt(TARGET_META_CSV + "cases.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your cases.csv
    case_id_column, = np.where(cases[0]=='case_id')[0]
    submitter_id_column, = np.where(cases[0]=='submitter_id')[0]

    cases = cases[1:]
    
    # create dictionary from submitter_id to case_uuid
    new_dict = {}
    for i in range(len(cases)):
        # find where the case id column is located in your cases.csv
        case_id = cases[i,case_id_column]
        submitter_id = cases[i,submitter_id_column]
        
        new_dict[submitter_id] = case_id
    
    # save as json 
    new_dict_j = json.dumps(new_dict, indent=2)
    with open(TARGET_META_JSON + "submitter_id_to_case_uuid.json", "w") as text_file:
        text_file.write(new_dict_j)

    print("submitter_id_to_case_uuid.json is created.")



if __name__ == '__main__':
    start = timeit.default_timer()
    requests_meta()
    meta_per_case()
    file_amount()
    submitter_id_to_case_uuid()
    stop = timeit.default_timer()
    print stop - start
