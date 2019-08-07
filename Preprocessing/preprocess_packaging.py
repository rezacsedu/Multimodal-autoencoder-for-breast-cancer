from folder_location import *

import os
import csv
import json
import yaml
import numpy as np
import timeit
import gzip
import numpy as np



# Create the label set of cancer type classification
# We use the pathological receptor (ER, PGR, HER2/neu) status (positive, negative, indeterminate)
def label_cancer_type(dataset=3):
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_all = cases[:,0]                  # [1098,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    
    if (dataset==1) or (dataset==5):
        cases_met = cases[cases[:,8]!="0",0]    # [1095,]
        with open(TARGET_METHYLATION + "cases_met_long.json") as f:
            temp = yaml.safe_load(f)
        cases_metlong = np.asarray(temp)  # [782,]
        cases_met_cli = np.intersect1d(cases_met,cases_cli) # [1095,]
        cases_metlong_cli = np.intersect1d(cases_metlong,cases_cli) # [782,]
    
    if (dataset==2) or (dataset==4) or (dataset==5):
        cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
        # remove cases in cases_gen where there are no tumor sample
        cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
        cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    
    if (dataset==3) or (dataset==4) or (dataset==5):
        cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
        # remove cases in cases_mir where there are no tumor sample
        cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
        cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    
    if (dataset==4) or (dataset==5):
        cases_gen_mir_cli = np.intersect1d(cases_gen_cli,cases_mir) # [1071,]
    
    if (dataset==5):
        cases_met_gen_mir_cli = np.intersect1d(np.intersect1d(cases_met_cli,cases_gen),cases_mir) # [1069,]
        cases_metlong_gen_mir_cli = np.intersect1d(np.intersect1d(cases_metlong_cli,cases_gen),cases_mir) # [768,]


    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_pat_rec = np.genfromtxt(TARGET_CLINICAL + "pathology_receptor.csv", dtype=str, delimiter=',', skip_header=1)
    pat_rec = temp_pat_rec[temp_pat_rec[:,0].argsort()]

    cases_no_er_null = pat_rec[pat_rec[:,2]!="",0]     # [1048,]
    cases_no_pgr_null = pat_rec[pat_rec[:,4]!="",0]    # [1047,]
    cases_no_her2_null = pat_rec[pat_rec[:,7]!="",0]   # [919,]
    cases_no_null = np.intersect1d(np.intersect1d(cases_no_er_null,cases_no_pgr_null),cases_no_her2_null)   # [917,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    if (dataset==1) or (dataset==5):
        cases_met_no_er_null = np.intersect1d(cases_met_cli,cases_no_er_null)       # (1046,)
        cases_met_no_pgr_null = np.intersect1d(cases_met_cli,cases_no_pgr_null)     # (1045,)
        cases_met_no_her2_null = np.intersect1d(cases_met_cli,cases_no_her2_null)   # (917,)
        cases_met_no_null = np.intersect1d(cases_met_cli,cases_no_null)             # (915,)

        cases_metlong_no_er_null = np.intersect1d(cases_metlong_cli,cases_no_er_null)       # (738,)
        cases_metlong_no_pgr_null = np.intersect1d(cases_metlong_cli,cases_no_pgr_null)     # (737,)
        cases_metlong_no_her2_null = np.intersect1d(cases_metlong_cli,cases_no_her2_null)   # (642,)
        cases_metlong_no_null = np.intersect1d(cases_metlong_cli,cases_no_null)             # (640,)

    if (dataset==2) or (dataset==4) or (dataset==5):
        cases_gen_no_er_null = np.intersect1d(cases_gen_cli,cases_no_er_null)       # (1042,)
        cases_gen_no_pgr_null = np.intersect1d(cases_gen_cli,cases_no_pgr_null)     # (1041,)
        cases_gen_no_her2_null = np.intersect1d(cases_gen_cli,cases_no_her2_null)   # (913,)
        cases_gen_no_null = np.intersect1d(cases_gen_cli,cases_no_null)             # (912,)
    
    if (dataset==3) or (dataset==4) or (dataset==5):
        cases_mir_no_er_null = np.intersect1d(cases_mir_cli,cases_no_er_null)       # (1029,)
        cases_mir_no_pgr_null = np.intersect1d(cases_mir_cli,cases_no_pgr_null)     # (1028,)
        cases_mir_no_her2_null = np.intersect1d(cases_mir_cli,cases_no_her2_null)   # (902,)
        cases_mir_no_null = np.intersect1d(cases_mir_cli,cases_no_null)             # (900,)
    
    if (dataset==4) or (dataset==5):
        cases_gen_mir_no_er_null = np.intersect1d(cases_gen_mir_cli,cases_no_er_null)       # (1024,)
        cases_gen_mir_no_pgr_null = np.intersect1d(cases_gen_mir_cli,cases_no_pgr_null)     # (1023,)
        cases_gen_mir_no_her2_null = np.intersect1d(cases_gen_mir_cli,cases_no_her2_null)   # (897,)
        cases_gen_mir_no_null = np.intersect1d(cases_gen_mir_cli,cases_no_null)             # (896,)
    
    if (dataset==5):
        cases_met_gen_mir_no_er_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_er_null)       # (1022,)
        cases_met_gen_mir_no_pgr_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_pgr_null)     # (1021,)
        cases_met_gen_mir_no_her2_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_her2_null)   # (895,)
        cases_met_gen_mir_no_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_null)             # (894,)
        
        cases_metlong_gen_mir_no_er_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_er_null)       # (726,)
        cases_metlong_gen_mir_no_pgr_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_pgr_null)     # (725,)
        cases_metlong_gen_mir_no_her2_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_her2_null)   # (632,)
        cases_metlong_gen_mir_no_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_null)             # (631,)

    
    ######################################
    #### LABELS OF TYPE CLASSIFICATION ###
    ######################################
    if (dataset==1) or (dataset==5):
        # 1. Methylation
        label_er_met_no_er_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_met_no_er_null]),2]
        label_er_met_no_er_null = label_er_met_no_er_null.tolist()
        label_er_met_no_er_null = [0 if x=='Positive' else x for x in label_er_met_no_er_null]
        label_er_met_no_er_null = [1 if x=='Negative' else x for x in label_er_met_no_er_null]
        label_er_met_no_er_null = [2 if x=='Indeterminate' else x for x in label_er_met_no_er_null]
        label_er_met_no_er_null = np.asarray(label_er_met_no_er_null)
        
        label_pgr_met_no_pgr_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_met_no_pgr_null]),4]
        label_pgr_met_no_pgr_null = label_pgr_met_no_pgr_null.tolist()
        label_pgr_met_no_pgr_null = [0 if x=='Positive' else x for x in label_pgr_met_no_pgr_null]
        label_pgr_met_no_pgr_null = [1 if x=='Negative' else x for x in label_pgr_met_no_pgr_null]
        label_pgr_met_no_pgr_null = [2 if x=='Indeterminate' else x for x in label_pgr_met_no_pgr_null]
        label_pgr_met_no_pgr_null = np.asarray(label_pgr_met_no_pgr_null)
        
        label_her2_met_no_her2_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_met_no_her2_null]),7]
        label_her2_met_no_her2_null = label_her2_met_no_her2_null.tolist()
        label_her2_met_no_her2_null = [0 if x=='Positive' else x for x in label_her2_met_no_her2_null]
        label_her2_met_no_her2_null = [1 if x=='Negative' else x for x in label_her2_met_no_her2_null]
        label_her2_met_no_her2_null = [2 if x=='Indeterminate' else x for x in label_her2_met_no_her2_null]
        label_her2_met_no_her2_null = [3 if x=='Equivocal' else x for x in label_her2_met_no_her2_null]
        label_her2_met_no_her2_null = np.asarray(label_her2_met_no_her2_null)
        
        label_er_met_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_met_no_null]),2]
        label_er_met_no_null = label_er_met_no_null.tolist()
        label_er_met_no_null = [0 if x=='Positive' else x for x in label_er_met_no_null]
        label_er_met_no_null = [1 if x=='Negative' else x for x in label_er_met_no_null]
        label_er_met_no_null = [2 if x=='Indeterminate' else x for x in label_er_met_no_null]
        label_er_met_no_null = np.asarray(label_er_met_no_null)
        
        label_pgr_met_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_met_no_null]),4]
        label_pgr_met_no_null = label_pgr_met_no_null.tolist()
        label_pgr_met_no_null = [0 if x=='Positive' else x for x in label_pgr_met_no_null]
        label_pgr_met_no_null = [1 if x=='Negative' else x for x in label_pgr_met_no_null]
        label_pgr_met_no_null = [2 if x=='Indeterminate' else x for x in label_pgr_met_no_null]
        label_pgr_met_no_null = np.asarray(label_pgr_met_no_null)
        
        label_her2_met_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_met_no_null]),7]
        label_her2_met_no_null = label_her2_met_no_null.tolist()
        label_her2_met_no_null = [0 if x=='Positive' else x for x in label_her2_met_no_null]
        label_her2_met_no_null = [1 if x=='Negative' else x for x in label_her2_met_no_null]
        label_her2_met_no_null = [2 if x=='Indeterminate' else x for x in label_her2_met_no_null]
        label_her2_met_no_null = [3 if x=='Equivocal' else x for x in label_her2_met_no_null]
        label_her2_met_no_null = np.asarray(label_her2_met_no_null)


        # 2. Methylation Long
        label_er_metlong_no_er_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_metlong_no_er_null]),2]
        label_er_metlong_no_er_null = label_er_metlong_no_er_null.tolist()
        label_er_metlong_no_er_null = [0 if x=='Positive' else x for x in label_er_metlong_no_er_null]
        label_er_metlong_no_er_null = [1 if x=='Negative' else x for x in label_er_metlong_no_er_null]
        label_er_metlong_no_er_null = [2 if x=='Indeterminate' else x for x in label_er_metlong_no_er_null]
        label_er_metlong_no_er_null = np.asarray(label_er_metlong_no_er_null)
        
        label_pgr_metlong_no_pgr_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_metlong_no_pgr_null]),4]
        label_pgr_metlong_no_pgr_null = label_pgr_metlong_no_pgr_null.tolist()
        label_pgr_metlong_no_pgr_null = [0 if x=='Positive' else x for x in label_pgr_metlong_no_pgr_null]
        label_pgr_metlong_no_pgr_null = [1 if x=='Negative' else x for x in label_pgr_metlong_no_pgr_null]
        label_pgr_metlong_no_pgr_null = [2 if x=='Indeterminate' else x for x in label_pgr_metlong_no_pgr_null]
        label_pgr_metlong_no_pgr_null = np.asarray(label_pgr_metlong_no_pgr_null)
        
        label_her2_metlong_no_her2_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_metlong_no_her2_null]),7]
        label_her2_metlong_no_her2_null = label_her2_metlong_no_her2_null.tolist()
        label_her2_metlong_no_her2_null = [0 if x=='Positive' else x for x in label_her2_metlong_no_her2_null]
        label_her2_metlong_no_her2_null = [1 if x=='Negative' else x for x in label_her2_metlong_no_her2_null]
        label_her2_metlong_no_her2_null = [2 if x=='Indeterminate' else x for x in label_her2_metlong_no_her2_null]
        label_her2_metlong_no_her2_null = [3 if x=='Equivocal' else x for x in label_her2_metlong_no_her2_null]
        label_her2_metlong_no_her2_null = np.asarray(label_her2_metlong_no_her2_null)
        
        label_er_metlong_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_metlong_no_null]),2]
        label_er_metlong_no_null = label_er_metlong_no_null.tolist()
        label_er_metlong_no_null = [0 if x=='Positive' else x for x in label_er_metlong_no_null]
        label_er_metlong_no_null = [1 if x=='Negative' else x for x in label_er_metlong_no_null]
        label_er_metlong_no_null = [2 if x=='Indeterminate' else x for x in label_er_metlong_no_null]
        label_er_metlong_no_null = np.asarray(label_er_metlong_no_null)
        
        label_pgr_metlong_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_metlong_no_null]),4]
        label_pgr_metlong_no_null = label_pgr_metlong_no_null.tolist()
        label_pgr_metlong_no_null = [0 if x=='Positive' else x for x in label_pgr_metlong_no_null]
        label_pgr_metlong_no_null = [1 if x=='Negative' else x for x in label_pgr_metlong_no_null]
        label_pgr_metlong_no_null = [2 if x=='Indeterminate' else x for x in label_pgr_metlong_no_null]
        label_pgr_metlong_no_null = np.asarray(label_pgr_metlong_no_null)
        
        label_her2_metlong_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_metlong_no_null]),7]
        label_her2_metlong_no_null = label_her2_metlong_no_null.tolist()
        label_her2_metlong_no_null = [0 if x=='Positive' else x for x in label_her2_metlong_no_null]
        label_her2_metlong_no_null = [1 if x=='Negative' else x for x in label_her2_metlong_no_null]
        label_her2_metlong_no_null = [2 if x=='Indeterminate' else x for x in label_her2_metlong_no_null]
        label_her2_metlong_no_null = [3 if x=='Equivocal' else x for x in label_her2_metlong_no_null]
        label_her2_metlong_no_null = np.asarray(label_her2_metlong_no_null)
    

    if (dataset==2) or (dataset==4) or (dataset==5):
        # 3. Gene
        label_er_gen_no_er_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_gen_no_er_null]),2]
        label_er_gen_no_er_null = label_er_gen_no_er_null.tolist()
        label_er_gen_no_er_null = [0 if x=='Positive' else x for x in label_er_gen_no_er_null]
        label_er_gen_no_er_null = [1 if x=='Negative' else x for x in label_er_gen_no_er_null]
        label_er_gen_no_er_null = [2 if x=='Indeterminate' else x for x in label_er_gen_no_er_null]
        label_er_gen_no_er_null = np.asarray(label_er_gen_no_er_null)
        
        label_pgr_gen_no_pgr_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_gen_no_pgr_null]),4]
        label_pgr_gen_no_pgr_null = label_pgr_gen_no_pgr_null.tolist()
        label_pgr_gen_no_pgr_null = [0 if x=='Positive' else x for x in label_pgr_gen_no_pgr_null]
        label_pgr_gen_no_pgr_null = [1 if x=='Negative' else x for x in label_pgr_gen_no_pgr_null]
        label_pgr_gen_no_pgr_null = [2 if x=='Indeterminate' else x for x in label_pgr_gen_no_pgr_null]
        label_pgr_gen_no_pgr_null = np.asarray(label_pgr_gen_no_pgr_null)
        
        label_her2_gen_no_her2_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_gen_no_her2_null]),7]
        label_her2_gen_no_her2_null = label_her2_gen_no_her2_null.tolist()
        label_her2_gen_no_her2_null = [0 if x=='Positive' else x for x in label_her2_gen_no_her2_null]
        label_her2_gen_no_her2_null = [1 if x=='Negative' else x for x in label_her2_gen_no_her2_null]
        label_her2_gen_no_her2_null = [2 if x=='Indeterminate' else x for x in label_her2_gen_no_her2_null]
        label_her2_gen_no_her2_null = [3 if x=='Equivocal' else x for x in label_her2_gen_no_her2_null]
        label_her2_gen_no_her2_null = np.asarray(label_her2_gen_no_her2_null)
        
        label_er_gen_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_gen_no_null]),2]
        label_er_gen_no_null = label_er_gen_no_null.tolist()
        label_er_gen_no_null = [0 if x=='Positive' else x for x in label_er_gen_no_null]
        label_er_gen_no_null = [1 if x=='Negative' else x for x in label_er_gen_no_null]
        label_er_gen_no_null = [2 if x=='Indeterminate' else x for x in label_er_gen_no_null]
        label_er_gen_no_null = np.asarray(label_er_gen_no_null)
        
        label_pgr_gen_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_gen_no_null]),4]
        label_pgr_gen_no_null = label_pgr_gen_no_null.tolist()
        label_pgr_gen_no_null = [0 if x=='Positive' else x for x in label_pgr_gen_no_null]
        label_pgr_gen_no_null = [1 if x=='Negative' else x for x in label_pgr_gen_no_null]
        label_pgr_gen_no_null = [2 if x=='Indeterminate' else x for x in label_pgr_gen_no_null]
        label_pgr_gen_no_null = np.asarray(label_pgr_gen_no_null)
        
        label_her2_gen_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_gen_no_null]),7]
        label_her2_gen_no_null = label_her2_gen_no_null.tolist()
        label_her2_gen_no_null = [0 if x=='Positive' else x for x in label_her2_gen_no_null]
        label_her2_gen_no_null = [1 if x=='Negative' else x for x in label_her2_gen_no_null]
        label_her2_gen_no_null = [2 if x=='Indeterminate' else x for x in label_her2_gen_no_null]
        label_her2_gen_no_null = [3 if x=='Equivocal' else x for x in label_her2_gen_no_null]
        label_her2_gen_no_null = np.asarray(label_her2_gen_no_null)
    
    
    if (dataset==3) or (dataset==4) or (dataset==5):
        # 4. miRNA
        label_er_mir_no_er_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_mir_no_er_null]),2]
        label_er_mir_no_er_null = label_er_mir_no_er_null.tolist()
        label_er_mir_no_er_null = [0 if x=='Positive' else x for x in label_er_mir_no_er_null]
        label_er_mir_no_er_null = [1 if x=='Negative' else x for x in label_er_mir_no_er_null]
        label_er_mir_no_er_null = [2 if x=='Indeterminate' else x for x in label_er_mir_no_er_null]
        label_er_mir_no_er_null = np.asarray(label_er_mir_no_er_null)
        
        label_pgr_mir_no_pgr_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_mir_no_pgr_null]),4]
        label_pgr_mir_no_pgr_null = label_pgr_mir_no_pgr_null.tolist()
        label_pgr_mir_no_pgr_null = [0 if x=='Positive' else x for x in label_pgr_mir_no_pgr_null]
        label_pgr_mir_no_pgr_null = [1 if x=='Negative' else x for x in label_pgr_mir_no_pgr_null]
        label_pgr_mir_no_pgr_null = [2 if x=='Indeterminate' else x for x in label_pgr_mir_no_pgr_null]
        label_pgr_mir_no_pgr_null = np.asarray(label_pgr_mir_no_pgr_null)
        
        label_her2_mir_no_her2_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_mir_no_her2_null]),7]
        label_her2_mir_no_her2_null = label_her2_mir_no_her2_null.tolist()
        label_her2_mir_no_her2_null = [0 if x=='Positive' else x for x in label_her2_mir_no_her2_null]
        label_her2_mir_no_her2_null = [1 if x=='Negative' else x for x in label_her2_mir_no_her2_null]
        label_her2_mir_no_her2_null = [2 if x=='Indeterminate' else x for x in label_her2_mir_no_her2_null]
        label_her2_mir_no_her2_null = [3 if x=='Equivocal' else x for x in label_her2_mir_no_her2_null]
        label_her2_mir_no_her2_null = np.asarray(label_her2_mir_no_her2_null)
        
        label_er_mir_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_mir_no_null]),2]
        label_er_mir_no_null = label_er_mir_no_null.tolist()
        label_er_mir_no_null = [0 if x=='Positive' else x for x in label_er_mir_no_null]
        label_er_mir_no_null = [1 if x=='Negative' else x for x in label_er_mir_no_null]
        label_er_mir_no_null = [2 if x=='Indeterminate' else x for x in label_er_mir_no_null]
        label_er_mir_no_null = np.asarray(label_er_mir_no_null)
        
        label_pgr_mir_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_mir_no_null]),4]
        label_pgr_mir_no_null = label_pgr_mir_no_null.tolist()
        label_pgr_mir_no_null = [0 if x=='Positive' else x for x in label_pgr_mir_no_null]
        label_pgr_mir_no_null = [1 if x=='Negative' else x for x in label_pgr_mir_no_null]
        label_pgr_mir_no_null = [2 if x=='Indeterminate' else x for x in label_pgr_mir_no_null]
        label_pgr_mir_no_null = np.asarray(label_pgr_mir_no_null)
        
        label_her2_mir_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_mir_no_null]),7]
        label_her2_mir_no_null = label_her2_mir_no_null.tolist()
        label_her2_mir_no_null = [0 if x=='Positive' else x for x in label_her2_mir_no_null]
        label_her2_mir_no_null = [1 if x=='Negative' else x for x in label_her2_mir_no_null]
        label_her2_mir_no_null = [2 if x=='Indeterminate' else x for x in label_her2_mir_no_null]
        label_her2_mir_no_null = [3 if x=='Equivocal' else x for x in label_her2_mir_no_null]
        label_her2_mir_no_null = np.asarray(label_her2_mir_no_null)
    
    
    if (dataset==4) or (dataset==5):
        # 5. Gene + miRNA
        label_er_gen_mir_no_er_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_gen_mir_no_er_null]),2]
        label_er_gen_mir_no_er_null = label_er_gen_mir_no_er_null.tolist()
        label_er_gen_mir_no_er_null = [0 if x=='Positive' else x for x in label_er_gen_mir_no_er_null]
        label_er_gen_mir_no_er_null = [1 if x=='Negative' else x for x in label_er_gen_mir_no_er_null]
        label_er_gen_mir_no_er_null = [2 if x=='Indeterminate' else x for x in label_er_gen_mir_no_er_null]
        label_er_gen_mir_no_er_null = np.asarray(label_er_gen_mir_no_er_null)
        
        label_pgr_gen_mir_no_pgr_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_gen_mir_no_pgr_null]),4]
        label_pgr_gen_mir_no_pgr_null = label_pgr_gen_mir_no_pgr_null.tolist()
        label_pgr_gen_mir_no_pgr_null = [0 if x=='Positive' else x for x in label_pgr_gen_mir_no_pgr_null]
        label_pgr_gen_mir_no_pgr_null = [1 if x=='Negative' else x for x in label_pgr_gen_mir_no_pgr_null]
        label_pgr_gen_mir_no_pgr_null = [2 if x=='Indeterminate' else x for x in label_pgr_gen_mir_no_pgr_null]
        label_pgr_gen_mir_no_pgr_null = np.asarray(label_pgr_gen_mir_no_pgr_null)
        
        label_her2_gen_mir_no_her2_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_gen_mir_no_her2_null]),7]
        label_her2_gen_mir_no_her2_null = label_her2_gen_mir_no_her2_null.tolist()
        label_her2_gen_mir_no_her2_null = [0 if x=='Positive' else x for x in label_her2_gen_mir_no_her2_null]
        label_her2_gen_mir_no_her2_null = [1 if x=='Negative' else x for x in label_her2_gen_mir_no_her2_null]
        label_her2_gen_mir_no_her2_null = [2 if x=='Indeterminate' else x for x in label_her2_gen_mir_no_her2_null]
        label_her2_gen_mir_no_her2_null = [3 if x=='Equivocal' else x for x in label_her2_gen_mir_no_her2_null]
        label_her2_gen_mir_no_her2_null = np.asarray(label_her2_gen_mir_no_her2_null)
        
        label_er_gen_mir_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_gen_mir_no_null]),2]
        label_er_gen_mir_no_null = label_er_gen_mir_no_null.tolist()
        label_er_gen_mir_no_null = [0 if x=='Positive' else x for x in label_er_gen_mir_no_null]
        label_er_gen_mir_no_null = [1 if x=='Negative' else x for x in label_er_gen_mir_no_null]
        label_er_gen_mir_no_null = [2 if x=='Indeterminate' else x for x in label_er_gen_mir_no_null]
        label_er_gen_mir_no_null = np.asarray(label_er_gen_mir_no_null)
        
        label_pgr_gen_mir_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_gen_mir_no_null]),4]
        label_pgr_gen_mir_no_null = label_pgr_gen_mir_no_null.tolist()
        label_pgr_gen_mir_no_null = [0 if x=='Positive' else x for x in label_pgr_gen_mir_no_null]
        label_pgr_gen_mir_no_null = [1 if x=='Negative' else x for x in label_pgr_gen_mir_no_null]
        label_pgr_gen_mir_no_null = [2 if x=='Indeterminate' else x for x in label_pgr_gen_mir_no_null]
        label_pgr_gen_mir_no_null = np.asarray(label_pgr_gen_mir_no_null)
        
        label_her2_gen_mir_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_gen_mir_no_null]),7]
        label_her2_gen_mir_no_null = label_her2_gen_mir_no_null.tolist()
        label_her2_gen_mir_no_null = [0 if x=='Positive' else x for x in label_her2_gen_mir_no_null]
        label_her2_gen_mir_no_null = [1 if x=='Negative' else x for x in label_her2_gen_mir_no_null]
        label_her2_gen_mir_no_null = [2 if x=='Indeterminate' else x for x in label_her2_gen_mir_no_null]
        label_her2_gen_mir_no_null = [3 if x=='Equivocal' else x for x in label_her2_gen_mir_no_null]
        label_her2_gen_mir_no_null = np.asarray(label_her2_gen_mir_no_null)
    
    
    if (dataset==5):
        # 6. Methylation + Gene + miRNA
        label_er_met_gen_mir_no_er_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_met_gen_mir_no_er_null]),2]
        label_er_met_gen_mir_no_er_null = label_er_met_gen_mir_no_er_null.tolist()
        label_er_met_gen_mir_no_er_null = [0 if x=='Positive' else x for x in label_er_met_gen_mir_no_er_null]
        label_er_met_gen_mir_no_er_null = [1 if x=='Negative' else x for x in label_er_met_gen_mir_no_er_null]
        label_er_met_gen_mir_no_er_null = [2 if x=='Indeterminate' else x for x in label_er_met_gen_mir_no_er_null]
        label_er_met_gen_mir_no_er_null = np.asarray(label_er_met_gen_mir_no_er_null)
        
        label_pgr_met_gen_mir_no_pgr_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_met_gen_mir_no_pgr_null]),4]
        label_pgr_met_gen_mir_no_pgr_null = label_pgr_met_gen_mir_no_pgr_null.tolist()
        label_pgr_met_gen_mir_no_pgr_null = [0 if x=='Positive' else x for x in label_pgr_met_gen_mir_no_pgr_null]
        label_pgr_met_gen_mir_no_pgr_null = [1 if x=='Negative' else x for x in label_pgr_met_gen_mir_no_pgr_null]
        label_pgr_met_gen_mir_no_pgr_null = [2 if x=='Indeterminate' else x for x in label_pgr_met_gen_mir_no_pgr_null]
        label_pgr_met_gen_mir_no_pgr_null = np.asarray(label_pgr_met_gen_mir_no_pgr_null)
        
        label_her2_met_gen_mir_no_her2_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_met_gen_mir_no_her2_null]),7]
        label_her2_met_gen_mir_no_her2_null = label_her2_met_gen_mir_no_her2_null.tolist()
        label_her2_met_gen_mir_no_her2_null = [0 if x=='Positive' else x for x in label_her2_met_gen_mir_no_her2_null]
        label_her2_met_gen_mir_no_her2_null = [1 if x=='Negative' else x for x in label_her2_met_gen_mir_no_her2_null]
        label_her2_met_gen_mir_no_her2_null = [2 if x=='Indeterminate' else x for x in label_her2_met_gen_mir_no_her2_null]
        label_her2_met_gen_mir_no_her2_null = [3 if x=='Equivocal' else x for x in label_her2_met_gen_mir_no_her2_null]
        label_her2_met_gen_mir_no_her2_null = np.asarray(label_her2_met_gen_mir_no_her2_null)
        
        label_er_met_gen_mir_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_met_gen_mir_no_null]),2]
        label_er_met_gen_mir_no_null = label_er_met_gen_mir_no_null.tolist()
        label_er_met_gen_mir_no_null = [0 if x=='Positive' else x for x in label_er_met_gen_mir_no_null]
        label_er_met_gen_mir_no_null = [1 if x=='Negative' else x for x in label_er_met_gen_mir_no_null]
        label_er_met_gen_mir_no_null = [2 if x=='Indeterminate' else x for x in label_er_met_gen_mir_no_null]
        label_er_met_gen_mir_no_null = np.asarray(label_er_met_gen_mir_no_null)
        
        label_pgr_met_gen_mir_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_met_gen_mir_no_null]),4]
        label_pgr_met_gen_mir_no_null = label_pgr_met_gen_mir_no_null.tolist()
        label_pgr_met_gen_mir_no_null = [0 if x=='Positive' else x for x in label_pgr_met_gen_mir_no_null]
        label_pgr_met_gen_mir_no_null = [1 if x=='Negative' else x for x in label_pgr_met_gen_mir_no_null]
        label_pgr_met_gen_mir_no_null = [2 if x=='Indeterminate' else x for x in label_pgr_met_gen_mir_no_null]
        label_pgr_met_gen_mir_no_null = np.asarray(label_pgr_met_gen_mir_no_null)
        
        label_her2_met_gen_mir_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_met_gen_mir_no_null]),7]
        label_her2_met_gen_mir_no_null = label_her2_met_gen_mir_no_null.tolist()
        label_her2_met_gen_mir_no_null = [0 if x=='Positive' else x for x in label_her2_met_gen_mir_no_null]
        label_her2_met_gen_mir_no_null = [1 if x=='Negative' else x for x in label_her2_met_gen_mir_no_null]
        label_her2_met_gen_mir_no_null = [2 if x=='Indeterminate' else x for x in label_her2_met_gen_mir_no_null]
        label_her2_met_gen_mir_no_null = [3 if x=='Equivocal' else x for x in label_her2_met_gen_mir_no_null]
        label_her2_met_gen_mir_no_null = np.asarray(label_her2_met_gen_mir_no_null)
        
        
        # 7. Methylation Long + Gene + miRNA
        label_er_metlong_gen_mir_no_er_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_metlong_gen_mir_no_er_null]),2]
        label_er_metlong_gen_mir_no_er_null = label_er_metlong_gen_mir_no_er_null.tolist()
        label_er_metlong_gen_mir_no_er_null = [0 if x=='Positive' else x for x in label_er_metlong_gen_mir_no_er_null]
        label_er_metlong_gen_mir_no_er_null = [1 if x=='Negative' else x for x in label_er_metlong_gen_mir_no_er_null]
        label_er_metlong_gen_mir_no_er_null = [2 if x=='Indeterminate' else x for x in label_er_metlong_gen_mir_no_er_null]
        label_er_metlong_gen_mir_no_er_null = np.asarray(label_er_metlong_gen_mir_no_er_null)
        
        label_pgr_metlong_gen_mir_no_pgr_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_metlong_gen_mir_no_pgr_null]),4]
        label_pgr_metlong_gen_mir_no_pgr_null = label_pgr_metlong_gen_mir_no_pgr_null.tolist()
        label_pgr_metlong_gen_mir_no_pgr_null = [0 if x=='Positive' else x for x in label_pgr_metlong_gen_mir_no_pgr_null]
        label_pgr_metlong_gen_mir_no_pgr_null = [1 if x=='Negative' else x for x in label_pgr_metlong_gen_mir_no_pgr_null]
        label_pgr_metlong_gen_mir_no_pgr_null = [2 if x=='Indeterminate' else x for x in label_pgr_metlong_gen_mir_no_pgr_null]
        label_pgr_metlong_gen_mir_no_pgr_null = np.asarray(label_pgr_metlong_gen_mir_no_pgr_null)
        
        label_her2_metlong_gen_mir_no_her2_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_metlong_gen_mir_no_her2_null]),7]
        label_her2_metlong_gen_mir_no_her2_null = label_her2_metlong_gen_mir_no_her2_null.tolist()
        label_her2_metlong_gen_mir_no_her2_null = [0 if x=='Positive' else x for x in label_her2_metlong_gen_mir_no_her2_null]
        label_her2_metlong_gen_mir_no_her2_null = [1 if x=='Negative' else x for x in label_her2_metlong_gen_mir_no_her2_null]
        label_her2_metlong_gen_mir_no_her2_null = [2 if x=='Indeterminate' else x for x in label_her2_metlong_gen_mir_no_her2_null]
        label_her2_metlong_gen_mir_no_her2_null = [3 if x=='Equivocal' else x for x in label_her2_metlong_gen_mir_no_her2_null]
        label_her2_metlong_gen_mir_no_her2_null = np.asarray(label_her2_metlong_gen_mir_no_her2_null)
        
        label_er_metlong_gen_mir_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_metlong_gen_mir_no_null]),2]
        label_er_metlong_gen_mir_no_null = label_er_metlong_gen_mir_no_null.tolist()
        label_er_metlong_gen_mir_no_null = [0 if x=='Positive' else x for x in label_er_metlong_gen_mir_no_null]
        label_er_metlong_gen_mir_no_null = [1 if x=='Negative' else x for x in label_er_metlong_gen_mir_no_null]
        label_er_metlong_gen_mir_no_null = [2 if x=='Indeterminate' else x for x in label_er_metlong_gen_mir_no_null]
        label_er_metlong_gen_mir_no_null = np.asarray(label_er_metlong_gen_mir_no_null)
        
        label_pgr_metlong_gen_mir_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_metlong_gen_mir_no_null]),4]
        label_pgr_metlong_gen_mir_no_null = label_pgr_metlong_gen_mir_no_null.tolist()
        label_pgr_metlong_gen_mir_no_null = [0 if x=='Positive' else x for x in label_pgr_metlong_gen_mir_no_null]
        label_pgr_metlong_gen_mir_no_null = [1 if x=='Negative' else x for x in label_pgr_metlong_gen_mir_no_null]
        label_pgr_metlong_gen_mir_no_null = [2 if x=='Indeterminate' else x for x in label_pgr_metlong_gen_mir_no_null]
        label_pgr_metlong_gen_mir_no_null = np.asarray(label_pgr_metlong_gen_mir_no_null)
        
        label_her2_metlong_gen_mir_no_null = pat_rec[np.logical_or.reduce([pat_rec[:,0] == i for i in cases_metlong_gen_mir_no_null]),7]
        label_her2_metlong_gen_mir_no_null = label_her2_metlong_gen_mir_no_null.tolist()
        label_her2_metlong_gen_mir_no_null = [0 if x=='Positive' else x for x in label_her2_metlong_gen_mir_no_null]
        label_her2_metlong_gen_mir_no_null = [1 if x=='Negative' else x for x in label_her2_metlong_gen_mir_no_null]
        label_her2_metlong_gen_mir_no_null = [2 if x=='Indeterminate' else x for x in label_her2_metlong_gen_mir_no_null]
        label_her2_metlong_gen_mir_no_null = [3 if x=='Equivocal' else x for x in label_her2_metlong_gen_mir_no_null]
        label_her2_metlong_gen_mir_no_null = np.asarray(label_her2_metlong_gen_mir_no_null)


    ######################################
    # SAVE LABELS OF TYPE CLASSIFICATION #
    ######################################
    if (dataset==1) or (dataset==5):
        # 1. Methylation
        if not os.path.isdir(DATASET_LABELS_MET_TYPE):
            os.makedirs(DATASET_LABELS_MET_TYPE)
        os.chdir(DATASET_LABELS_MET_TYPE)
        np.save('label_type_er_met.npy', label_er_met_no_er_null)
        np.save('label_type_pgr_met.npy', label_pgr_met_no_pgr_null)
        np.save('label_type_her2_met.npy', label_her2_met_no_her2_null)
        np.save('label_type_er_met_univ.npy', label_er_met_no_null)
        np.save('label_type_pgr_met_univ.npy', label_pgr_met_no_null)
        np.save('label_type_her2_met_univ.npy', label_her2_met_no_null)
        
        # 2. Methylation Long
        if not os.path.isdir(DATASET_LABELS_METLONG_TYPE):
            os.makedirs(DATASET_LABELS_METLONG_TYPE)
        os.chdir(DATASET_LABELS_METLONG_TYPE)
        np.save('label_type_er_metlong.npy', label_er_metlong_no_er_null)
        np.save('label_type_pgr_metlong.npy', label_pgr_metlong_no_pgr_null)
        np.save('label_type_her2_metlong.npy', label_her2_metlong_no_her2_null)
        np.save('label_type_er_metlong_univ.npy', label_er_metlong_no_null)
        np.save('label_type_pgr_metlong_univ.npy', label_pgr_metlong_no_null)
        np.save('label_type_her2_metlong_univ.npy', label_her2_metlong_no_null)

    if (dataset==2) or (dataset==4) or (dataset==5):
        # 3. Gene
        if not os.path.isdir(DATASET_LABELS_GEN_TYPE):
            os.makedirs(DATASET_LABELS_GEN_TYPE)
        os.chdir(DATASET_LABELS_GEN_TYPE)
        np.save('label_type_er_gen.npy', label_er_gen_no_er_null)
        np.save('label_type_pgr_gen.npy', label_pgr_gen_no_pgr_null)
        np.save('label_type_her2_gen.npy', label_her2_gen_no_her2_null)
        np.save('label_type_er_gen_univ.npy', label_er_gen_no_null)
        np.save('label_type_pgr_gen_univ.npy', label_pgr_gen_no_null)
        np.save('label_type_her2_gen_univ.npy', label_her2_gen_no_null)
        
    if (dataset==3) or (dataset==4) or (dataset==5):
        # 4. miRNA
        if not os.path.isdir(DATASET_LABELS_MIR_TYPE):
            os.makedirs(DATASET_LABELS_MIR_TYPE)
        os.chdir(DATASET_LABELS_MIR_TYPE)
        np.save('label_type_er_mir.npy', label_er_mir_no_er_null)
        np.save('label_type_pgr_mir.npy', label_pgr_mir_no_pgr_null)
        np.save('label_type_her2_mir.npy', label_her2_mir_no_her2_null)
        np.save('label_type_er_mir_univ.npy', label_er_mir_no_null)
        np.save('label_type_pgr_mir_univ.npy', label_pgr_mir_no_null)
        np.save('label_type_her2_mir_univ.npy', label_her2_mir_no_null)
        
    if (dataset==4) or (dataset==5):
        # 5. Gene + miRNA
        if not os.path.isdir(DATASET_LABELS_GEN_MIR_TYPE):
            os.makedirs(DATASET_LABELS_GEN_MIR_TYPE)
        os.chdir(DATASET_LABELS_GEN_MIR_TYPE)
        np.save('label_type_er_gen_mir.npy', label_er_gen_mir_no_er_null)
        np.save('label_type_pgr_gen_mir.npy', label_pgr_gen_mir_no_pgr_null)
        np.save('label_type_her2_gen_mir.npy', label_her2_gen_mir_no_her2_null)
        np.save('label_type_er_gen_mir_univ.npy', label_er_gen_mir_no_null)
        np.save('label_type_pgr_gen_mir_univ.npy', label_pgr_gen_mir_no_null)
        np.save('label_type_her2_gen_mir_univ.npy', label_her2_gen_mir_no_null)
        
    if (dataset==5):
        # 6. Methylation + Gene + miRNA
        if not os.path.isdir(DATASET_LABELS_MET_GEN_MIR_TYPE):
            os.makedirs(DATASET_LABELS_MET_GEN_MIR_TYPE)
        os.chdir(DATASET_LABELS_MET_GEN_MIR_TYPE)
        np.save('label_type_er_met_gen_mir.npy', label_er_met_gen_mir_no_er_null)
        np.save('label_type_pgr_met_gen_mir.npy', label_pgr_met_gen_mir_no_pgr_null)
        np.save('label_type_her2_met_gen_mir.npy', label_her2_met_gen_mir_no_her2_null)
        np.save('label_type_er_met_gen_mir_univ.npy', label_er_met_gen_mir_no_null)
        np.save('label_type_pgr_met_gen_mir_univ.npy', label_pgr_met_gen_mir_no_null)
        np.save('label_type_her2_met_gen_mir_univ.npy', label_her2_met_gen_mir_no_null)
            
        # 7. Methylation Long + Gene + miRNA
        if not os.path.isdir(DATASET_LABELS_METLONG_GEN_MIR_TYPE):
            os.makedirs(DATASET_LABELS_METLONG_GEN_MIR_TYPE)
        os.chdir(DATASET_LABELS_METLONG_GEN_MIR_TYPE)
        np.save('label_type_er_metlong_gen_mir.npy', label_er_metlong_gen_mir_no_er_null)
        np.save('label_type_pgr_metlong_gen_mir.npy', label_pgr_metlong_gen_mir_no_pgr_null)
        np.save('label_type_her2_metlong_gen_mir.npy', label_her2_metlong_gen_mir_no_her2_null)
        np.save('label_type_er_metlong_gen_mir_univ.npy', label_er_metlong_gen_mir_no_null)
        np.save('label_type_pgr_metlong_gen_mir_univ.npy', label_pgr_metlong_gen_mir_no_null)
        np.save('label_type_her2_metlong_gen_mir_univ.npy', label_her2_metlong_gen_mir_no_null)

    print("all cancer type labels are created")
    
    

# Create the input (feature) set of methylation data for cancer type classification
# available for both NCBI Platform GPL8490 and NCBI Platform GPL16304
# We use the beta value from CPG sites
def input_met_cancer_type():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_met = cases[cases[:,8]!="0",0]    # [1095,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_met_cli = np.intersect1d(cases_met,cases_cli) # [1095,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_pat_rec = np.genfromtxt(TARGET_CLINICAL + "pathology_receptor.csv", dtype=str, delimiter=',', skip_header=1)
    pat_rec = temp_pat_rec[temp_pat_rec[:,0].argsort()]

    cases_no_er_null = pat_rec[pat_rec[:,2]!="",0]     # [1048,]
    cases_no_pgr_null = pat_rec[pat_rec[:,4]!="",0]    # [1047,]
    cases_no_her2_null = pat_rec[pat_rec[:,7]!="",0]   # [919,]
    cases_no_null = np.intersect1d(np.intersect1d(cases_no_er_null,cases_no_pgr_null),cases_no_her2_null)   # [917,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_met_no_er_null = np.intersect1d(cases_met_cli,cases_no_er_null)       # (1046,)
    cases_met_no_pgr_null = np.intersect1d(cases_met_cli,cases_no_pgr_null)     # (1045,)
    cases_met_no_her2_null = np.intersect1d(cases_met_cli,cases_no_her2_null)   # (917,)
    cases_met_no_null = np.intersect1d(cases_met_cli,cases_no_null)             # (915,)
    
    
    ##############################################
    ## INPUT METHYLATION OF TYPE CLASSIFICATION ##
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_MET_TYPE)):
        os.makedirs(DATASET_INPUT_MET_TYPE)

    data_met = np.genfromtxt(TARGET_META_CSV + "methylation_beta_value.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_met[0]=='file_id')[0]
    file_name_column, = np.where(data_met[0]=='file_name')[0]
    case_id_column, = np.where(data_met[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_met[0]=='cases.0.samples.0.sample_type')[0]

    data_met = data_met[1:]

    with open(TARGET_METHYLATION + "cpg.json") as f:
        cpg = yaml.safe_load(f)

    with open(TARGET_METHYLATION + "cpg_in_cpg_short_idx.json") as f:
        cpg_in_cpg_short_idx = yaml.safe_load(f)

    with open(TARGET_METHYLATION + "cpg_in_cpg_long_idx.json") as f:
        cpg_in_cpg_long_idx = yaml.safe_load(f)


    # 1. Methylation ER classification
    input_met_type_er = np.empty((0,25978), float)
    for case in cases_met_no_er_null:
        file_id = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
            file_met = [row.split("\t") for row in f]

        file_met = file_met[1:]
        
        temp = np.empty((0,1), float)

        if len(file_met) == 27578:
            for idx in sorted(cpg_in_cpg_short_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])

                temp = np.append(temp, beta)

        elif len(file_met) == 485577:
            for idx in sorted(cpg_in_cpg_long_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])
                
                temp = np.append(temp, beta)
        
        print(case)
        input_met_type_er = np.vstack([input_met_type_er,temp])

    np.save(DATASET_INPUT_MET_TYPE + 'input_met_type_er.npy', input_met_type_er)
    print('input_met_type_er.npy is created')


    # 2. Methylation PGR classification
    input_met_type_pgr = np.empty((0,25978), float)
    for case in cases_met_no_pgr_null:
        file_id = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
            file_met = [row.split("\t") for row in f]

        file_met = file_met[1:]
        
        temp = np.empty((0,1), float)

        if len(file_met) == 27578:
            for idx in sorted(cpg_in_cpg_short_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])

                temp = np.append(temp, beta)

        elif len(file_met) == 485577:
            for idx in sorted(cpg_in_cpg_long_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])
                
                temp = np.append(temp, beta)
        
        print(case)
        input_met_type_pgr = np.vstack([input_met_type_pgr,temp])

    np.save(DATASET_INPUT_MET_TYPE + 'input_met_type_pgr.npy', input_met_type_pgr)
    print('input_met_type_pgr.npy is created')


    # 3. Methylation HER2 classification
    input_met_type_her2 = np.empty((0,25978), float)
    for case in cases_met_no_her2_null:
        file_id = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
            file_met = [row.split("\t") for row in f]

        file_met = file_met[1:]
        
        temp = np.empty((0,1), float)

        if len(file_met) == 27578:
            for idx in sorted(cpg_in_cpg_short_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])

                temp = np.append(temp, beta)

        elif len(file_met) == 485577:
            for idx in sorted(cpg_in_cpg_long_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])
                
                temp = np.append(temp, beta)
        
        print(case)
        input_met_type_her2 = np.vstack([input_met_type_her2,temp])

    np.save(DATASET_INPUT_MET_TYPE + 'input_met_type_her2.npy', input_met_type_her2)
    print('input_met_type_her2.npy is created')


    # 4. Methylation universal classification
    input_met_type_univ = np.empty((0,25978), float)
    for case in cases_met_no_null:
        file_id = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
            file_met = [row.split("\t") for row in f]

        file_met = file_met[1:]
        
        temp = np.empty((0,1), float)

        if len(file_met) == 27578:
            for idx in sorted(cpg_in_cpg_short_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])

                temp = np.append(temp, beta)

        elif len(file_met) == 485577:
            for idx in sorted(cpg_in_cpg_long_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])
                
                temp = np.append(temp, beta)
        
        print(case)
        input_met_type_univ = np.vstack([input_met_type_univ,temp])

    np.save(DATASET_INPUT_MET_TYPE + 'input_met_type_univ.npy', input_met_type_univ)
    print('input_met_type_univ.npy is created')



# Create the input (feature) set of methylation long data for cancer type classification
# only for NCBI Platform GPL16304
# We use the beta value from CPG sites
def input_metlong_cancer_type():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    with open(TARGET_METHYLATION + "cases_met_long.json") as f:
        temp = yaml.safe_load(f)
    cases_metlong = np.asarray(temp)  # [782,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_metlong_cli = np.intersect1d(cases_metlong,cases_cli) # [782,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_pat_rec = np.genfromtxt(TARGET_CLINICAL + "pathology_receptor.csv", dtype=str, delimiter=',', skip_header=1)
    pat_rec = temp_pat_rec[temp_pat_rec[:,0].argsort()]

    cases_no_er_null = pat_rec[pat_rec[:,2]!="",0]     # [1048,]
    cases_no_pgr_null = pat_rec[pat_rec[:,4]!="",0]    # [1047,]
    cases_no_her2_null = pat_rec[pat_rec[:,7]!="",0]   # [919,]
    cases_no_null = np.intersect1d(np.intersect1d(cases_no_er_null,cases_no_pgr_null),cases_no_her2_null)   # [917,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_metlong_no_er_null = np.intersect1d(cases_metlong_cli,cases_no_er_null)       # (738,)
    cases_metlong_no_pgr_null = np.intersect1d(cases_metlong_cli,cases_no_pgr_null)     # (737,)
    cases_metlong_no_her2_null = np.intersect1d(cases_metlong_cli,cases_no_her2_null)   # (642,)
    cases_metlong_no_null = np.intersect1d(cases_metlong_cli,cases_no_null)             # (640,)
    
    
    #####################################################
    ### INPUT LONG METHYLATION OF TYPE CLASSIFICATION ###
    #####################################################
    if not(os.path.isdir(DATASET_INPUT_METLONG_TYPE)):
        os.makedirs(DATASET_INPUT_METLONG_TYPE)

    data_metlong = np.genfromtxt(TARGET_META_CSV + "methylation_long_beta_value.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_metlong[0]=='file_id')[0]
    file_name_column, = np.where(data_metlong[0]=='file_name')[0]
    case_id_column, = np.where(data_metlong[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_metlong[0]=='cases.0.samples.0.sample_type')[0]

    data_metlong = data_metlong[1:]

    # 1. Methylation ER classification
    # we divided the resulting file in 4 because the size is too large for 1 file
    for i in range(4):
        input_metlong_type_er = []
        for case in cases_metlong_no_er_null[200*i:200*(i+1)]:
            file_id = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
            file_name = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
            
            with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
                file_metlong = [row.split("\t") for row in f]

            file_metlong = file_metlong[1:]
            
            temp = []

            for j in range(len(file_metlong)):
                if file_metlong[j][1] == "NA":
                    beta = 0.
                else:
                    beta = float(file_metlong[j][1])
                temp.append(beta)
            
            input_metlong_type_er.append(temp)
            print(str(len(input_metlong_type_er)) + ". " + case)

        np.save(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_er_' + str(i) + '.npy', input_metlong_type_er)
        print('input_metlong_type_er_' + str(i) + '.npy is created')

    # combine it in 1 file
    temp_0 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_er_0.npy')
    temp_1 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_er_1.npy')
    temp_2 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_er_2.npy')
    temp_3 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_er_3.npy')
    input_metlong_type_er = np.concatenate((np.concatenate((temp_0, temp_1), axis=0),
                                            np.concatenate((temp_2, temp_3), axis=0)),
                                           axis=0)
    np.save(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_er.npy', input_metlong_type_er)

    # remove temp file
    for i in range(4):
        os.remove(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_er_' + str(i) + '.npy')


    # 2. Methylation PGR classification
    # we divided the resulting file in 4 because the size is too large for 1 file
    for i in range(4):
        input_metlong_type_pgr = []
        for case in cases_metlong_no_pgr_null[200*i:200*(i+1)]:
            file_id = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
            file_name = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
            
            with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
                file_metlong = [row.split("\t") for row in f]

            file_metlong = file_metlong[1:]
            
            temp = []

            for j in range(len(file_metlong)):
                if file_metlong[j][1] == "NA":
                    beta = 0.
                else:
                    beta = float(file_metlong[j][1])
                temp.append(beta)
            
            input_metlong_type_pgr.append(temp)
            print(str(len(input_metlong_type_pgr)) + ". " + case)

        np.save(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_pgr_' + str(i) + '.npy', input_metlong_type_pgr)
        print('input_metlong_type_pgr_' + str(i) + '.npy is created')

    # combine it in 1 file
    temp_0 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_pgr_0.npy')
    temp_1 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_pgr_1.npy')
    temp_2 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_pgr_2.npy')
    temp_3 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_pgr_3.npy')
    input_metlong_type_pgr = np.concatenate((np.concatenate((temp_0, temp_1), axis=0),
                                             np.concatenate((temp_2, temp_3), axis=0)),
                                            axis=0)
    np.save(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_pgr.npy', input_metlong_type_pgr)

    # remove temp file
    for i in range(4):
        os.remove(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_pgr_' + str(i) + '.npy')


    # 3. Methylation HER2 classification
    # we divided the resulting file in 4 because the size is too large for 1 file
    for i in range(4):
        input_metlong_type_her2 = []
        for case in cases_metlong_no_her2_null[200*i:200*(i+1)]:
            file_id = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
            file_name = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
            
            with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
                file_metlong = [row.split("\t") for row in f]

            file_metlong = file_metlong[1:]
            
            temp = []

            for j in range(len(file_metlong)):
                if file_metlong[j][1] == "NA":
                    beta = 0.
                else:
                    beta = float(file_metlong[j][1])
                temp.append(beta)
            
            input_metlong_type_her2.append(temp)
            print(str(len(input_metlong_type_her2)) + ". " + case)

        np.save(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_her2_' + str(i) + '.npy', input_metlong_type_her2)
        print('input_metlong_type_her2_' + str(i) + '.npy is created')

    # combine it in 1 file
    temp_0 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_her2_0.npy')
    temp_1 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_her2_1.npy')
    temp_2 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_her2_2.npy')
    temp_3 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_her2_3.npy')
    input_metlong_type_her2 = np.concatenate((np.concatenate((temp_0, temp_1), axis=0),
                                              np.concatenate((temp_2, temp_3), axis=0)),
                                             axis=0)
    np.save(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_her2.npy', input_metlong_type_her2)

    # remove temp file
    for i in range(4):
        os.remove(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_her2_' + str(i) + '.npy')


    # 4. Methylation universal classification
    # we divided the resulting file in 4 because the size is too large for 1 file
    for i in range(4):
        input_metlong_type_univ = []
        for case in cases_metlong_no_null[200*i:200*(i+1)]:
            file_id = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
            file_name = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
            
            with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
                file_metlong = [row.split("\t") for row in f]

            file_metlong = file_metlong[1:]
            
            temp = []

            for j in range(len(file_metlong)):
                if file_metlong[j][1] == "NA":
                    beta = 0.
                else:
                    beta = float(file_metlong[j][1])
                temp.append(beta)
            
            input_metlong_type_univ.append(temp)
            print(str(len(input_metlong_type_univ)) + ". " + case)

        np.save(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_univ_' + str(i) + '.npy', input_metlong_type_univ)
        print('input_metlong_type_univ_' + str(i) + 'npy is created')

    # combine it in 1 file
    temp_0 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_univ_0.npy')
    temp_1 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_univ_1.npy')
    temp_2 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_univ_2.npy')
    temp_3 = np.load(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_univ_3.npy')
    input_metlong_type_univ = np.concatenate((np.concatenate((temp_0, temp_1), axis=0),
                                              np.concatenate((temp_2, temp_3), axis=0)),
                                             axis=0)
    np.save(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_univ.npy', input_metlong_type_univ)

    # remove temp file
    for i in range(4):
        os.remove(DATASET_INPUT_METLONG_TYPE + 'input_metlong_type_univ_' + str(i) + '.npy')



# Create the input (feature) set of gene expression for cancer type classification
# We use the expression value of each genes
# There are three different workflow analysis (HTSEC count, HTSEC FPKM, HTSEC FPKM-UQ) that was used to compute the expression value
# Based on this, we create input set for all 3 types of workflow analysis
def input_gen_cancer_type():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_pat_rec = np.genfromtxt(TARGET_CLINICAL + "pathology_receptor.csv", dtype=str, delimiter=',', skip_header=1)
    pat_rec = temp_pat_rec[temp_pat_rec[:,0].argsort()]

    cases_no_er_null = pat_rec[pat_rec[:,2]!="",0]     # [1048,]
    cases_no_pgr_null = pat_rec[pat_rec[:,4]!="",0]    # [1047,]
    cases_no_her2_null = pat_rec[pat_rec[:,7]!="",0]   # [919,]
    cases_no_null = np.intersect1d(np.intersect1d(cases_no_er_null,cases_no_pgr_null),cases_no_her2_null)   # [917,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_gen_no_er_null = np.intersect1d(cases_gen_cli,cases_no_er_null)       # (1042,)
    cases_gen_no_pgr_null = np.intersect1d(cases_gen_cli,cases_no_pgr_null)     # (1041,)
    cases_gen_no_her2_null = np.intersect1d(cases_gen_cli,cases_no_her2_null)   # (913,)
    cases_gen_no_null = np.intersect1d(cases_gen_cli,cases_no_null)             # (912,)
    
    
    ##############################################
    ###### INPUT GENE OF TYPE CLASSIFICATION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_GEN_TYPE)):
        os.makedirs(DATASET_INPUT_GEN_TYPE)

    data_gene = np.genfromtxt(TARGET_META_CSV + "gene_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_gene[0]=='file_id')[0]
    file_name_column, = np.where(data_gene[0]=='file_name')[0]
    case_id_column, = np.where(data_gene[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_gene[0]=='cases.0.samples.0.sample_type')[0]
    workflow_column, = np.where(data_gene[0]=='analysis.workflow_type')[0]

    data_gene = data_gene[1:]

    # 1.a. Gene (count) ER classification
    input_gen_count_type_er = np.empty((0,60483), int)
    for case in cases_gen_no_er_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_count_type_er = np.vstack([input_gen_count_type_er,row])

    np.save(DATASET_INPUT_GEN_TYPE + 'input_gen_count_type_er.npy', input_gen_count_type_er)
    print('input_gen_count_type_er.npy is created')


    # 1.b. Gene (FPKM) ER classification
    input_gen_fpkm_type_er = np.empty((0,60483), int)
    for case in cases_gen_no_er_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_fpkm_type_er = np.vstack([input_gen_fpkm_type_er,row])

    np.save(DATASET_INPUT_GEN_TYPE + 'input_gen_fpkm_type_er.npy', input_gen_fpkm_type_er)
    print('input_gen_fpkm_type_er.npy is created')


    # 1.c. Gene (FPKM-UQ) ER classification
    input_gen_fpkmuq_type_er = np.empty((0,60483), int)
    for case in cases_gen_no_er_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_fpkmuq_type_er = np.vstack([input_gen_fpkmuq_type_er,row])

    np.save(DATASET_INPUT_GEN_TYPE + 'input_gen_fpkmuq_type_er.npy', input_gen_fpkmuq_type_er)
    print('input_gen_fpkmuq_type_er.npy is created')


    # 2.a. Gene (count) PGR classification
    input_gen_count_type_pgr = np.empty((0,60483), int)
    for case in cases_gen_no_pgr_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_count_type_pgr = np.vstack([input_gen_count_type_pgr,row])

    np.save(DATASET_INPUT_GEN_TYPE + 'input_gen_count_type_pgr.npy', input_gen_count_type_pgr)
    print('input_gen_count_type_pgr.npy is created')


    # 2.b. Gene (FPKM) PGR classification
    input_gen_fpkm_type_pgr = np.empty((0,60483), int)
    for case in cases_gen_no_pgr_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_fpkm_type_pgr = np.vstack([input_gen_fpkm_type_pgr,row])

    np.save(DATASET_INPUT_GEN_TYPE + 'input_gen_fpkm_type_pgr.npy', input_gen_fpkm_type_pgr)
    print('input_gen_fpkm_type_pgr.npy is created')


    # 2.c. Gene (FPKM-UQ) PGR classification
    input_gen_fpkmuq_type_pgr = np.empty((0,60483), int)
    for case in cases_gen_no_pgr_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_fpkmuq_type_pgr = np.vstack([input_gen_fpkmuq_type_pgr,row])

    np.save(DATASET_INPUT_GEN_TYPE + 'input_gen_fpkmuq_type_pgr.npy', input_gen_fpkmuq_type_pgr)
    print('input_gen_fpkmuq_type_pgr.npy is created')


    # 3.a. Gene (count) HER2 classification
    input_gen_count_type_her2 = np.empty((0,60483), int)
    for case in cases_gen_no_her2_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_count_type_her2 = np.vstack([input_gen_count_type_her2,row])

    np.save(DATASET_INPUT_GEN_TYPE + 'input_gen_count_type_her2.npy', input_gen_count_type_her2)
    print('input_gen_count_type_her2.npy is created')


    # 3.b. Gene (FPKM) HER2 classification
    input_gen_fpkm_type_her2 = np.empty((0,60483), int)
    for case in cases_gen_no_her2_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_fpkm_type_her2 = np.vstack([input_gen_fpkm_type_her2,row])

    np.save(DATASET_INPUT_GEN_TYPE + 'input_gen_fpkm_type_her2.npy', input_gen_fpkm_type_her2)
    print('input_gen_fpkm_type_her2.npy is created')


    # 3.c. Gene (FPKM-UQ) HER2 classification
    input_gen_fpkmuq_type_her2 = np.empty((0,60483), int)
    for case in cases_gen_no_her2_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_fpkmuq_type_her2 = np.vstack([input_gen_fpkmuq_type_her2,row])

    np.save(DATASET_INPUT_GEN_TYPE + 'input_gen_fpkmuq_type_her2.npy', input_gen_fpkmuq_type_her2)
    print('input_gen_fpkmuq_type_her2.npy is created')


    # 4.a. Gene (count) universal classification
    input_gen_count_type_univ = np.empty((0,60483), int)
    for case in cases_gen_no_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_count_type_univ = np.vstack([input_gen_count_type_univ,row])

    np.save(DATASET_INPUT_GEN_TYPE + 'input_gen_count_type_univ.npy', input_gen_count_type_univ)
    print('input_gen_count_type_univ.npy is created')


    # 4.b. Gene (FPKM) universal classification
    input_gen_fpkm_type_univ = np.empty((0,60483), int)
    for case in cases_gen_no_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_fpkm_type_univ = np.vstack([input_gen_fpkm_type_univ,row])

    np.save(DATASET_INPUT_GEN_TYPE + 'input_gen_fpkm_type_univ.npy', input_gen_fpkm_type_univ)
    print('input_gen_fpkm_type_univ.npy is created')


    # 4.c. Gene (FPKM-UQ) universal classification
    input_gen_fpkmuq_type_univ = np.empty((0,60483), int)
    for case in cases_gen_no_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_fpkmuq_type_univ = np.vstack([input_gen_fpkmuq_type_univ,row])

    np.save(DATASET_INPUT_GEN_TYPE + 'input_gen_fpkmuq_type_univ.npy', input_gen_fpkmuq_type_univ)
    print('input_gen_fpkmuq_type_univ.npy is created')



# Create the input (feature) set of miRNA expression data for cancer type classification
# We use the read count value of each miRNA
def input_mir_cancer_type():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_pat_rec = np.genfromtxt(TARGET_CLINICAL + "pathology_receptor.csv", dtype=str, delimiter=',', skip_header=1)
    pat_rec = temp_pat_rec[temp_pat_rec[:,0].argsort()]

    cases_no_er_null = pat_rec[pat_rec[:,2]!="",0]     # [1048,]
    cases_no_pgr_null = pat_rec[pat_rec[:,4]!="",0]    # [1047,]
    cases_no_her2_null = pat_rec[pat_rec[:,7]!="",0]   # [919,]
    cases_no_null = np.intersect1d(np.intersect1d(cases_no_er_null,cases_no_pgr_null),cases_no_her2_null)   # [917,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_mir_no_er_null = np.intersect1d(cases_mir_cli,cases_no_er_null)       # (1029,)
    cases_mir_no_pgr_null = np.intersect1d(cases_mir_cli,cases_no_pgr_null)     # (1028,)
    cases_mir_no_her2_null = np.intersect1d(cases_mir_cli,cases_no_her2_null)   # (902,)
    cases_mir_no_null = np.intersect1d(cases_mir_cli,cases_no_null)             # (900,)
    
    
    ##############################################
    ##### INPUT MIRNA OF TYPE CLASSIFICATION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_MIR_TYPE)):
        os.makedirs(DATASET_INPUT_MIR_TYPE)

    data_mir = np.genfromtxt(TARGET_META_CSV + "mirna_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_mir[0]=='file_id')[0]
    file_name_column, = np.where(data_mir[0]=='file_name')[0]
    case_id_column, = np.where(data_mir[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_mir[0]=='cases.0.samples.0.sample_type')[0]

    data_mir = data_mir[1:]

    # 1. miRNA ER classification
    input_mir_type_er = np.empty((0,1881), float)
    for case in cases_mir_no_er_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_type_er = np.vstack([input_mir_type_er,row])

    np.save(DATASET_INPUT_MIR_TYPE + 'input_mir_type_er.npy', input_mir_type_er)
    print('input_mir_type_er.npy is created')


    # 2. miRNA PGR classification
    input_mir_type_pgr = np.empty((0,1881), float)
    for case in cases_mir_no_pgr_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_type_pgr = np.vstack([input_mir_type_pgr,row])

    np.save(DATASET_INPUT_MIR_TYPE + 'input_mir_type_pgr.npy', input_mir_type_pgr)
    print('input_mir_type_pgr.npy is created')


    # 3. miRNA HER2 classification
    input_mir_type_her2 = np.empty((0,1881), float)
    for case in cases_mir_no_her2_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_type_her2 = np.vstack([input_mir_type_her2,row])

    np.save(DATASET_INPUT_MIR_TYPE + 'input_mir_type_her2.npy', input_mir_type_her2)
    print('input_mir_type_her2.npy is created')


    # 4. miRNA universal classification
    input_mir_type_univ = np.empty((0,1881), float)
    for case in cases_mir_no_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_type_univ = np.vstack([input_mir_type_univ,row])

    np.save(DATASET_INPUT_MIR_TYPE + 'input_mir_type_univ.npy', input_mir_type_univ)
    print('input_mir_type_univ.npy is created')



# Create the input (feature) set of gene expression for cancer type classification
# implemented in mDBN of gene-miRNA
# We use the expression value of each genes
# There are three different workflow analysis (HTSEC count, HTSEC FPKM, HTSEC FPKM-UQ) that was used to compute the expression value
# Based on this, we create input set for all 3 types of workflow analysis
def input_gen_genmir_cancer_type():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_gen_mir_cli = np.intersect1d(cases_gen_cli,cases_mir) # [1071,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_pat_rec = np.genfromtxt(TARGET_CLINICAL + "pathology_receptor.csv", dtype=str, delimiter=',', skip_header=1)
    pat_rec = temp_pat_rec[temp_pat_rec[:,0].argsort()]

    cases_no_er_null = pat_rec[pat_rec[:,2]!="",0]     # [1048,]
    cases_no_pgr_null = pat_rec[pat_rec[:,4]!="",0]    # [1047,]
    cases_no_her2_null = pat_rec[pat_rec[:,7]!="",0]   # [919,]
    cases_no_null = np.intersect1d(np.intersect1d(cases_no_er_null,cases_no_pgr_null),cases_no_her2_null)   # [917,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_gen_mir_no_er_null = np.intersect1d(cases_gen_mir_cli,cases_no_er_null)       # (1024,)
    cases_gen_mir_no_pgr_null = np.intersect1d(cases_gen_mir_cli,cases_no_pgr_null)     # (1023,)
    cases_gen_mir_no_her2_null = np.intersect1d(cases_gen_mir_cli,cases_no_her2_null)   # (897,)
    cases_gen_mir_no_null = np.intersect1d(cases_gen_mir_cli,cases_no_null)             # (896,)
    
    
    ##############################################
    ###### INPUT GENE OF TYPE CLASSIFICATION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_GEN_GEN_MIR_TYPE)):
        os.makedirs(DATASET_INPUT_GEN_GEN_MIR_TYPE)

    data_gene = np.genfromtxt(TARGET_META_CSV + "gene_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_gene[0]=='file_id')[0]
    file_name_column, = np.where(data_gene[0]=='file_name')[0]
    case_id_column, = np.where(data_gene[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_gene[0]=='cases.0.samples.0.sample_type')[0]
    workflow_column, = np.where(data_gene[0]=='analysis.workflow_type')[0]

    data_gene = data_gene[1:]

    # 1.a. Gene (count) ER classification
    input_gen_genmir_count_type_er = np.empty((0,60483), int)
    for case in cases_gen_mir_no_er_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_genmir_count_type_er = np.vstack([input_gen_genmir_count_type_er,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_TYPE + 'input_gen_genmir_count_type_er.npy', input_gen_genmir_count_type_er)
    print('input_gen_genmir_count_type_er.npy is created')


    # 1.b. Gene (FPKM) ER classification
    input_gen_genmir_fpkm_type_er = np.empty((0,60483), int)
    for case in cases_gen_mir_no_er_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_genmir_fpkm_type_er = np.vstack([input_gen_genmir_fpkm_type_er,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_TYPE + 'input_gen_genmir_fpkm_type_er.npy', input_gen_genmir_fpkm_type_er)
    print('input_gen_genmir_fpkm_type_er.npy is created')


    # 1.c. Gene (FPKM-UQ) ER classification
    input_gen_genmir_fpkmuq_type_er = np.empty((0,60483), int)
    for case in cases_gen_mir_no_er_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_genmir_fpkmuq_type_er = np.vstack([input_gen_genmir_fpkmuq_type_er,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_TYPE + 'input_gen_genmir_fpkmuq_type_er.npy', input_gen_genmir_fpkmuq_type_er)
    print('input_gen_genmir_fpkmuq_type_er.npy is created')


    # 2.a. Gene (count) PGR classification
    input_gen_genmir_count_type_pgr = np.empty((0,60483), int)
    for case in cases_gen_mir_no_pgr_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_genmir_count_type_pgr = np.vstack([input_gen_genmir_count_type_pgr,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_TYPE + 'input_gen_genmir_count_type_pgr.npy', input_gen_genmir_count_type_pgr)
    print('input_gen_genmir_count_type_pgr.npy is created')


    # 2.b. Gene (FPKM) PGR classification
    input_gen_genmir_fpkm_type_pgr = np.empty((0,60483), int)
    for case in cases_gen_mir_no_pgr_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_genmir_fpkm_type_pgr = np.vstack([input_gen_genmir_fpkm_type_pgr,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_TYPE + 'input_gen_genmir_fpkm_type_pgr.npy', input_gen_genmir_fpkm_type_pgr)
    print('input_gen_genmir_fpkm_type_pgr.npy is created')


    # 2.c. Gene (FPKM-UQ) PGR classification
    input_gen_genmir_fpkmuq_type_pgr = np.empty((0,60483), int)
    for case in cases_gen_mir_no_pgr_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_genmir_fpkmuq_type_pgr = np.vstack([input_gen_genmir_fpkmuq_type_pgr,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_TYPE + 'input_gen_genmir_fpkmuq_type_pgr.npy', input_gen_genmir_fpkmuq_type_pgr)
    print('input_gen_genmir_fpkmuq_type_pgr.npy is created')


    # 3.a. Gene (count) HER2 classification
    input_gen_genmir_count_type_her2 = np.empty((0,60483), int)
    for case in cases_gen_mir_no_her2_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_genmir_count_type_her2 = np.vstack([input_gen_genmir_count_type_her2,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_TYPE + 'input_gen_genmir_count_type_her2.npy', input_gen_genmir_count_type_her2)
    print('input_gen_genmir_count_type_her2.npy is created')


    # 3.b. Gene (FPKM) HER2 classification
    input_gen_genmir_fpkm_type_her2 = np.empty((0,60483), int)
    for case in cases_gen_mir_no_her2_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_genmir_fpkm_type_her2 = np.vstack([input_gen_genmir_fpkm_type_her2,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_TYPE + 'input_gen_genmir_fpkm_type_her2.npy', input_gen_genmir_fpkm_type_her2)
    print('input_gen_genmir_fpkm_type_her2.npy is created')


    # 3.c. Gene (FPKM-UQ) HER2 classification
    input_gen_genmir_fpkmuq_type_her2 = np.empty((0,60483), int)
    for case in cases_gen_mir_no_her2_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_genmir_fpkmuq_type_her2 = np.vstack([input_gen_genmir_fpkmuq_type_her2,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_TYPE + 'input_gen_genmir_fpkmuq_type_her2.npy', input_gen_genmir_fpkmuq_type_her2)
    print('input_gen_genmir_fpkmuq_type_her2.npy is created')


    # 4.a. Gene (count) universal classification
    input_gen_genmir_count_type_univ = np.empty((0,60483), int)
    for case in cases_gen_mir_no_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_genmir_count_type_univ = np.vstack([input_gen_genmir_count_type_univ,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_TYPE + 'input_gen_genmir_count_type_univ.npy', input_gen_genmir_count_type_univ)
    print('input_gen_genmir_count_type_univ.npy is created')


    # 4.b. Gene (FPKM) universal classification
    input_gen_genmir_fpkm_type_univ = np.empty((0,60483), int)
    for case in cases_gen_mir_no_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_genmir_fpkm_type_univ = np.vstack([input_gen_genmir_fpkm_type_univ,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_TYPE + 'input_gen_genmir_fpkm_type_univ.npy', input_gen_genmir_fpkm_type_univ)
    print('input_gen_genmir_fpkm_type_univ.npy is created')


    # 4.c. Gene (FPKM-UQ) universal classification
    input_gen_genmir_fpkmuq_type_univ = np.empty((0,60483), int)
    for case in cases_gen_mir_no_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_genmir_fpkmuq_type_univ = np.vstack([input_gen_genmir_fpkmuq_type_univ,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_TYPE + 'input_gen_genmir_fpkmuq_type_univ.npy', input_gen_genmir_fpkmuq_type_univ)
    print('input_gen_genmir_fpkmuq_type_univ.npy is created')



# Create the input (feature) set of miRNA expression data for cancer type classification
# implemented in mDBN of gene-miRNA
# We use the read count value of each miRNA
def input_mir_genmir_cancer_type():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_gen_mir_cli = np.intersect1d(cases_gen_cli,cases_mir) # [1071,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_pat_rec = np.genfromtxt(TARGET_CLINICAL + "pathology_receptor.csv", dtype=str, delimiter=',', skip_header=1)
    pat_rec = temp_pat_rec[temp_pat_rec[:,0].argsort()]

    cases_no_er_null = pat_rec[pat_rec[:,2]!="",0]     # [1048,]
    cases_no_pgr_null = pat_rec[pat_rec[:,4]!="",0]    # [1047,]
    cases_no_her2_null = pat_rec[pat_rec[:,7]!="",0]   # [919,]
    cases_no_null = np.intersect1d(np.intersect1d(cases_no_er_null,cases_no_pgr_null),cases_no_her2_null)   # [917,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_gen_mir_no_er_null = np.intersect1d(cases_gen_mir_cli,cases_no_er_null)       # (1024,)
    cases_gen_mir_no_pgr_null = np.intersect1d(cases_gen_mir_cli,cases_no_pgr_null)     # (1023,)
    cases_gen_mir_no_her2_null = np.intersect1d(cases_gen_mir_cli,cases_no_her2_null)   # (897,)
    cases_gen_mir_no_null = np.intersect1d(cases_gen_mir_cli,cases_no_null)             # (896,)
    
    
    ##############################################
    ##### INPUT MIRNA OF TYPE CLASSIFICATION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_MIR_GEN_MIR_TYPE)):
        os.makedirs(DATASET_INPUT_MIR_GEN_MIR_TYPE)

    data_mir = np.genfromtxt(TARGET_META_CSV + "mirna_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_mir[0]=='file_id')[0]
    file_name_column, = np.where(data_mir[0]=='file_name')[0]
    case_id_column, = np.where(data_mir[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_mir[0]=='cases.0.samples.0.sample_type')[0]

    data_mir = data_mir[1:]

    # 1. miRNA ER classification
    input_mir_genmir_type_er = np.empty((0,1881), float)
    for case in cases_gen_mir_no_er_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_genmir_type_er = np.vstack([input_mir_genmir_type_er,row])

    np.save(DATASET_INPUT_MIR_GEN_MIR_TYPE + 'input_mir_genmir_type_er.npy', input_mir_genmir_type_er)
    print('input_mir_genmir_type_er.npy is created')


    # 2. miRNA PGR classification
    input_mir_genmir_type_pgr = np.empty((0,1881), float)
    for case in cases_gen_mir_no_pgr_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_genmir_type_pgr = np.vstack([input_mir_genmir_type_pgr,row])

    np.save(DATASET_INPUT_MIR_GEN_MIR_TYPE + 'input_mir_genmir_type_pgr.npy', input_mir_genmir_type_pgr)
    print('input_mir_genmir_type_pgr.npy is created')


    # 3. miRNA HER2 classification
    input_mir_genmir_type_her2 = np.empty((0,1881), float)
    for case in cases_gen_mir_no_her2_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_genmir_type_her2 = np.vstack([input_mir_genmir_type_her2,row])

    np.save(DATASET_INPUT_MIR_GEN_MIR_TYPE + 'input_mir_genmir_type_her2.npy', input_mir_genmir_type_her2)
    print('input_mir_genmir_type_her2.npy is created')


    # 4. miRNA universal classification
    input_mir_genmir_type_univ = np.empty((0,1881), float)
    for case in cases_gen_mir_no_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_genmir_type_univ = np.vstack([input_mir_genmir_type_univ,row])

    np.save(DATASET_INPUT_MIR_GEN_MIR_TYPE + 'input_mir_genmir_type_univ.npy', input_mir_genmir_type_univ)
    print('input_mir_genmir_type_univ.npy is created')



# Create the input (feature) set of methylation data for cancer type classification
# implemented in mDBN of methylation-gene-miRNA
# available for both NCBI Platform GPL8490 and NCBI Platform GPL16304
# We use the beta value from CPG sites
def input_met_metgenmir_cancer_type():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_met = cases[cases[:,8]!="0",0]    # [1095,]
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_met_cli = np.intersect1d(cases_met,cases_cli) # [1095,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_met_gen_mir_cli = np.intersect1d(np.intersect1d(cases_met_cli,cases_gen),cases_mir) # [1069,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_pat_rec = np.genfromtxt(TARGET_CLINICAL + "pathology_receptor.csv", dtype=str, delimiter=',', skip_header=1)
    pat_rec = temp_pat_rec[temp_pat_rec[:,0].argsort()]

    cases_no_er_null = pat_rec[pat_rec[:,2]!="",0]     # [1048,]
    cases_no_pgr_null = pat_rec[pat_rec[:,4]!="",0]    # [1047,]
    cases_no_her2_null = pat_rec[pat_rec[:,7]!="",0]   # [919,]
    cases_no_null = np.intersect1d(np.intersect1d(cases_no_er_null,cases_no_pgr_null),cases_no_her2_null)   # [917,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_met_gen_mir_no_er_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_er_null)       # (1022,)
    cases_met_gen_mir_no_pgr_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_pgr_null)     # (1021,)
    cases_met_gen_mir_no_her2_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_her2_null)   # (895,)
    cases_met_gen_mir_no_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_null)             # (894,)
    
    
    ##############################################
    ## INPUT METHYLATION OF TYPE CLASSIFICATION ##
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_MET_MET_GEN_MIR_TYPE)):
        os.makedirs(DATASET_INPUT_MET_MET_GEN_MIR_TYPE)

    data_met = np.genfromtxt(TARGET_META_CSV + "methylation_beta_value.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_met[0]=='file_id')[0]
    file_name_column, = np.where(data_met[0]=='file_name')[0]
    case_id_column, = np.where(data_met[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_met[0]=='cases.0.samples.0.sample_type')[0]

    data_met = data_met[1:]

    with open(TARGET_METHYLATION + "cpg.json") as f:
        cpg = yaml.safe_load(f)

    with open(TARGET_METHYLATION + "cpg_in_cpg_short_idx.json") as f:
        cpg_in_cpg_short_idx = yaml.safe_load(f)

    with open(TARGET_METHYLATION + "cpg_in_cpg_long_idx.json") as f:
        cpg_in_cpg_long_idx = yaml.safe_load(f)


    # 1. Methylation ER classification
    input_met_metgenmir_type_er = np.empty((0,25978), float)
    for case in cases_met_gen_mir_no_er_null:
        file_id = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
            file_met = [row.split("\t") for row in f]

        file_met = file_met[1:]
        
        temp = np.empty((0,1), float)

        if len(file_met) == 27578:
            for idx in sorted(cpg_in_cpg_short_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])

                temp = np.append(temp, beta)

        elif len(file_met) == 485577:
            for idx in sorted(cpg_in_cpg_long_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])
                
                temp = np.append(temp, beta)
        
        print(case)
        input_met_metgenmir_type_er = np.vstack([input_met_metgenmir_type_er,temp])

    np.save(DATASET_INPUT_MET_MET_GEN_MIR_TYPE + 'input_met_metgenmir_type_er.npy', input_met_metgenmir_type_er)
    print('input_met_metgenmir_type_er.npy is created')


    # 2. Methylation PGR classification
    input_met_metgenmir_type_pgr = np.empty((0,25978), float)
    for case in cases_met_gen_mir_no_pgr_null:
        file_id = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
            file_met = [row.split("\t") for row in f]

        file_met = file_met[1:]
        
        temp = np.empty((0,1), float)

        if len(file_met) == 27578:
            for idx in sorted(cpg_in_cpg_short_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])

                temp = np.append(temp, beta)

        elif len(file_met) == 485577:
            for idx in sorted(cpg_in_cpg_long_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])
                
                temp = np.append(temp, beta)
        
        print(case)
        input_met_metgenmir_type_pgr = np.vstack([input_met_metgenmir_type_pgr,temp])

    np.save(DATASET_INPUT_MET_MET_GEN_MIR_TYPE + 'input_met_metgenmir_type_pgr.npy', input_met_metgenmir_type_pgr)
    print('input_met_metgenmir_type_pgr.npy is created')


    # 3. Methylation HER2 classification
    input_met_metgenmir_type_her2 = np.empty((0,25978), float)
    for case in cases_met_gen_mir_no_her2_null:
        file_id = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
            file_met = [row.split("\t") for row in f]

        file_met = file_met[1:]
        
        temp = np.empty((0,1), float)

        if len(file_met) == 27578:
            for idx in sorted(cpg_in_cpg_short_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])

                temp = np.append(temp, beta)

        elif len(file_met) == 485577:
            for idx in sorted(cpg_in_cpg_long_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])
                
                temp = np.append(temp, beta)
        
        print(case)
        input_met_metgenmir_type_her2 = np.vstack([input_met_metgenmir_type_her2,temp])

    np.save(DATASET_INPUT_MET_MET_GEN_MIR_TYPE + 'input_met_metgenmir_type_her2.npy', input_met_metgenmir_type_her2)
    print('input_met_metgenmir_type_her2.npy is created')


    # 4. Methylation universal classification
    input_met_metgenmir_type_univ = np.empty((0,25978), float)
    for case in cases_met_gen_mir_no_null:
        file_id = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
            file_met = [row.split("\t") for row in f]

        file_met = file_met[1:]
        
        temp = np.empty((0,1), float)

        if len(file_met) == 27578:
            for idx in sorted(cpg_in_cpg_short_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])

                temp = np.append(temp, beta)

        elif len(file_met) == 485577:
            for idx in sorted(cpg_in_cpg_long_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])
                
                temp = np.append(temp, beta)
        
        print(case)
        input_met_metgenmir_type_univ = np.vstack([input_met_metgenmir_type_univ,temp])

    np.save(DATASET_INPUT_MET_MET_GEN_MIR_TYPE + 'input_met_metgenmir_type_univ.npy', input_met_metgenmir_type_univ)
    print('input_met_metgenmir_type_univ.npy is created')



# Create the input (feature) set of gene expression for cancer type classification
# implemented in mDBN of methylation-gene-miRNA
# We use the expression value of each genes
# There are three different workflow analysis (HTSEC count, HTSEC FPKM, HTSEC FPKM-UQ) that was used to compute the expression value
# Based on this, we create input set for all 3 types of workflow analysis
def input_gen_metgenmir_cancer_type():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_met = cases[cases[:,8]!="0",0]    # [1095,]
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_met_cli = np.intersect1d(cases_met,cases_cli) # [1095,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_met_gen_mir_cli = np.intersect1d(np.intersect1d(cases_met_cli,cases_gen),cases_mir) # [1069,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_pat_rec = np.genfromtxt(TARGET_CLINICAL + "pathology_receptor.csv", dtype=str, delimiter=',', skip_header=1)
    pat_rec = temp_pat_rec[temp_pat_rec[:,0].argsort()]

    cases_no_er_null = pat_rec[pat_rec[:,2]!="",0]     # [1048,]
    cases_no_pgr_null = pat_rec[pat_rec[:,4]!="",0]    # [1047,]
    cases_no_her2_null = pat_rec[pat_rec[:,7]!="",0]   # [919,]
    cases_no_null = np.intersect1d(np.intersect1d(cases_no_er_null,cases_no_pgr_null),cases_no_her2_null)   # [917,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_met_gen_mir_no_er_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_er_null)       # (1022,)
    cases_met_gen_mir_no_pgr_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_pgr_null)     # (1021,)
    cases_met_gen_mir_no_her2_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_her2_null)   # (895,)
    cases_met_gen_mir_no_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_null)             # (894,)
    
    
    ##############################################
    ###### INPUT GENE OF TYPE CLASSIFICATION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE)):
        os.makedirs(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE)

    data_gene = np.genfromtxt(TARGET_META_CSV + "gene_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_gene[0]=='file_id')[0]
    file_name_column, = np.where(data_gene[0]=='file_name')[0]
    case_id_column, = np.where(data_gene[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_gene[0]=='cases.0.samples.0.sample_type')[0]
    workflow_column, = np.where(data_gene[0]=='analysis.workflow_type')[0]

    data_gene = data_gene[1:]

    # 1.a. Gene (count) ER classification
    input_gen_metgenmir_count_type_er = np.empty((0,60483), int)
    for case in cases_met_gen_mir_no_er_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_metgenmir_count_type_er = np.vstack([input_gen_metgenmir_count_type_er,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE + 'input_gen_metgenmir_count_type_er.npy', input_gen_metgenmir_count_type_er)
    print('input_gen_metgenmir_count_type_er.npy is created')


    # 1.b. Gene (FPKM) ER classification
    input_gen_metgenmir_fpkm_type_er = np.empty((0,60483), int)
    for case in cases_met_gen_mir_no_er_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metgenmir_fpkm_type_er = np.vstack([input_gen_metgenmir_fpkm_type_er,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE + 'input_gen_metgenmir_fpkm_type_er.npy', input_gen_metgenmir_fpkm_type_er)
    print('input_gen_metgenmir_fpkm_type_er.npy is created')


    # 1.c. Gene (FPKM-UQ) ER classification
    input_gen_metgenmir_fpkmuq_type_er = np.empty((0,60483), int)
    for case in cases_met_gen_mir_no_er_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metgenmir_fpkmuq_type_er = np.vstack([input_gen_metgenmir_fpkmuq_type_er,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE + 'input_gen_metgenmir_fpkmuq_type_er.npy', input_gen_metgenmir_fpkmuq_type_er)
    print('input_gen_metgenmir_fpkmuq_type_er.npy is created')


    # 2.a. Gene (count) PGR classification
    input_gen_metgenmir_count_type_pgr = np.empty((0,60483), int)
    for case in cases_met_gen_mir_no_pgr_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_metgenmir_count_type_pgr = np.vstack([input_gen_metgenmir_count_type_pgr,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE + 'input_gen_metgenmir_count_type_pgr.npy', input_gen_metgenmir_count_type_pgr)
    print('input_gen_metgenmir_count_type_pgr.npy is created')


    # 2.b. Gene (FPKM) PGR classification
    input_gen_metgenmir_fpkm_type_pgr = np.empty((0,60483), int)
    for case in cases_met_gen_mir_no_pgr_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metgenmir_fpkm_type_pgr = np.vstack([input_gen_metgenmir_fpkm_type_pgr,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE + 'input_gen_metgenmir_fpkm_type_pgr.npy', input_gen_metgenmir_fpkm_type_pgr)
    print('input_gen_metgenmir_fpkm_type_pgr.npy is created')


    # 2.c. Gene (FPKM-UQ) PGR classification
    input_gen_metgenmir_fpkmuq_type_pgr = np.empty((0,60483), int)
    for case in cases_met_gen_mir_no_pgr_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metgenmir_fpkmuq_type_pgr = np.vstack([input_gen_metgenmir_fpkmuq_type_pgr,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE + 'input_gen_metgenmir_fpkmuq_type_pgr.npy', input_gen_metgenmir_fpkmuq_type_pgr)
    print('input_gen_metgenmir_fpkmuq_type_pgr.npy is created')


    # 3.a. Gene (count) HER2 classification
    input_gen_metgenmir_count_type_her2 = np.empty((0,60483), int)
    for case in cases_met_gen_mir_no_her2_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_metgenmir_count_type_her2 = np.vstack([input_gen_metgenmir_count_type_her2,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE + 'input_gen_metgenmir_count_type_her2.npy', input_gen_metgenmir_count_type_her2)
    print('input_gen_metgenmir_count_type_her2.npy is created')


    # 3.b. Gene (FPKM) HER2 classification
    input_gen_metgenmir_fpkm_type_her2 = np.empty((0,60483), int)
    for case in cases_met_gen_mir_no_her2_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metgenmir_fpkm_type_her2 = np.vstack([input_gen_metgenmir_fpkm_type_her2,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE + 'input_gen_metgenmir_fpkm_type_her2.npy', input_gen_metgenmir_fpkm_type_her2)
    print('input_gen_metgenmir_fpkm_type_her2.npy is created')


    # 3.c. Gene (FPKM-UQ) HER2 classification
    input_gen_metgenmir_fpkmuq_type_her2 = np.empty((0,60483), int)
    for case in cases_met_gen_mir_no_her2_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metgenmir_fpkmuq_type_her2 = np.vstack([input_gen_metgenmir_fpkmuq_type_her2,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE + 'input_gen_metgenmir_fpkmuq_type_her2.npy', input_gen_metgenmir_fpkmuq_type_her2)
    print('input_gen_metgenmir_fpkmuq_type_her2.npy is created')


    # 4.a. Gene (count) universal classification
    input_gen_metgenmir_count_type_univ = np.empty((0,60483), int)
    for case in cases_met_gen_mir_no_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_metgenmir_count_type_univ = np.vstack([input_gen_metgenmir_count_type_univ,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE + 'input_gen_metgenmir_count_type_univ.npy', input_gen_metgenmir_count_type_univ)
    print('input_gen_metgenmir_count_type_univ.npy is created')


    # 4.b. Gene (FPKM) universal classification
    input_gen_metgenmir_fpkm_type_univ = np.empty((0,60483), int)
    for case in cases_met_gen_mir_no_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metgenmir_fpkm_type_univ = np.vstack([input_gen_metgenmir_fpkm_type_univ,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE + 'input_gen_metgenmir_fpkm_type_univ.npy', input_gen_metgenmir_fpkm_type_univ)
    print('input_gen_metgenmir_fpkm_type_univ.npy is created')


    # 4.c. Gene (FPKM-UQ) universal classification
    input_gen_metgenmir_fpkmuq_type_univ = np.empty((0,60483), int)
    for case in cases_met_gen_mir_no_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metgenmir_fpkmuq_type_univ = np.vstack([input_gen_metgenmir_fpkmuq_type_univ,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_TYPE + 'input_gen_metgenmir_fpkmuq_type_univ.npy', input_gen_metgenmir_fpkmuq_type_univ)
    print('input_gen_metgenmir_fpkmuq_type_univ.npy is created')



# Create the input (feature) set of miRNA expression data for cancer type classification
# implemented in mDBN of methylation-gene-miRNA
# We use the read count value of each miRNA
def input_mir_metgenmir_cancer_type():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_met = cases[cases[:,8]!="0",0]    # [1095,]
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_met_cli = np.intersect1d(cases_met,cases_cli) # [1095,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_met_gen_mir_cli = np.intersect1d(np.intersect1d(cases_met_cli,cases_gen),cases_mir) # [1069,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_pat_rec = np.genfromtxt(TARGET_CLINICAL + "pathology_receptor.csv", dtype=str, delimiter=',', skip_header=1)
    pat_rec = temp_pat_rec[temp_pat_rec[:,0].argsort()]

    cases_no_er_null = pat_rec[pat_rec[:,2]!="",0]     # [1048,]
    cases_no_pgr_null = pat_rec[pat_rec[:,4]!="",0]    # [1047,]
    cases_no_her2_null = pat_rec[pat_rec[:,7]!="",0]   # [919,]
    cases_no_null = np.intersect1d(np.intersect1d(cases_no_er_null,cases_no_pgr_null),cases_no_her2_null)   # [917,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_met_gen_mir_no_er_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_er_null)       # (1022,)
    cases_met_gen_mir_no_pgr_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_pgr_null)     # (1021,)
    cases_met_gen_mir_no_her2_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_her2_null)   # (895,)
    cases_met_gen_mir_no_null = np.intersect1d(cases_met_gen_mir_cli,cases_no_null)             # (894,)
    
    
    ##############################################
    ##### INPUT MIRNA OF TYPE CLASSIFICATION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_MIR_MET_GEN_MIR_TYPE)):
        os.makedirs(DATASET_INPUT_MIR_MET_GEN_MIR_TYPE)

    data_mir = np.genfromtxt(TARGET_META_CSV + "mirna_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_mir[0]=='file_id')[0]
    file_name_column, = np.where(data_mir[0]=='file_name')[0]
    case_id_column, = np.where(data_mir[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_mir[0]=='cases.0.samples.0.sample_type')[0]

    data_mir = data_mir[1:]

    # 1. miRNA ER classification
    input_mir_metgenmir_type_er = np.empty((0,1881), float)
    for case in cases_met_gen_mir_no_er_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_metgenmir_type_er = np.vstack([input_mir_metgenmir_type_er,row])

    np.save(DATASET_INPUT_MIR_MET_GEN_MIR_TYPE + 'input_mir_metgenmir_type_er.npy', input_mir_metgenmir_type_er)
    print('input_mir_metgenmir_type_er.npy is created')


    # 2. miRNA PGR classification
    input_mir_metgenmir_type_pgr = np.empty((0,1881), float)
    for case in cases_met_gen_mir_no_pgr_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_metgenmir_type_pgr = np.vstack([input_mir_metgenmir_type_pgr,row])

    np.save(DATASET_INPUT_MIR_MET_GEN_MIR_TYPE + 'input_mir_metgenmir_type_pgr.npy', input_mir_metgenmir_type_pgr)
    print('input_mir_metgenmir_type_pgr.npy is created')


    # 3. miRNA HER2 classification
    input_mir_metgenmir_type_her2 = np.empty((0,1881), float)
    for case in cases_met_gen_mir_no_her2_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_metgenmir_type_her2 = np.vstack([input_mir_metgenmir_type_her2,row])

    np.save(DATASET_INPUT_MIR_MET_GEN_MIR_TYPE + 'input_mir_metgenmir_type_her2.npy', input_mir_metgenmir_type_her2)
    print('input_mir_metgenmir_type_her2.npy is created')


    # 4. miRNA universal classification
    input_mir_metgenmir_type_univ = np.empty((0,1881), float)
    for case in cases_met_gen_mir_no_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_metgenmir_type_univ = np.vstack([input_mir_metgenmir_type_univ,row])

    np.save(DATASET_INPUT_MIR_MET_GEN_MIR_TYPE + 'input_mir_metgenmir_type_univ.npy', input_mir_metgenmir_type_univ)
    print('input_mir_metgenmir_type_univ.npy is created')



# Create the input (feature) set of methylation long data for cancer type classification
# implemented in mDBN of methylation_long-gene-miRNA
# only for NCBI Platform GPL16304
# We use the beta value from CPG sites
def input_metlong_metlonggenmir_cancer_type():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    with open(TARGET_METHYLATION + "cases_met_long.json") as f:
        temp = yaml.safe_load(f)
    cases_metlong = np.asarray(temp)  # [782,]
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_metlong_cli = np.intersect1d(cases_metlong,cases_cli) # [782,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_metlong_gen_mir_cli = np.intersect1d(np.intersect1d(cases_metlong_cli,cases_gen),cases_mir) # [768,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_pat_rec = np.genfromtxt(TARGET_CLINICAL + "pathology_receptor.csv", dtype=str, delimiter=',', skip_header=1)
    pat_rec = temp_pat_rec[temp_pat_rec[:,0].argsort()]

    cases_no_er_null = pat_rec[pat_rec[:,2]!="",0]     # [1048,]
    cases_no_pgr_null = pat_rec[pat_rec[:,4]!="",0]    # [1047,]
    cases_no_her2_null = pat_rec[pat_rec[:,7]!="",0]   # [919,]
    cases_no_null = np.intersect1d(np.intersect1d(cases_no_er_null,cases_no_pgr_null),cases_no_her2_null)   # [917,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_metlong_gen_mir_no_er_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_er_null)       # (726,)
    cases_metlong_gen_mir_no_pgr_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_pgr_null)     # (725,)
    cases_metlong_gen_mir_no_her2_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_her2_null)   # (632,)
    cases_metlong_gen_mir_no_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_null)             # (631,)
    
    
    #####################################################
    ### INPUT LONG METHYLATION OF TYPE CLASSIFICATION ###
    #####################################################
    if not(os.path.isdir(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE)):
        os.makedirs(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE)

    data_metlong = np.genfromtxt(TARGET_META_CSV + "methylation_long_beta_value.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_metlong[0]=='file_id')[0]
    file_name_column, = np.where(data_metlong[0]=='file_name')[0]
    case_id_column, = np.where(data_metlong[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_metlong[0]=='cases.0.samples.0.sample_type')[0]

    data_metlong = data_metlong[1:]

    # 1. Methylation ER classification
    # we divided the resulting file in 4 because the size is too large for 1 file
    for i in range(4):
        input_metlong_metlonggenmir_type_er = []
        for case in cases_metlong_gen_mir_no_er_null[200*i:200*(i+1)]:
            file_id = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
            file_name = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
            
            with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
                file_metlong = [row.split("\t") for row in f]

            file_metlong = file_metlong[1:]
            
            temp = []

            for j in range(len(file_metlong)):
                if file_metlong[j][1] == "NA":
                    beta = 0.
                else:
                    beta = float(file_metlong[j][1])
                temp.append(beta)
            
            input_metlong_metlonggenmir_type_er.append(temp)
            print(str(len(input_metlong_metlonggenmir_type_er)) + ". " + case)

        np.save(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_er_' + str(i) + '.npy', input_metlong_metlonggenmir_type_er)
        print('input_metlong_metlonggenmir_type_er_' + str(i) + '.npy is created')

    # combine it in 1 file
    temp_0 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_er_0.npy')
    temp_1 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_er_1.npy')
    temp_2 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_er_2.npy')
    temp_3 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_er_3.npy')
    input_metlong_metlonggenmir_type_er = np.concatenate((np.concatenate((temp_0, temp_1), axis=0),
                                                          np.concatenate((temp_2, temp_3), axis=0)),
                                                         axis=0)
    np.save(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_er.npy', input_metlong_metlonggenmir_type_er)

    # remove temp file
    for i in range(4):
        os.remove(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_er_' + str(i) + '.npy')


    # 2. Methylation PGR classification
    # we divided the resulting file in 4 because the size is too large for 1 file
    for i in range(4):
        input_metlong_metlonggenmir_type_pgr = []
        for case in cases_metlong_gen_mir_no_pgr_null[200*i:200*(i+1)]:
            file_id = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
            file_name = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
            
            with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
                file_metlong = [row.split("\t") for row in f]

            file_metlong = file_metlong[1:]
            
            temp = []

            for j in range(len(file_metlong)):
                if file_metlong[j][1] == "NA":
                    beta = 0.
                else:
                    beta = float(file_metlong[j][1])
                temp.append(beta)
            
            input_metlong_metlonggenmir_type_pgr.append(temp)
            print(str(len(input_metlong_metlonggenmir_type_pgr)) + ". " + case)

        np.save(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_pgr_' + str(i) + '.npy', input_metlong_metlonggenmir_type_pgr)
        print('input_metlong_metlonggenmir_type_pgr_' + str(i) + '.npy is created')

    # combine it in 1 file
    temp_0 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_pgr_0.npy')
    temp_1 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_pgr_1.npy')
    temp_2 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_pgr_2.npy')
    temp_3 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_pgr_3.npy')
    input_metlong_metlonggenmir_type_pgr = np.concatenate((np.concatenate((temp_0, temp_1), axis=0),
                                                           np.concatenate((temp_2, temp_3), axis=0)),
                                                          axis=0)
    np.save(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_pgr.npy', input_metlong_metlonggenmir_type_pgr)

    # remove temp file
    for i in range(4):
        os.remove(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_pgr_' + str(i) + '.npy')


    # 3. Methylation HER2 classification
    # we divided the resulting file in 4 because the size is too large for 1 file
    for i in range(4):
        input_metlong_metlonggenmir_type_her2 = []
        for case in cases_metlong_gen_mir_no_her2_null[200*i:200*(i+1)]:
            file_id = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
            file_name = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
            
            with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
                file_metlong = [row.split("\t") for row in f]

            file_metlong = file_metlong[1:]
            
            temp = []

            for j in range(len(file_metlong)):
                if file_metlong[j][1] == "NA":
                    beta = 0.
                else:
                    beta = float(file_metlong[j][1])
                temp.append(beta)
            
            input_metlong_metlonggenmir_type_her2.append(temp)
            print(str(len(input_metlong_metlonggenmir_type_her2)) + ". " + case)

        np.save(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_her2_' + str(i) + '.npy', input_metlong_metlonggenmir_type_her2)
        print('input_metlong_metlonggenmir_type_her2_' + str(i) + '.npy is created')

    # combine it in 1 file
    temp_0 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_her2_0.npy')
    temp_1 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_her2_1.npy')
    temp_2 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_her2_2.npy')
    temp_3 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_her2_3.npy')
    input_metlong_metlonggenmir_type_her2 = np.concatenate((np.concatenate((temp_0, temp_1), axis=0),
                                                            np.concatenate((temp_2, temp_3), axis=0)),
                                                           axis=0)
    np.save(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_her2.npy', input_metlong_metlonggenmir_type_her2)

    # remove temp file
    for i in range(4):
        os.remove(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_her2_' + str(i) + '.npy')


    # 4. Methylation universal classification
    # we divided the resulting file in 4 because the size is too large for 1 file
    for i in range(4):
        input_metlong_metlonggenmir_type_univ = []
        for case in cases_metlong_gen_mir_no_null[200*i:200*(i+1)]:
            file_id = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
            file_name = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
            
            with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
                file_metlong = [row.split("\t") for row in f]

            file_metlong = file_metlong[1:]
            
            temp = []

            for j in range(len(file_metlong)):
                if file_metlong[j][1] == "NA":
                    beta = 0.
                else:
                    beta = float(file_metlong[j][1])
                temp.append(beta)
            
            input_metlong_metlonggenmir_type_univ.append(temp)
            print(str(len(input_metlong_metlonggenmir_type_univ)) + ". " + case)

        np.save(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_univ_' + str(i) + '.npy', input_metlong_metlonggenmir_type_univ)
        print('input_metlong_metlonggenmir_type_univ_' + str(i) + 'npy is created')

    # combine it in 1 file
    temp_0 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_univ_0.npy')
    temp_1 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_univ_1.npy')
    temp_2 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_univ_2.npy')
    temp_3 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_univ_3.npy')
    input_metlong_metlonggenmir_type_univ = np.concatenate((np.concatenate((temp_0, temp_1), axis=0),
                                                            np.concatenate((temp_2, temp_3), axis=0)),
                                                           axis=0)
    np.save(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_univ.npy', input_metlong_metlonggenmir_type_univ)

    # remove temp file
    for i in range(4):
        os.remove(DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE + 'input_metlong_metlonggenmir_type_univ_' + str(i) + '.npy')



# Create the input (feature) set of gene expression for cancer type classification
# implemented in mDBN of methylation_long-gene-miRNA
# We use the expression value of each genes
# There are three different workflow analysis (HTSEC count, HTSEC FPKM, HTSEC FPKM-UQ) that was used to compute the expression value
# Based on this, we create input set for all 3 types of workflow analysis
def input_gen_metlonggenmir_cancer_type():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    with open(TARGET_METHYLATION + "cases_met_long.json") as f:
        temp = yaml.safe_load(f)
    cases_metlong = np.asarray(temp)  # [782,]
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_metlong_cli = np.intersect1d(cases_metlong,cases_cli) # [782,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_metlong_gen_mir_cli = np.intersect1d(np.intersect1d(cases_metlong_cli,cases_gen),cases_mir) # [768,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_pat_rec = np.genfromtxt(TARGET_CLINICAL + "pathology_receptor.csv", dtype=str, delimiter=',', skip_header=1)
    pat_rec = temp_pat_rec[temp_pat_rec[:,0].argsort()]

    cases_no_er_null = pat_rec[pat_rec[:,2]!="",0]     # [1048,]
    cases_no_pgr_null = pat_rec[pat_rec[:,4]!="",0]    # [1047,]
    cases_no_her2_null = pat_rec[pat_rec[:,7]!="",0]   # [919,]
    cases_no_null = np.intersect1d(np.intersect1d(cases_no_er_null,cases_no_pgr_null),cases_no_her2_null)   # [917,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_metlong_gen_mir_no_er_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_er_null)       # (726,)
    cases_metlong_gen_mir_no_pgr_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_pgr_null)     # (725,)
    cases_metlong_gen_mir_no_her2_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_her2_null)   # (632,)
    cases_metlong_gen_mir_no_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_null)             # (631,)
    
    
    ##############################################
    ###### INPUT GENE OF TYPE CLASSIFICATION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE)):
        os.makedirs(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE)

    data_gene = np.genfromtxt(TARGET_META_CSV + "gene_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_gene[0]=='file_id')[0]
    file_name_column, = np.where(data_gene[0]=='file_name')[0]
    case_id_column, = np.where(data_gene[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_gene[0]=='cases.0.samples.0.sample_type')[0]
    workflow_column, = np.where(data_gene[0]=='analysis.workflow_type')[0]

    data_gene = data_gene[1:]

    # 1.a. Gene (count) ER classification
    input_gen_metlonggenmir_count_type_er = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_no_er_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_metlonggenmir_count_type_er = np.vstack([input_gen_metlonggenmir_count_type_er,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE + 'input_gen_metlonggenmir_count_type_er.npy', input_gen_metlonggenmir_count_type_er)
    print('input_gen_metlonggenmir_count_type_er.npy is created')


    # 1.b. Gene (FPKM) ER classification
    input_gen_metlonggenmir_fpkm_type_er = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_no_er_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metlonggenmir_fpkm_type_er = np.vstack([input_gen_metlonggenmir_fpkm_type_er,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE + 'input_gen_metlonggenmir_fpkm_type_er.npy', input_gen_metlonggenmir_fpkm_type_er)
    print('input_gen_metlonggenmir_fpkm_type_er.npy is created')


    # 1.c. Gene (FPKM-UQ) ER classification
    input_gen_metlonggenmir_fpkmuq_type_er = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_no_er_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metlonggenmir_fpkmuq_type_er = np.vstack([input_gen_metlonggenmir_fpkmuq_type_er,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE + 'input_gen_metlonggenmir_fpkmuq_type_er.npy', input_gen_metlonggenmir_fpkmuq_type_er)
    print('input_gen_metlonggenmir_fpkmuq_type_er.npy is created')


    # 2.a. Gene (count) PGR classification
    input_gen_metlonggenmir_count_type_pgr = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_no_pgr_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_metlonggenmir_count_type_pgr = np.vstack([input_gen_metlonggenmir_count_type_pgr,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE + 'input_gen_metlonggenmir_count_type_pgr.npy', input_gen_metlonggenmir_count_type_pgr)
    print('input_gen_metlonggenmir_count_type_pgr.npy is created')


    # 2.b. Gene (FPKM) PGR classification
    input_gen_metlonggenmir_fpkm_type_pgr = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_no_pgr_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metlonggenmir_fpkm_type_pgr = np.vstack([input_gen_metlonggenmir_fpkm_type_pgr,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE + 'input_gen_metlonggenmir_fpkm_type_pgr.npy', input_gen_metlonggenmir_fpkm_type_pgr)
    print('input_gen_metlonggenmir_fpkm_type_pgr.npy is created')


    # 2.c. Gene (FPKM-UQ) PGR classification
    input_gen_metlonggenmir_fpkmuq_type_pgr = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_no_pgr_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metlonggenmir_fpkmuq_type_pgr = np.vstack([input_gen_metlonggenmir_fpkmuq_type_pgr,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE + 'input_gen_metlonggenmir_fpkmuq_type_pgr.npy', input_gen_metlonggenmir_fpkmuq_type_pgr)
    print('input_gen_metlonggenmir_fpkmuq_type_pgr.npy is created')


    # 3.a. Gene (count) HER2 classification
    input_gen_metlonggenmir_count_type_her2 = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_no_her2_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_metlonggenmir_count_type_her2 = np.vstack([input_gen_metlonggenmir_count_type_her2,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE + 'input_gen_metlonggenmir_count_type_her2.npy', input_gen_metlonggenmir_count_type_her2)
    print('input_gen_metlonggenmir_count_type_her2.npy is created')


    # 3.b. Gene (FPKM) HER2 classification
    input_gen_metlonggenmir_fpkm_type_her2 = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_no_her2_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metlonggenmir_fpkm_type_her2 = np.vstack([input_gen_metlonggenmir_fpkm_type_her2,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE + 'input_gen_metlonggenmir_fpkm_type_her2.npy', input_gen_metlonggenmir_fpkm_type_her2)
    print('input_gen_metlonggenmir_fpkm_type_her2.npy is created')


    # 3.c. Gene (FPKM-UQ) HER2 classification
    input_gen_metlonggenmir_fpkmuq_type_her2 = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_no_her2_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metlonggenmir_fpkmuq_type_her2 = np.vstack([input_gen_metlonggenmir_fpkmuq_type_her2,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE + 'input_gen_metlonggenmir_fpkmuq_type_her2.npy', input_gen_metlonggenmir_fpkmuq_type_her2)
    print('input_gen_metlonggenmir_fpkmuq_type_her2.npy is created')


    # 4.a. Gene (count) universal classification
    input_gen_metlonggenmir_count_type_univ = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_no_null:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_metlonggenmir_count_type_univ = np.vstack([input_gen_metlonggenmir_count_type_univ,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE + 'input_gen_metlonggenmir_count_type_univ.npy', input_gen_metlonggenmir_count_type_univ)
    print('input_gen_metlonggenmir_count_type_univ.npy is created')


    # 4.b. Gene (FPKM) universal classification
    input_gen_metlonggenmir_fpkm_type_univ = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_no_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metlonggenmir_fpkm_type_univ = np.vstack([input_gen_metlonggenmir_fpkm_type_univ,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE + 'input_gen_metlonggenmir_fpkm_type_univ.npy', input_gen_metlonggenmir_fpkm_type_univ)
    print('input_gen_metlonggenmir_fpkm_type_univ.npy is created')


    # 4.c. Gene (FPKM-UQ) universal classification
    input_gen_metlonggenmir_fpkmuq_type_univ = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_no_null:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metlonggenmir_fpkmuq_type_univ = np.vstack([input_gen_metlonggenmir_fpkmuq_type_univ,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE + 'input_gen_fpkmuq_type_univ.npy', input_gen_metlonggenmir_fpkmuq_type_univ)
    print('input_gen_metlonggenmir_fpkmuq_type_univ.npy is created')



# Create the input (feature) set of miRNA expression data for cancer type classification
# implemented in mDBN of methylation_long-gene-miRNA
# We use the read count value of each miRNA
def input_mir_metlonggenmir_cancer_type():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    with open(TARGET_METHYLATION + "cases_met_long.json") as f:
        temp = yaml.safe_load(f)
    cases_metlong = np.asarray(temp)  # [782,]
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_metlong_cli = np.intersect1d(cases_metlong,cases_cli) # [782,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_metlong_gen_mir_cli = np.intersect1d(np.intersect1d(cases_metlong_cli,cases_gen),cases_mir) # [768,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_pat_rec = np.genfromtxt(TARGET_CLINICAL + "pathology_receptor.csv", dtype=str, delimiter=',', skip_header=1)
    pat_rec = temp_pat_rec[temp_pat_rec[:,0].argsort()]

    cases_no_er_null = pat_rec[pat_rec[:,2]!="",0]     # [1048,]
    cases_no_pgr_null = pat_rec[pat_rec[:,4]!="",0]    # [1047,]
    cases_no_her2_null = pat_rec[pat_rec[:,7]!="",0]   # [919,]
    cases_no_null = np.intersect1d(np.intersect1d(cases_no_er_null,cases_no_pgr_null),cases_no_her2_null)   # [917,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_metlong_gen_mir_no_er_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_er_null)       # (726,)
    cases_metlong_gen_mir_no_pgr_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_pgr_null)     # (725,)
    cases_metlong_gen_mir_no_her2_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_her2_null)   # (632,)
    cases_metlong_gen_mir_no_null = np.intersect1d(cases_metlong_gen_mir_cli,cases_no_null)             # (631,)
    
    
    ##############################################
    ##### INPUT MIRNA OF TYPE CLASSIFICATION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_MIR_METLONG_GEN_MIR_TYPE)):
        os.makedirs(DATASET_INPUT_MIR_METLONG_GEN_MIR_TYPE)

    data_mir = np.genfromtxt(TARGET_META_CSV + "mirna_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_mir[0]=='file_id')[0]
    file_name_column, = np.where(data_mir[0]=='file_name')[0]
    case_id_column, = np.where(data_mir[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_mir[0]=='cases.0.samples.0.sample_type')[0]

    data_mir = data_mir[1:]

    # 1. miRNA ER classification
    input_mir_metlonggenmir_type_er = np.empty((0,1881), float)
    for case in cases_metlong_gen_mir_no_er_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_metlonggenmir_type_er = np.vstack([input_mir_metlonggenmir_type_er,row])

    np.save(DATASET_INPUT_MIR_METLONG_GEN_MIR_TYPE + 'input_mir_metlonggenmir_type_er.npy', input_mir_metlonggenmir_type_er)
    print('input_mir_metlonggenmir_type_er.npy is created')


    # 2. miRNA PGR classification
    input_mir_metlonggenmir_type_pgr = np.empty((0,1881), float)
    for case in cases_metlong_gen_mir_no_pgr_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_metlonggenmir_type_pgr = np.vstack([input_mir_metlonggenmir_type_pgr,row])

    np.save(DATASET_INPUT_MIR_METLONG_GEN_MIR_TYPE + 'input_mir_metlonggenmir_type_pgr.npy', input_mir_metlonggenmir_type_pgr)
    print('input_mir_metlonggenmir_type_pgr.npy is created')


    # 3. miRNA HER2 classification
    input_mir_metlonggenmir_type_her2 = np.empty((0,1881), float)
    for case in cases_metlong_gen_mir_no_her2_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_metlonggenmir_type_her2 = np.vstack([input_mir_metlonggenmir_type_her2,row])

    np.save(DATASET_INPUT_MIR_METLONG_GEN_MIR_TYPE + 'input_mir_metlonggenmir_type_her2.npy', input_mir_metlonggenmir_type_her2)
    print('input_mir_metlonggenmir_type_her2.npy is created')


    # 4. miRNA universal classification
    input_mir_metlonggenmir_type_univ = np.empty((0,1881), float)
    for case in cases_metlong_gen_mir_no_null:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_metlonggenmir_type_univ = np.vstack([input_mir_metlonggenmir_type_univ,row])

    np.save(DATASET_INPUT_MIR_METLONG_GEN_MIR_TYPE + 'input_mir_metlonggenmir_type_univ.npy', input_mir_metlonggenmir_type_univ)
    print('input_mir_metlonggenmir_type_univ.npy is created')



# Create the label set of survival rate regression
def label_survival(dataset=3):
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_all = cases[:,0]                  # [1098,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    
    if (dataset==1) or (dataset==5):
        cases_met = cases[cases[:,8]!="0",0]    # [1095,]
        with open(TARGET_METHYLATION + "cases_met_long.json") as f:
            temp = yaml.safe_load(f)
        cases_metlong = np.asarray(temp)  # [782,]
        cases_met_cli = np.intersect1d(cases_met,cases_cli) # [1095,]
        cases_metlong_cli = np.intersect1d(cases_metlong,cases_cli) # [782,]
    
    if (dataset==2) or (dataset==4) or (dataset==5):
        cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
        # remove cases in cases_gen where there are no tumor sample
        cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
        cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    
    if (dataset==3) or (dataset==4) or (dataset==5):
        cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
        # remove cases in cases_mir where there are no tumor sample
        cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
        cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    
    if (dataset==4) or (dataset==5):
        cases_gen_mir_cli = np.intersect1d(cases_gen_cli,cases_mir) # [1071,]
    
    if (dataset==5):
        cases_met_gen_mir_cli = np.intersect1d(np.intersect1d(cases_met_cli,cases_gen),cases_mir) # [1069,]
        cases_metlong_gen_mir_cli = np.intersect1d(np.intersect1d(cases_metlong_cli,cases_gen),cases_mir) # [768,]


    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_sur = np.genfromtxt(TARGET_CLINICAL + "survival_plot.tsv", dtype=str, delimiter='\t', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    case_id_column, = np.where(temp_sur[0]=='id')[0]
    survival_estimate_column, = np.where(temp_sur[0]=='survivalEstimate')[0]

    temp_sur = temp_sur[1:]

    sur = temp_sur[temp_sur[:,case_id_column].argsort()]

    cases_sur = sur[:,case_id_column]     # [1084,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    if (dataset==1) or (dataset==5):
        cases_met_sur = np.intersect1d(cases_met_cli,cases_sur) # (1082,)
        cases_metlong_sur = np.intersect1d(cases_metlong_cli,cases_sur) # (781,)

    if (dataset==2) or (dataset==4) or (dataset==5):
        cases_gen_sur = np.intersect1d(cases_gen_cli,cases_sur) # (1077,)

    if (dataset==3) or (dataset==4) or (dataset==5):
        cases_mir_sur = np.intersect1d(cases_mir_cli,cases_sur) # (1064,)

    if (dataset==4) or (dataset==5):
        cases_gen_mir_sur = np.intersect1d(cases_gen_mir_cli,cases_sur) # (1058,)

    if (dataset==5):
        cases_met_gen_mir_sur = np.intersect1d(cases_met_gen_mir_cli,cases_sur) # (1056,)
        cases_metlong_gen_mir_sur = np.intersect1d(cases_metlong_gen_mir_cli,cases_sur) # (767,)
    
    
    ######################################
    # LABELS OF SURVIVAL RATE REGRESSION #
    ######################################
    if (dataset==1) or (dataset==5):
        label_sur_met = sur[np.logical_or.reduce([sur[:,0] == i for i in cases_met_sur]),survival_estimate_column]
        label_sur_met = label_sur_met.astype(np.float)
        label_sur_metlong = sur[np.logical_or.reduce([sur[:,0] == i for i in cases_metlong_sur]),survival_estimate_column]
        label_sur_metlong = label_sur_metlong.astype(np.float)
    
    if (dataset==2) or (dataset==4) or (dataset==5):
        label_sur_gen = sur[np.logical_or.reduce([sur[:,0] == i for i in cases_gen_sur]),survival_estimate_column]
        label_sur_gen = label_sur_gen.astype(np.float)
    
    if (dataset==3) or (dataset==4) or (dataset==5):
        label_sur_mir = sur[np.logical_or.reduce([sur[:,0] == i for i in cases_mir_sur]),survival_estimate_column]
        label_sur_mir = label_sur_mir.astype(np.float)
    
    if (dataset==4) or (dataset==5):
        label_sur_gen_mir = sur[np.logical_or.reduce([sur[:,0] == i for i in cases_gen_mir_sur]),survival_estimate_column]
        label_sur_gen_mir = label_sur_gen_mir.astype(np.float)
    
    if (dataset==5):
        label_sur_met_gen_mir = sur[np.logical_or.reduce([sur[:,0] == i for i in cases_met_gen_mir_sur]),survival_estimate_column]
        label_sur_metlong_gen_mir = sur[np.logical_or.reduce([sur[:,0] == i for i in cases_metlong_gen_mir_sur]),survival_estimate_column]
        label_sur_met_gen_mir = label_sur_met_gen_mir.astype(np.float)
        label_sur_metlong_gen_mir = label_sur_metlong_gen_mir.astype(np.float)

        
    ###########################################
    # SAVE LABELS OF SURVIVAL RATE REGRESSION #
    ###########################################
    if (dataset==1) or (dataset==5):
        if not os.path.isdir(DATASET_LABELS_MET_SURVIVAL):
            os.makedirs(DATASET_LABELS_MET_SURVIVAL)
        os.chdir(DATASET_LABELS_MET_SURVIVAL)
        np.save('label_sur_met.npy', label_sur_met)

        if not os.path.isdir(DATASET_LABELS_METLONG_SURVIVAL):
            os.makedirs(DATASET_LABELS_METLONG_SURVIVAL)
        os.chdir(DATASET_LABELS_METLONG_SURVIVAL)
        np.save('label_sur_metlong.npy', label_sur_metlong)
    
    if (dataset==2) or (dataset==4) or (dataset==5):
        if not os.path.isdir(DATASET_LABELS_GEN_SURVIVAL):
            os.makedirs(DATASET_LABELS_GEN_SURVIVAL)
        os.chdir(DATASET_LABELS_GEN_SURVIVAL)
        np.save('label_sur_gen.npy', label_sur_gen)
    
    if (dataset==3) or (dataset==4) or (dataset==5):
        if not os.path.isdir(DATASET_LABELS_MIR_SURVIVAL):
            os.makedirs(DATASET_LABELS_MIR_SURVIVAL)
        os.chdir(DATASET_LABELS_MIR_SURVIVAL)
        np.save('label_sur_mir.npy', label_sur_mir)
    
    if (dataset==4) or (dataset==5):
        if not os.path.isdir(DATASET_LABELS_GEN_MIR_SURVIVAL):
            os.makedirs(DATASET_LABELS_GEN_MIR_SURVIVAL)
        os.chdir(DATASET_LABELS_GEN_MIR_SURVIVAL)
        np.save('label_sur_gen_mir.npy', label_sur_gen_mir)
    
    if (dataset==5):
        if not os.path.isdir(DATASET_LABELS_MET_GEN_MIR_SURVIVAL):
            os.makedirs(DATASET_LABELS_MET_GEN_MIR_SURVIVAL)
        os.chdir(DATASET_LABELS_MET_GEN_MIR_SURVIVAL)
        np.save('label_sur_met_gen_mir.npy', label_sur_met_gen_mir)
        
        if not os.path.isdir(DATASET_LABELS_METLONG_GEN_MIR_SURVIVAL):
            os.makedirs(DATASET_LABELS_METLONG_GEN_MIR_SURVIVAL)
        os.chdir(DATASET_LABELS_METLONG_GEN_MIR_SURVIVAL)
        np.save('label_sur_metlong_gen_mir.npy', label_sur_metlong_gen_mir)

    print("all survival labels are created")
    


# Create the input (feature) set of methylation data for survival rate regression
# available for both NCBI Platform GPL8490 and NCBI Platform GPL16304
# We use the beta value from CPG sites
def input_met_survival():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_met = cases[cases[:,8]!="0",0]    # [1095,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_met_cli = np.intersect1d(cases_met,cases_cli) # [1095,]


    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_sur = np.genfromtxt(TARGET_CLINICAL + "survival_plot.tsv", dtype=str, delimiter='\t', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    case_id_column, = np.where(temp_sur[0]=='id')[0]

    temp_sur = temp_sur[1:]

    sur = temp_sur[temp_sur[:,case_id_column].argsort()]

    cases_sur = sur[:,case_id_column]     # [1084,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_met_sur = np.intersect1d(cases_met_cli,cases_sur) # (1082,)
    
    
    ##############################################
    ## INPUT METHYLATION OF SURVIVAL REGRESSION ##
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_MET_SURVIVAL)):
        os.makedirs(DATASET_INPUT_MET_SURVIVAL)

    data_met = np.genfromtxt(TARGET_META_CSV + "methylation_beta_value.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_met[0]=='file_id')[0]
    file_name_column, = np.where(data_met[0]=='file_name')[0]
    case_id_column, = np.where(data_met[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_met[0]=='cases.0.samples.0.sample_type')[0]

    data_met = data_met[1:]

    with open(TARGET_METHYLATION + "cpg.json") as f:
        cpg = yaml.safe_load(f)

    with open(TARGET_METHYLATION + "cpg_in_cpg_short_idx.json") as f:
        cpg_in_cpg_short_idx = yaml.safe_load(f)

    with open(TARGET_METHYLATION + "cpg_in_cpg_long_idx.json") as f:
        cpg_in_cpg_long_idx = yaml.safe_load(f)


    input_met_sur = np.empty((0,25978), float)
    for case in cases_met_sur:
        file_id = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
            file_met = [row.split("\t") for row in f]

        file_met = file_met[1:]
        
        temp = np.empty((0,1), float)

        if len(file_met) == 27578:
            for idx in sorted(cpg_in_cpg_short_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])

                temp = np.append(temp, beta)

        elif len(file_met) == 485577:
            for idx in sorted(cpg_in_cpg_long_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])
                
                temp = np.append(temp, beta)
        
        print(case)
        input_met_sur = np.vstack([input_met_sur,temp])

    np.save(DATASET_INPUT_MET_SURVIVAL + 'input_met_sur.npy', input_met_sur)
    print('input_met_sur.npy is created')



# Create the input (feature) set of methylation long data for survival rate regression
# only for NCBI Platform GPL16304
# We use the beta value from CPG sites
def input_metlong_survival():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    with open(TARGET_METHYLATION + "cases_met_long.json") as f:
        temp = yaml.safe_load(f)
    cases_metlong = np.asarray(temp)  # [782,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_metlong_cli = np.intersect1d(cases_metlong,cases_cli) # [782,]


    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_sur = np.genfromtxt(TARGET_CLINICAL + "survival_plot.tsv", dtype=str, delimiter='\t', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    case_id_column, = np.where(temp_sur[0]=='id')[0]

    temp_sur = temp_sur[1:]

    sur = temp_sur[temp_sur[:,case_id_column].argsort()]

    cases_sur = sur[:,case_id_column]     # [1084,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_metlong_sur = np.intersect1d(cases_metlong_cli,cases_sur) # (781,)
    
    
    #####################################################
    ### INPUT LONG METHYLATION OF SURVIVAL REGRESSION ###
    #####################################################
    if not(os.path.isdir(DATASET_INPUT_METLONG_SURVIVAL)):
        os.makedirs(DATASET_INPUT_METLONG_SURVIVAL)

    data_metlong = np.genfromtxt(TARGET_META_CSV + "methylation_long_beta_value.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_metlong[0]=='file_id')[0]
    file_name_column, = np.where(data_metlong[0]=='file_name')[0]
    case_id_column, = np.where(data_metlong[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_metlong[0]=='cases.0.samples.0.sample_type')[0]

    data_metlong = data_metlong[1:]

    # we divided the resulting file in 4 because the size is too large for 1 file
    for i in range(4):
        input_metlong_sur = []
        for case in cases_metlong_sur[200*i:200*(i+1)]:
            file_id = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
            file_name = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
            
            with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
                file_metlong = [row.split("\t") for row in f]

            file_metlong = file_metlong[1:]
            
            temp = []

            for j in range(len(file_metlong)):
                if file_metlong[j][1] == "NA":
                    beta = 0.
                else:
                    beta = float(file_metlong[j][1])
                temp.append(beta)
            
            input_metlong_sur.append(temp)
            print(str(len(input_metlong_sur)) + ". " + case)

        np.save(DATASET_INPUT_METLONG_SURVIVAL + 'input_metlong_sur_' + str(i) + '.npy', input_metlong_sur)
        print('input_metlong_sur_' + str(i) + '.npy is created')

    # combine it in 1 file
    temp_0 = np.load(DATASET_INPUT_METLONG_SURVIVAL + 'input_metlong_sur_0.npy')
    temp_1 = np.load(DATASET_INPUT_METLONG_SURVIVAL + 'input_metlong_sur_1.npy')
    temp_2 = np.load(DATASET_INPUT_METLONG_SURVIVAL + 'input_metlong_sur_2.npy')
    temp_3 = np.load(DATASET_INPUT_METLONG_SURVIVAL + 'input_metlong_sur_3.npy')
    input_metlong_sur = np.concatenate((np.concatenate((temp_0, temp_1), axis=0),
                                        np.concatenate((temp_2, temp_3), axis=0)),
                                       axis=0)
    np.save(DATASET_INPUT_METLONG_SURVIVAL + 'input_metlong_sur.npy', input_metlong_sur)

    # remove temp file
    for i in range(4):
        os.remove(DATASET_INPUT_METLONG_SURVIVAL + 'input_metlong_sur_' + str(i) + '.npy')



# Create the input (feature) set of gene expression for survival rate regression
# We use the expression value of each genes
# There are three different workflow analysis (HTSEC count, HTSEC FPKM, HTSEC FPKM-UQ) that was used to compute the expression value
# Based on this, we create input set for all 3 types of workflow analysis
def input_gen_survival():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]


    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_sur = np.genfromtxt(TARGET_CLINICAL + "survival_plot.tsv", dtype=str, delimiter='\t', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    case_id_column, = np.where(temp_sur[0]=='id')[0]

    temp_sur = temp_sur[1:]

    sur = temp_sur[temp_sur[:,case_id_column].argsort()]

    cases_sur = sur[:,case_id_column]     # [1084,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_gen_sur = np.intersect1d(cases_gen_cli,cases_sur) # (1077,)
    
    
    ##############################################
    ###### INPUT GENE OF SURVIVAL REGRESSION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_GEN_SURVIVAL)):
        os.makedirs(DATASET_INPUT_GEN_SURVIVAL)

    data_gene = np.genfromtxt(TARGET_META_CSV + "gene_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_gene[0]=='file_id')[0]
    file_name_column, = np.where(data_gene[0]=='file_name')[0]
    case_id_column, = np.where(data_gene[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_gene[0]=='cases.0.samples.0.sample_type')[0]
    workflow_column, = np.where(data_gene[0]=='analysis.workflow_type')[0]

    data_gene = data_gene[1:]

    # 1. Gene (count) survival regression
    input_gen_count_sur = np.empty((0,60483), int)
    for case in cases_gen_sur:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_count_sur = np.vstack([input_gen_count_sur,row])

    np.save(DATASET_INPUT_GEN_SURVIVAL + 'input_gen_count_sur.npy', input_gen_count_sur)
    print('input_gen_count_sur.npy is created')


    # 2. Gene (FPKM) survival regression
    input_gen_fpkm_sur = np.empty((0,60483), int)
    for case in cases_gen_sur:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_fpkm_sur = np.vstack([input_gen_fpkm_sur,row])

    np.save(DATASET_INPUT_GEN_SURVIVAL + 'input_gen_fpkm_sur.npy', input_gen_fpkm_sur)
    print('input_gen_fpkm_sur.npy is created')


    # 3. Gene (FPKM-UQ) survival regression
    input_gen_fpkmuq_sur = np.empty((0,60483), int)
    for case in cases_gen_sur:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_fpkmuq_sur = np.vstack([input_gen_fpkmuq_sur,row])

    np.save(DATASET_INPUT_GEN_SURVIVAL + 'input_gen_fpkmuq_sur.npy', input_gen_fpkmuq_sur)
    print('input_gen_fpkmuq_sur.npy is created')



# Create the input (feature) set of miRNA expression data for survival rate regression
# We use the read count value of each miRNA
def input_mir_survival():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_sur = np.genfromtxt(TARGET_CLINICAL + "survival_plot.tsv", dtype=str, delimiter='\t', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    case_id_column, = np.where(temp_sur[0]=='id')[0]

    temp_sur = temp_sur[1:]

    sur = temp_sur[temp_sur[:,case_id_column].argsort()]

    cases_sur = sur[:,case_id_column]     # [1084,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_mir_sur = np.intersect1d(cases_mir_cli,cases_sur) # (1064,)
    
    
    ##############################################
    ##### INPUT MIRNA OF TYPE CLASSIFICATION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_MIR_SURVIVAL)):
        os.makedirs(DATASET_INPUT_MIR_SURVIVAL)

    data_mir = np.genfromtxt(TARGET_META_CSV + "mirna_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_mir[0]=='file_id')[0]
    file_name_column, = np.where(data_mir[0]=='file_name')[0]
    case_id_column, = np.where(data_mir[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_mir[0]=='cases.0.samples.0.sample_type')[0]

    data_mir = data_mir[1:]

    input_mir_sur = np.empty((0,1881), float)
    for case in cases_mir_sur:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_sur = np.vstack([input_mir_sur,row])

    np.save(DATASET_INPUT_MIR_SURVIVAL + 'input_mir_sur.npy', input_mir_sur)
    print('input_mir_sur.npy is created')



# Create the input (feature) set of gene expression for survival rate regression
# implemented in mDBN of gene-miRNA
# We use the expression value of each genes
# There are three different workflow analysis (HTSEC count, HTSEC FPKM, HTSEC FPKM-UQ) that was used to compute the expression value
# Based on this, we create input set for all 3 types of workflow analysis
def input_gen_genmir_survival():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_gen_mir_cli = np.intersect1d(cases_gen_cli,cases_mir) # [1071,]


    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_sur = np.genfromtxt(TARGET_CLINICAL + "survival_plot.tsv", dtype=str, delimiter='\t', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    case_id_column, = np.where(temp_sur[0]=='id')[0]

    temp_sur = temp_sur[1:]

    sur = temp_sur[temp_sur[:,case_id_column].argsort()]

    cases_sur = sur[:,case_id_column]     # [1084,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_gen_mir_sur = np.intersect1d(cases_gen_mir_cli,cases_sur) # (1058,)
    
    
    ##############################################
    ###### INPUT GENE OF SURVIVAL REGRESSION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_GEN_GEN_MIR_SURVIVAL)):
        os.makedirs(DATASET_INPUT_GEN_GEN_MIR_SURVIVAL)

    data_gene = np.genfromtxt(TARGET_META_CSV + "gene_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_gene[0]=='file_id')[0]
    file_name_column, = np.where(data_gene[0]=='file_name')[0]
    case_id_column, = np.where(data_gene[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_gene[0]=='cases.0.samples.0.sample_type')[0]
    workflow_column, = np.where(data_gene[0]=='analysis.workflow_type')[0]

    data_gene = data_gene[1:]

    # 1. Gene (count) survival regression
    input_gen_genmir_count_sur = np.empty((0,60483), int)
    for case in cases_gen_mir_sur:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_genmir_count_sur = np.vstack([input_gen_genmir_count_sur,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_SURVIVAL + 'input_gen_genmir_count_sur.npy', input_gen_genmir_count_sur)
    print('input_gen_genmir_count_sur.npy is created')


    # 2. Gene (FPKM) survival regression
    input_gen_genmir_fpkm_sur = np.empty((0,60483), int)
    for case in cases_gen_mir_sur:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_genmir_fpkm_sur = np.vstack([input_gen_genmir_fpkm_sur,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_SURVIVAL + 'input_gen_genmir_fpkm_sur.npy', input_gen_genmir_fpkm_sur)
    print('input_gen_genmir_fpkm_sur.npy is created')


    # 3. Gene (FPKM-UQ) survival regression
    input_gen_genmir_fpkmuq_sur = np.empty((0,60483), int)
    for case in cases_gen_mir_sur:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_genmir_fpkmuq_sur = np.vstack([input_gen_genmir_fpkmuq_sur,row])

    np.save(DATASET_INPUT_GEN_GEN_MIR_SURVIVAL + 'input_gen_genmir_fpkmuq_sur.npy', input_gen_genmir_fpkmuq_sur)
    print('input_gen_genmir_fpkmuq_sur.npy is created')



# Create the input (feature) set of miRNA expression data for survival rate regression
# implemented in mDBN of gene-miRNA
# We use the read count value of each miRNA
def input_mir_genmir_survival():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_gen_mir_cli = np.intersect1d(cases_gen_cli,cases_mir) # [1071,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_sur = np.genfromtxt(TARGET_CLINICAL + "survival_plot.tsv", dtype=str, delimiter='\t', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    case_id_column, = np.where(temp_sur[0]=='id')[0]

    temp_sur = temp_sur[1:]

    sur = temp_sur[temp_sur[:,case_id_column].argsort()]

    cases_sur = sur[:,case_id_column]     # [1084,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_gen_mir_sur = np.intersect1d(cases_gen_mir_cli,cases_sur) # (1058,)
    
    
    ##############################################
    ##### INPUT MIRNA OF TYPE CLASSIFICATION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_MIR_GEN_MIR_SURVIVAL)):
        os.makedirs(DATASET_INPUT_MIR_GEN_MIR_SURVIVAL)

    data_mir = np.genfromtxt(TARGET_META_CSV + "mirna_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_mir[0]=='file_id')[0]
    file_name_column, = np.where(data_mir[0]=='file_name')[0]
    case_id_column, = np.where(data_mir[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_mir[0]=='cases.0.samples.0.sample_type')[0]

    data_mir = data_mir[1:]

    input_mir_genmir_sur = np.empty((0,1881), float)
    for case in cases_gen_mir_sur:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_genmir_sur = np.vstack([input_mir_genmir_sur,row])

    np.save(DATASET_INPUT_MIR_GEN_MIR_SURVIVAL + 'input_mir_genmir_sur.npy', input_mir_genmir_sur)
    print('input_mir_genmir_sur.npy is created')



# Create the input (feature) set of methylation data for survival rate regression
# implemented in mDBN of methylation-gene-miRNA
# available for both NCBI Platform GPL8490 and NCBI Platform GPL16304
# We use the beta value from CPG sites
def input_met_metgenmir_survival():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_met = cases[cases[:,8]!="0",0]    # [1095,]
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_met_cli = np.intersect1d(cases_met,cases_cli) # [1095,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_met_gen_mir_cli = np.intersect1d(np.intersect1d(cases_met_cli,cases_gen),cases_mir) # [1069,]


    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_sur = np.genfromtxt(TARGET_CLINICAL + "survival_plot.tsv", dtype=str, delimiter='\t', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    case_id_column, = np.where(temp_sur[0]=='id')[0]

    temp_sur = temp_sur[1:]

    sur = temp_sur[temp_sur[:,case_id_column].argsort()]

    cases_sur = sur[:,case_id_column]     # [1084,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_met_gen_mir_sur = np.intersect1d(cases_met_gen_mir_cli,cases_sur) # (1056,)
    
    
    ##############################################
    ## INPUT METHYLATION OF SURVIVAL REGRESSION ##
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_MET_MET_GEN_MIR_SURVIVAL)):
        os.makedirs(DATASET_INPUT_MET_MET_GEN_MIR_SURVIVAL)

    data_met = np.genfromtxt(TARGET_META_CSV + "methylation_beta_value.csv", dtype=str, delimiter=',', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_met[0]=='file_id')[0]
    file_name_column, = np.where(data_met[0]=='file_name')[0]
    case_id_column, = np.where(data_met[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_met[0]=='cases.0.samples.0.sample_type')[0]

    data_met = data_met[1:]

    with open(TARGET_METHYLATION + "cpg.json") as f:
        cpg = yaml.safe_load(f)

    with open(TARGET_METHYLATION + "cpg_in_cpg_short_idx.json") as f:
        cpg_in_cpg_short_idx = yaml.safe_load(f)

    with open(TARGET_METHYLATION + "cpg_in_cpg_long_idx.json") as f:
        cpg_in_cpg_long_idx = yaml.safe_load(f)


    input_met_metgenmir_sur = np.empty((0,25978), float)
    for case in cases_met_gen_mir_sur:
        file_id = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_met[np.intersect1d(np.where(data_met[:,case_id_column] == case), np.where(data_met[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
            file_met = [row.split("\t") for row in f]

        file_met = file_met[1:]
        
        temp = np.empty((0,1), float)

        if len(file_met) == 27578:
            for idx in sorted(cpg_in_cpg_short_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])

                temp = np.append(temp, beta)

        elif len(file_met) == 485577:
            for idx in sorted(cpg_in_cpg_long_idx):
                if file_met[idx][1] == "NA":
                    beta = 0.

                else:
                    beta = float(file_met[idx][1])
                
                temp = np.append(temp, beta)
        
        print(case)
        input_met_metgenmir_sur = np.vstack([input_met_metgenmir_sur,temp])

    np.save(DATASET_INPUT_MET_MET_GEN_MIR_SURVIVAL + 'input_met_metgenmir_sur.npy', input_met_metgenmir_sur)
    print('input_met_metgenmir_sur.npy is created')



# Create the input (feature) set of gene expression for survival rate regression
# implemented in mDBN of methylation-gene-miRNA
# We use the expression value of each genes
# There are three different workflow analysis (HTSEC count, HTSEC FPKM, HTSEC FPKM-UQ) that was used to compute the expression value
# Based on this, we create input set for all 3 types of workflow analysis
def input_gen_metgenmir_survival():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_met = cases[cases[:,8]!="0",0]    # [1095,]
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_met_cli = np.intersect1d(cases_met,cases_cli) # [1095,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_met_gen_mir_cli = np.intersect1d(np.intersect1d(cases_met_cli,cases_gen),cases_mir) # [1069,]


    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_sur = np.genfromtxt(TARGET_CLINICAL + "survival_plot.tsv", dtype=str, delimiter='\t', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    case_id_column, = np.where(temp_sur[0]=='id')[0]

    temp_sur = temp_sur[1:]

    sur = temp_sur[temp_sur[:,case_id_column].argsort()]

    cases_sur = sur[:,case_id_column]     # [1084,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_met_gen_mir_sur = np.intersect1d(cases_met_gen_mir_cli,cases_sur) # (1056,)
    
    
    ##############################################
    ###### INPUT GENE OF SURVIVAL REGRESSION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_GEN_MET_GEN_MIR_SURVIVAL)):
        os.makedirs(DATASET_INPUT_GEN_MET_GEN_MIR_SURVIVAL)

    data_gene = np.genfromtxt(TARGET_META_CSV + "gene_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_gene[0]=='file_id')[0]
    file_name_column, = np.where(data_gene[0]=='file_name')[0]
    case_id_column, = np.where(data_gene[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_gene[0]=='cases.0.samples.0.sample_type')[0]
    workflow_column, = np.where(data_gene[0]=='analysis.workflow_type')[0]

    data_gene = data_gene[1:]

    # 1. Gene (count) survival regression
    input_gen_metgenmir_count_sur = np.empty((0,60483), int)
    for case in cases_met_gen_mir_sur:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_metgenmir_count_sur = np.vstack([input_gen_metgenmir_count_sur,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_SURVIVAL + 'input_gen_metgenmir_count_sur.npy', input_gen_metgenmir_count_sur)
    print('input_gen_metgenmir_count_sur.npy is created')


    # 2. Gene (FPKM) survival regression
    input_gen_metgenmir_fpkm_sur = np.empty((0,60483), int)
    for case in cases_met_gen_mir_sur:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metgenmir_fpkm_sur = np.vstack([input_gen_metgenmir_fpkm_sur,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_SURVIVAL + 'input_gen_metgenmir_fpkm_sur.npy', input_gen_metgenmir_fpkm_sur)
    print('input_gen_metgenmir_fpkm_sur.npy is created')


    # 3. Gene (FPKM-UQ) survival regression
    input_gen_metgenmir_fpkmuq_sur = np.empty((0,60483), int)
    for case in cases_met_gen_mir_sur:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metgenmir_fpkmuq_sur = np.vstack([input_gen_metgenmir_fpkmuq_sur,row])

    np.save(DATASET_INPUT_GEN_MET_GEN_MIR_SURVIVAL + 'input_gen_metgenmir_fpkmuq_sur.npy', input_gen_metgenmir_fpkmuq_sur)
    print('input_gen_metgenmir_fpkmuq_sur.npy is created')



# Create the input (feature) set of miRNA expression data for survival rate regression
# implemented in mDBN of methylation-gene-miRNA
# We use the read count value of each miRNA
def input_mir_metgenmir_survival():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    cases_met = cases[cases[:,8]!="0",0]    # [1095,]
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_met_cli = np.intersect1d(cases_met,cases_cli) # [1095,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_met_gen_mir_cli = np.intersect1d(np.intersect1d(cases_met_cli,cases_gen),cases_mir) # [1069,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_sur = np.genfromtxt(TARGET_CLINICAL + "survival_plot.tsv", dtype=str, delimiter='\t', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    case_id_column, = np.where(temp_sur[0]=='id')[0]

    temp_sur = temp_sur[1:]

    sur = temp_sur[temp_sur[:,case_id_column].argsort()]

    cases_sur = sur[:,case_id_column]     # [1084,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_met_gen_mir_sur = np.intersect1d(cases_met_gen_mir_cli,cases_sur) # (1056,)
    
    
    ##############################################
    ##### INPUT MIRNA OF TYPE CLASSIFICATION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_MIR_MET_GEN_MIR_SURVIVAL)):
        os.makedirs(DATASET_INPUT_MIR_MET_GEN_MIR_SURVIVAL)

    data_mir = np.genfromtxt(TARGET_META_CSV + "mirna_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_mir[0]=='file_id')[0]
    file_name_column, = np.where(data_mir[0]=='file_name')[0]
    case_id_column, = np.where(data_mir[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_mir[0]=='cases.0.samples.0.sample_type')[0]

    data_mir = data_mir[1:]

    input_mir_metgenmir_sur = np.empty((0,1881), float)
    for case in cases_met_gen_mir_sur:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_metgenmir_sur = np.vstack([input_mir_metgenmir_sur,row])

    np.save(DATASET_INPUT_MIR_MET_GEN_MIR_SURVIVAL + 'input_mir_metgenmir_sur.npy', input_mir_metgenmir_sur)
    print('input_mir_metgenmir_sur.npy is created')



# Create the input (feature) set of methylation long data for survival rate regression
# implemented in mDBN of methylation_long-gene-miRNA
# only for NCBI Platform GPL16304
# We use the beta value from CPG sites
def input_metlong_metlonggenmir_survival():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    with open(TARGET_METHYLATION + "cases_met_long.json") as f:
        temp = yaml.safe_load(f)
    cases_metlong = np.asarray(temp)  # [782,]
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_metlong_cli = np.intersect1d(cases_metlong,cases_cli) # [782,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_metlong_gen_mir_cli = np.intersect1d(np.intersect1d(cases_metlong_cli,cases_gen),cases_mir) # [768,]


    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_sur = np.genfromtxt(TARGET_CLINICAL + "survival_plot.tsv", dtype=str, delimiter='\t', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    case_id_column, = np.where(temp_sur[0]=='id')[0]

    temp_sur = temp_sur[1:]

    sur = temp_sur[temp_sur[:,case_id_column].argsort()]

    cases_sur = sur[:,case_id_column]     # [1084,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_metlong_gen_mir_sur = np.intersect1d(cases_metlong_gen_mir_cli,cases_sur) # (767,)
    
    
    #####################################################
    ### INPUT LONG METHYLATION OF SURVIVAL REGRESSION ###
    #####################################################
    if not(os.path.isdir(DATASET_INPUT_METLONG_METLONG_GEN_MIR_SURVIVAL)):
        os.makedirs(DATASET_INPUT_METLONG_METLONG_GEN_MIR_SURVIVAL)

    data_metlong = np.genfromtxt(TARGET_META_CSV + "methylation_long_beta_value.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_metlong[0]=='file_id')[0]
    file_name_column, = np.where(data_metlong[0]=='file_name')[0]
    case_id_column, = np.where(data_metlong[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_metlong[0]=='cases.0.samples.0.sample_type')[0]

    data_metlong = data_metlong[1:]

    # we divided the resulting file in 4 because the size is too large for 1 file
    for i in range(4):
        input_metlong_metlonggenmir_sur = []
        for case in cases_metlong_gen_mir_sur[200*i:200*(i+1)]:
            file_id = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
            file_name = data_metlong[np.intersect1d(np.where(data_metlong[:,case_id_column] == case), np.where(data_metlong[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
            
            with open(DATASET_METHYLATION + file_id + "/" + file_name) as f:
                file_metlong = [row.split("\t") for row in f]

            file_metlong = file_metlong[1:]
            
            temp = []

            for j in range(len(file_metlong)):
                if file_metlong[j][1] == "NA":
                    beta = 0.
                else:
                    beta = float(file_metlong[j][1])
                temp.append(beta)
            
            input_metlong_metlonggenmir_sur.append(temp)
            print(str(len(input_metlong_metlonggenmir_sur)) + ". " + case)

        np.save(DATASET_INPUT_METLONG_METLONG_GEN_MIR_SURVIVAL + 'input_metlong_metlonggenmir_sur_' + str(i) + '.npy', input_metlong_metlonggenmir_sur)
        print('input_metlong_metlonggenmir_sur_' + str(i) + '.npy is created')

    # combine it in 1 file
    temp_0 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_SURVIVAL + 'input_metlong_metlonggenmir_sur_0.npy')
    temp_1 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_SURVIVAL + 'input_metlong_metlonggenmir_sur_1.npy')
    temp_2 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_SURVIVAL + 'input_metlong_metlonggenmir_sur_2.npy')
    temp_3 = np.load(DATASET_INPUT_METLONG_METLONG_GEN_MIR_SURVIVAL + 'input_metlong_metlonggenmir_sur_3.npy')
    input_metlong_metlonggenmir_sur = np.concatenate((np.concatenate((temp_0, temp_1), axis=0),
                                                      np.concatenate((temp_2, temp_3), axis=0)),
                                                     axis=0)
    np.save(DATASET_INPUT_METLONG_METLONG_GEN_MIR_SURVIVAL + 'input_metlong_metlonggenmir_sur.npy', input_metlong_metlonggenmir_sur)

    # remove temp file
    for i in range(4):
        os.remove(DATASET_INPUT_METLONG_METLONG_GEN_MIR_SURVIVAL + 'input_metlong_metlonggenmir_sur_' + str(i) + '.npy')



# Create the input (feature) set of gene expression for survival rate regression
# implemented in mDBN of methylation_long-gene-miRNA
# We use the expression value of each genes
# There are three different workflow analysis (HTSEC count, HTSEC FPKM, HTSEC FPKM-UQ) that was used to compute the expression value
# Based on this, we create input set for all 3 types of workflow analysis
def input_gen_metlonggenmir_survival():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    with open(TARGET_METHYLATION + "cases_met_long.json") as f:
        temp = yaml.safe_load(f)
    cases_metlong = np.asarray(temp)  # [782,]
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_metlong_cli = np.intersect1d(cases_metlong,cases_cli) # [782,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_metlong_gen_mir_cli = np.intersect1d(np.intersect1d(cases_metlong_cli,cases_gen),cases_mir) # [768,]


    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_sur = np.genfromtxt(TARGET_CLINICAL + "survival_plot.tsv", dtype=str, delimiter='\t', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    case_id_column, = np.where(temp_sur[0]=='id')[0]

    temp_sur = temp_sur[1:]

    sur = temp_sur[temp_sur[:,case_id_column].argsort()]

    cases_sur = sur[:,case_id_column]     # [1084,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_metlong_gen_mir_sur = np.intersect1d(cases_metlong_gen_mir_cli,cases_sur) # (767,)
    
    
    ##############################################
    ###### INPUT GENE OF SURVIVAL REGRESSION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_GEN_METLONG_GEN_MIR_SURVIVAL)):
        os.makedirs(DATASET_INPUT_GEN_METLONG_GEN_MIR_SURVIVAL)

    data_gene = np.genfromtxt(TARGET_META_CSV + "gene_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_gene[0]=='file_id')[0]
    file_name_column, = np.where(data_gene[0]=='file_name')[0]
    case_id_column, = np.where(data_gene[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_gene[0]=='cases.0.samples.0.sample_type')[0]
    workflow_column, = np.where(data_gene[0]=='analysis.workflow_type')[0]

    data_gene = data_gene[1:]

    # 1. Gene (count) survival regression
    input_gen_metlonggenmir_count_sur = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_sur:
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:60483,1].astype(float)
        input_gen_metlonggenmir_count_sur = np.vstack([input_gen_metlonggenmir_count_sur,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_SURVIVAL + 'input_gen_metlonggenmir_count_sur.npy', input_gen_metlonggenmir_count_sur)
    print('input_gen_metlonggenmir_count_sur.npy is created')


    # 2. Gene (FPKM) survival regression
    input_gen_metlonggenmir_fpkm_sur = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_sur:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metlonggenmir_fpkm_sur = np.vstack([input_gen_metlonggenmir_fpkm_sur,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_SURVIVAL + 'input_gen_metlonggenmir_fpkm_sur.npy', input_gen_metlonggenmir_fpkm_sur)
    print('input_gen_metlonggenmir_fpkm_sur.npy is created')


    # 3. Gene (FPKM-UQ) survival regression
    input_gen_metlonggenmir_fpkmuq_sur = np.empty((0,60483), int)
    for case in cases_metlong_gen_mir_sur:
        file_name = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,workflow_column] == "HTSeq - Counts")), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        file_name = file_name.split(".")[0] + ".FPKM-UQ.txt.gz"
        file_id = data_gene[np.intersect1d(np.intersect1d(np.where(data_gene[:,case_id_column] == case), np.where(data_gene[:,file_name_column] == file_name)), np.where(data_gene[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        
        with gzip.open(DATASET_GENE + file_id + "/" + file_name) as f:
            file = np.genfromtxt(f, dtype=str, delimiter='\t')
        print(case)
        
        row = file[:,1].astype(float)
        input_gen_metlonggenmir_fpkmuq_sur = np.vstack([input_gen_metlonggenmir_fpkmuq_sur,row])

    np.save(DATASET_INPUT_GEN_METLONG_GEN_MIR_SURVIVAL + 'input_gen_metlonggenmir_fpkmuq_sur.npy', input_gen_metlonggenmir_fpkmuq_sur)
    print('input_gen_metlonggenmir_fpkmuq_sur.npy is created')



# Create the input (feature) set of miRNA expression data for survival rate regression
# implemented in mDBN of methylation_long-gene-miRNA
# We use the read count value of each miRNA
def input_mir_metlonggenmir_survival():
    ######################################
    ### AVAILABLE CASES BASED ON INPUT ###
    ######################################
    cases = np.genfromtxt(TARGET_META_CSV + "file_amount.csv", dtype=str, delimiter=',', skip_header=1)
    with open(TARGET_METHYLATION + "cases_met_long.json") as f:
        temp = yaml.safe_load(f)
    cases_metlong = np.asarray(temp)  # [782,]
    cases_gen = cases[cases[:,9]!="0",0]    # [1092,]
    # remove cases in cases_gen where there are no tumor sample
    cases_gen = np.delete(cases_gen, np.argwhere(cases_gen=="2b22db1d-54a1-4b9e-a86e-a174cf51d95c"))    # [1091,]
    cases_mir = cases[cases[:,10]!="0",0]   # [1078,]
    # remove cases in cases_mir where there are no tumor sample
    cases_mir = np.delete(cases_mir, np.argwhere(cases_mir=="3c8b5af9-c34d-43c2-b8c9-39ea11e44fa6"))    # [1078,]
    cases_cli = cases[cases[:,12]!="0",0]   # [1097,]
    cases_metlong_cli = np.intersect1d(cases_metlong,cases_cli) # [782,]
    cases_gen_cli = np.intersect1d(cases_gen,cases_cli) # [1090,]
    cases_mir_cli = np.intersect1d(cases_mir,cases_cli) # [1077,]
    cases_metlong_gen_mir_cli = np.intersect1d(np.intersect1d(cases_metlong_cli,cases_gen),cases_mir) # [768,]
    

    ######################################
    ### AVAILABLE CASES BASED ON OUTPUT ##
    ######################################
    temp_sur = np.genfromtxt(TARGET_CLINICAL + "survival_plot.tsv", dtype=str, delimiter='\t', skip_header=0)
    
    # find where the case id column is located in your meta_clinicals.csv
    case_id_column, = np.where(temp_sur[0]=='id')[0]

    temp_sur = temp_sur[1:]

    sur = temp_sur[temp_sur[:,case_id_column].argsort()]

    cases_sur = sur[:,case_id_column]     # [1084,]

    
    ######################################
    ########### AVAILABLE CASES ##########
    ######################################
    cases_metlong_gen_mir_sur = np.intersect1d(cases_metlong_gen_mir_cli,cases_sur) # (767,)
    
    
    ##############################################
    ##### INPUT MIRNA OF TYPE CLASSIFICATION #####
    ##############################################
    if not(os.path.isdir(DATASET_INPUT_MIR_METLONG_GEN_MIR_SURVIVAL)):
        os.makedirs(DATASET_INPUT_MIR_METLONG_GEN_MIR_SURVIVAL)

    data_mir = np.genfromtxt(TARGET_META_CSV + "mirna_expression_quantification.csv", dtype=str, delimiter=',', skip_header=0)

    # find where the case id column is located in your meta_clinicals.csv
    file_id_column, = np.where(data_mir[0]=='file_id')[0]
    file_name_column, = np.where(data_mir[0]=='file_name')[0]
    case_id_column, = np.where(data_mir[0]=='cases.0.case_id')[0]
    sample_type_column, = np.where(data_mir[0]=='cases.0.samples.0.sample_type')[0]

    data_mir = data_mir[1:]

    input_mir_metlonggenmir_sur = np.empty((0,1881), float)
    for case in cases_metlong_gen_mir_sur:
        file_id = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_id_column][0]
        file_name = data_mir[np.intersect1d(np.where(data_mir[:,case_id_column] == case), np.where(data_mir[:,sample_type_column] == "Primary Tumor")),file_name_column][0]
        
        file = np.genfromtxt(DATASET_MIRNA + file_id + "/" + file_name, dtype=str, delimiter='\t', skip_header=1)
        
        row = file[:,1].astype(float)
        input_mir_metlonggenmir_sur = np.vstack([input_mir_metlonggenmir_sur,row])

    np.save(DATASET_INPUT_MIR_METLONG_GEN_MIR_SURVIVAL + 'input_mir_metlonggenmir_sur.npy', input_mir_metlonggenmir_sur)
    print('input_mir_metlonggenmir_sur.npy is created')



if __name__ == '__main__':
    start = timeit.default_timer()
    label_cancer_type()
    input_met_cancer_type()
    input_metlong_cancer_type()
    input_gen_cancer_type()
    input_mir_cancer_type()

    input_gen_genmir_cancer_type()
    input_mir_genmir_cancer_type()
    
    input_met_metgenmir_cancer_type()
    input_gen_metgenmir_cancer_type()
    input_mir_metgenmir_cancer_type()
    
    input_metlong_metlonggenmir_cancer_type()
    input_gen_metlonggenmir_cancer_type()
    input_mir_metlonggenmir_cancer_type()


    label_survival()
    input_met_survival()
    input_metlong_survival()
    input_gen_survival()
    input_mir_survival()
    
    input_gen_genmir_survival()
    input_mir_genmir_survival()
    
    input_met_metgenmir_survival()
    input_gen_metgenmir_survival()
    input_mir_metgenmir_survival()
    
    input_metlong_metlonggenmir_survival()
    input_gen_metlonggenmir_survival()
    input_mir_metlonggenmir_survival()
    stop = timeit.default_timer()
    print stop - start