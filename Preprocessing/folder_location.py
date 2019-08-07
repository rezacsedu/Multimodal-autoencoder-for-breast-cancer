######################################
########### DATASET SOURCE ###########
######################################
# Set the "Dataset Source" folders based on where you downloaded your all your TCGA BRCA files
# Separate your files based on the data type
# 1. Methylation Beta Value
DATASET_METHYLATION = MAIN_MDBN_TCGA_BRCA + "methylation/"
# 2. Gene Expression Quantification
DATASET_GENE = MAIN_MDBN_TCGA_BRCA + "EXP/gene_expression_quantification/"
# 3. miRNA Expression Quantification
DATASET_MIRNA = MAIN_MDBN_TCGA_BRCA + "EXP/mirna_expression_quantification/"
# 4. Clinical Supplement
DATASET_CLINICAL = MAIN_MDBN_TCGA_BRCA + "clinical/"



######################################
################ META ################
######################################
# The main folder for all your meta data
TARGET_META = MAIN_MDBN_TCGA_BRCA + "meta/"

# This 2 folders are for meta files you request directly from GDC API (https://api.gdc.cancer.gov/)
# There are 2 data format which we will request our main meta files
TARGET_META_CSV = TARGET_META + "general/csv/"
TARGET_META_JSON = TARGET_META + "general/json/"

# This 4 meta folders correspond specifically to each data type
TARGET_METHYLATION = TARGET_META + "methylation/"
TARGET_GENE = TARGET_META + "gene/"
TARGET_MIRNA = TARGET_META + "mirna/"
TARGET_CLINICAL = TARGET_META + "clinical/"



######################################
############ DATASET INPUT ###########
######################################
# The main folder for all the inputs/features of your training set
# You need at least 30GB of free space for this folder
DATASET_INPUT = MAIN_MDBN_TCGA_BRCA + "input/"

DATASET_INPUT_MET_TYPE = DATASET_INPUT + "type/met/"
DATASET_INPUT_MET_SURVIVAL = DATASET_INPUT + "survival/met/"
DATASET_INPUT_MET_DRUGS = DATASET_INPUT + "drugs/met/"

DATASET_INPUT_METLONG_TYPE = DATASET_INPUT + "type/metlong/"
DATASET_INPUT_METLONG_SURVIVAL = DATASET_INPUT + "survival/metlong/"
DATASET_INPUT_METLONG_DRUGS = DATASET_INPUT + "drugs/metlong/"

DATASET_INPUT_GEN_TYPE = DATASET_INPUT + "type/gen/"
DATASET_INPUT_GEN_SURVIVAL = DATASET_INPUT + "survival/gen/"
DATASET_INPUT_GEN_DRUGS = DATASET_INPUT + "drugs/gen/"

DATASET_INPUT_MIR_TYPE = DATASET_INPUT + "type/mir/"
DATASET_INPUT_MIR_SURVIVAL = DATASET_INPUT + "survival/mir/"
DATASET_INPUT_MIR_DRUGS = DATASET_INPUT + "drugs/mir/"

DATASET_INPUT_GEN_GEN_MIR_TYPE = DATASET_INPUT + "type/genmir/gen/"
DATASET_INPUT_GEN_GEN_MIR_SURVIVAL = DATASET_INPUT + "survival/genmir/gen/"
DATASET_INPUT_GEN_GEN_MIR_DRUGS = DATASET_INPUT + "drugs/genmir/gen/"
DATASET_INPUT_MIR_GEN_MIR_TYPE = DATASET_INPUT + "type/genmir/mir/"
DATASET_INPUT_MIR_GEN_MIR_SURVIVAL = DATASET_INPUT + "survival/genmir/mir/"
DATASET_INPUT_MIR_GEN_MIR_DRUGS = DATASET_INPUT + "drugs/genmir/mir/"

DATASET_INPUT_MET_MET_GEN_MIR_TYPE = DATASET_INPUT + "type/metgenmir/met/"
DATASET_INPUT_MET_MET_GEN_MIR_SURVIVAL = DATASET_INPUT + "survival/metgenmir/met/"
DATASET_INPUT_MET_MET_GEN_MIR_DRUGS = DATASET_INPUT + "drugs/metgenmir/met/"
DATASET_INPUT_GEN_MET_GEN_MIR_TYPE = DATASET_INPUT + "type/metgenmir/gen/"
DATASET_INPUT_GEN_MET_GEN_MIR_SURVIVAL = DATASET_INPUT + "survival/metgenmir/gen/"
DATASET_INPUT_GEN_MET_GEN_MIR_DRUGS = DATASET_INPUT + "drugs/metgenmir/gen/"
DATASET_INPUT_MIR_MET_GEN_MIR_TYPE = DATASET_INPUT + "type/metgenmir/mir/"
DATASET_INPUT_MIR_MET_GEN_MIR_SURVIVAL = DATASET_INPUT + "survival/metgenmir/mir/"
DATASET_INPUT_MIR_MET_GEN_MIR_DRUGS = DATASET_INPUT + "drugs/metgenmir/mir/"

DATASET_INPUT_METLONG_METLONG_GEN_MIR_TYPE = DATASET_INPUT + "type/metlonggenmir/metlong/"
DATASET_INPUT_METLONG_METLONG_GEN_MIR_SURVIVAL = DATASET_INPUT + "survival/metlonggenmir/metlong/"
DATASET_INPUT_METLONG_METLONG_GEN_MIR_DRUGS = DATASET_INPUT + "drugs/metlonggenmir/metlong/"
DATASET_INPUT_GEN_METLONG_GEN_MIR_TYPE = DATASET_INPUT + "type/metlonggenmir/gen/"
DATASET_INPUT_GEN_METLONG_GEN_MIR_SURVIVAL = DATASET_INPUT + "survival/metlonggenmir/gen/"
DATASET_INPUT_GEN_METLONG_GEN_MIR_DRUGS = DATASET_INPUT + "drugs/metlonggenmir/gen/"
DATASET_INPUT_MIR_METLONG_GEN_MIR_TYPE = DATASET_INPUT + "type/metlonggenmir/mir/"
DATASET_INPUT_MIR_METLONG_GEN_MIR_SURVIVAL = DATASET_INPUT + "survival/metlonggenmir/mir/"
DATASET_INPUT_MIR_METLONG_GEN_MIR_DRUGS = DATASET_INPUT + "drugs/metlonggenmir/mir/"



######################################
########### DATASET LABELS ###########
######################################
# The main folder for all the labels of your training set
# You need at least 30GB of free space for this folder
DATASET_LABELS = MAIN_MDBN_TCGA_BRCA + "labels/"

DATASET_LABELS_MET_TYPE = DATASET_LABELS + "type/met/"
DATASET_LABELS_MET_SURVIVAL = DATASET_LABELS + "survival/met/"
DATASET_LABELS_MET_DRUGS = DATASET_LABELS + "drugs/met/"

DATASET_LABELS_METLONG_TYPE = DATASET_LABELS + "type/metlong/"
DATASET_LABELS_METLONG_SURVIVAL = DATASET_LABELS + "survival/metlong/"
DATASET_LABELS_METLONG_DRUGS = DATASET_LABELS + "drugs/metlong/"

DATASET_LABELS_GEN_TYPE = DATASET_LABELS + "type/gen/"
DATASET_LABELS_GEN_SURVIVAL = DATASET_LABELS + "survival/gen/"
DATASET_LABELS_GEN_DRUGS = DATASET_LABELS + "drugs/gen/"

DATASET_LABELS_MIR_TYPE = DATASET_LABELS + "type/mir/"
DATASET_LABELS_MIR_SURVIVAL = DATASET_LABELS + "survival/mir/"
DATASET_LABELS_MIR_DRUGS = DATASET_LABELS + "drugs/mir/"

DATASET_LABELS_GEN_MIR_TYPE = DATASET_LABELS + "type/genmir/"
DATASET_LABELS_GEN_MIR_SURVIVAL = DATASET_LABELS + "survival/genmir/"
DATASET_LABELS_GEN_MIR_DRUGS = DATASET_LABELS + "drugs/genmir/"

DATASET_LABELS_MET_GEN_MIR_TYPE = DATASET_LABELS + "type/metgenmir/"
DATASET_LABELS_MET_GEN_MIR_SURVIVAL = DATASET_LABELS + "survival/metgenmir/"
DATASET_LABELS_MET_GEN_MIR_DRUGS = DATASET_LABELS + "drugs/metgenmir/"

DATASET_LABELS_METLONG_GEN_MIR_TYPE = DATASET_LABELS + "type/metlonggenmir/"
DATASET_LABELS_METLONG_GEN_MIR_SURVIVAL = DATASET_LABELS + "survival/metlonggenmir/"
DATASET_LABELS_METLONG_GEN_MIR_DRUGS = DATASET_LABELS + "drugs/metlonggenmir/"