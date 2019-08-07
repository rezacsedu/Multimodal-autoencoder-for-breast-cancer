########################
######### INPUT ########
########################
# The main folder for all the inputs/features of your training set
# You need at least 30GB of free space for this folder
# MAIN_MDBN_TCGA_BRCA = "main_data_folder"
DATASET_INPUT = MAIN_MDBN_TCGA_BRCA + "input/"

INPUT_MET_TYPE_ER = DATASET_INPUT + "type/met/input_met_type_er.npy"
INPUT_MET_TYPE_PGR = DATASET_INPUT + "type/met/input_met_type_pgr.npy"
INPUT_MET_TYPE_HER2 = DATASET_INPUT + "type/met/input_met_type_her2.npy"
INPUT_MET_SURVIVAL = DATASET_INPUT + "survival/met/input_met_sur.npy"

INPUT_METLONG_TYPE_ER = DATASET_INPUT + "type/metlong/input_metlong_type_er.npy"
INPUT_METLONG_TYPE_PGR = DATASET_INPUT + "type/metlong/input_metlong_type_pgr.npy"
INPUT_METLONG_TYPE_HER2 = DATASET_INPUT + "type/metlong/input_metlong_type_her2.npy"
INPUT_METLONG_SURVIVAL = DATASET_INPUT + "survival/metlong/input_metlong_sur.npy"

INPUT_GEN_TYPE_ER_COUNT = DATASET_INPUT + "type/gen/input_gen_count_type_er.npy"
INPUT_GEN_TYPE_ER_FPKM = DATASET_INPUT + "type/gen/input_gen_fpkm_type_er.npy"
INPUT_GEN_TYPE_ER_FPKMUQ = DATASET_INPUT + "type/gen/input_gen_fpkmuq_type_er.npy"
INPUT_GEN_TYPE_PGR_COUNT = DATASET_INPUT + "type/gen/input_gen_count_type_pgr.npy"
INPUT_GEN_TYPE_PGR_FPKM = DATASET_INPUT + "type/gen/input_gen_fpkm_type_pgr.npy"
INPUT_GEN_TYPE_PGR_FPKMUQ = DATASET_INPUT + "type/gen/input_gen_fpkmuq_type_pgr.npy"
INPUT_GEN_TYPE_HER2_COUNT = DATASET_INPUT + "type/gen/input_gen_count_type_her2.npy"
INPUT_GEN_TYPE_HER2_FPKM = DATASET_INPUT + "type/gen/input_gen_fpkm_type_her2.npy"
INPUT_GEN_TYPE_HER2_FPKMUQ = DATASET_INPUT + "type/gen/input_gen_fpkmuq_type_her2.npy"
INPUT_GEN_SURVIVAL_COUNT = DATASET_INPUT + "survival/gen/input_gen_count_sur.npy"
INPUT_GEN_SURVIVAL_FPKM = DATASET_INPUT + "survival/gen/input_gen_fpkm_sur.npy"
INPUT_GEN_SURVIVAL_FPKMUQ = DATASET_INPUT + "survival/gen/input_gen_fpkmuq_sur.npy"

INPUT_MIR_TYPE_ER = DATASET_INPUT + "type/mir/input_mir_type_er.npy"
INPUT_MIR_TYPE_PGR = DATASET_INPUT + "type/mir/input_mir_type_pgr.npy"
INPUT_MIR_TYPE_HER2 = DATASET_INPUT + "type/mir/input_mir_type_her2.npy"
INPUT_MIR_SURVIVAL = DATASET_INPUT + "survival/mir/input_mir_sur.npy"

INPUT_GEN_GEN_MIR_TYPE_ER_COUNT = DATASET_INPUT + "type/genmir/gen/input_gen_genmir_count_type_er.npy"
INPUT_GEN_GEN_MIR_TYPE_ER_FPKM = DATASET_INPUT + "type/genmir/gen/input_gen_genmir_fpkm_type_er.npy"
INPUT_GEN_GEN_MIR_TYPE_ER_FPKMUQ = DATASET_INPUT + "type/genmir/gen/input_gen_genmir_fpkmuq_type_er.npy"
INPUT_GEN_GEN_MIR_TYPE_PGR_COUNT = DATASET_INPUT + "type/genmir/gen/input_gen_genmir_count_type_pgr.npy"
INPUT_GEN_GEN_MIR_TYPE_PGR_FPKM = DATASET_INPUT + "type/genmir/gen/input_gen_genmir_fpkm_type_pgr.npy"
INPUT_GEN_GEN_MIR_TYPE_PGR_FPKMUQ = DATASET_INPUT + "type/genmir/gen/input_gen_genmir_fpkmuq_type_pgr.npy"
INPUT_GEN_GEN_MIR_TYPE_HER2_COUNT = DATASET_INPUT + "type/genmir/gen/input_gen_genmir_count_type_her2.npy"
INPUT_GEN_GEN_MIR_TYPE_HER2_FPKM = DATASET_INPUT + "type/genmir/gen/input_gen_genmir_fpkm_type_her2.npy"
INPUT_GEN_GEN_MIR_TYPE_HER2_FPKMUQ = DATASET_INPUT + "type/genmir/gen/input_gen_genmir_fpkmuq_type_her2.npy"
INPUT_GEN_GEN_MIR_SURVIVAL_COUNT = DATASET_INPUT + "survival/genmir/gen/input_gen_genmir_count_sur.npy"
INPUT_GEN_GEN_MIR_SURVIVAL_FPKM = DATASET_INPUT + "survival/genmir/gen/input_gen_genmir_fpkm_sur.npy"
INPUT_GEN_GEN_MIR_SURVIVAL_FPKMUQ = DATASET_INPUT + "survival/genmir/gen/input_gen_genmir_fpkmuq_sur.npy"
INPUT_MIR_GEN_MIR_TYPE_ER = DATASET_INPUT + "type/genmir/mir/input_mir_genmir_type_er.npy"
INPUT_MIR_GEN_MIR_TYPE_PGR = DATASET_INPUT + "type/genmir/mir/input_mir_genmir_type_pgr.npy"
INPUT_MIR_GEN_MIR_TYPE_HER2 = DATASET_INPUT + "type/genmir/mir/input_mir_genmir_type_her2.npy"
INPUT_MIR_GEN_MIR_SURVIVAL = DATASET_INPUT + "survival/genmir/mir/input_mir_genmir_sur.npy"

INPUT_MET_MET_GEN_MIR_TYPE_ER = DATASET_INPUT + "type/metgenmir/met/input_met_metgenmir_type_er.npy"
INPUT_MET_MET_GEN_MIR_TYPE_PGR = DATASET_INPUT + "type/metgenmir/met/input_met_metgenmir_type_pgr.npy"
INPUT_MET_MET_GEN_MIR_TYPE_HER2 = DATASET_INPUT + "type/metgenmir/met/input_met_metgenmir_type_her2.npy"
INPUT_MET_MET_GEN_MIR_SURVIVAL = DATASET_INPUT + "survival/metgenmir/met/input_met_metgenmir_sur.npy"
INPUT_GEN_MET_GEN_MIR_TYPE_ER_COUNT = DATASET_INPUT + "type/metgenmir/gen/input_gen_metgenmir_count_type_er.npy"
INPUT_GEN_MET_GEN_MIR_TYPE_ER_FPKM = DATASET_INPUT + "type/metgenmir/gen/input_gen_metgenmir_fpkm_type_er.npy"
INPUT_GEN_MET_GEN_MIR_TYPE_ER_FPKMUQ = DATASET_INPUT + "type/metgenmir/gen/input_gen_metgenmir_fpkmuq_type_er.npy"
INPUT_GEN_MET_GEN_MIR_TYPE_PGR_COUNT = DATASET_INPUT + "type/metgenmir/gen/input_gen_metgenmir_count_type_pgr.npy"
INPUT_GEN_MET_GEN_MIR_TYPE_PGR_FPKM = DATASET_INPUT + "type/metgenmir/gen/input_gen_metgenmir_fpkm_type_pgr.npy"
INPUT_GEN_MET_GEN_MIR_TYPE_PGR_FPKMUQ = DATASET_INPUT + "type/metgenmir/gen/input_gen_metgenmir_fpkmuq_type_pgr.npy"
INPUT_GEN_MET_GEN_MIR_TYPE_HER2_COUNT = DATASET_INPUT + "type/metgenmir/gen/input_gen_metgenmir_count_type_her2.npy"
INPUT_GEN_MET_GEN_MIR_TYPE_HER2_FPKM = DATASET_INPUT + "type/metgenmir/gen/input_gen_metgenmir_fpkm_type_her2.npy"
INPUT_GEN_MET_GEN_MIR_TYPE_HER2_FPKMUQ = DATASET_INPUT + "type/metgenmir/gen/input_gen_metgenmir_fpkmuq_type_her2.npy"
INPUT_GEN_MET_GEN_MIR_SURVIVAL_COUNT = DATASET_INPUT + "survival/metgenmir/gen/input_gen_metgenmir_count_sur.npy"
INPUT_GEN_MET_GEN_MIR_SURVIVAL_FPKM = DATASET_INPUT + "survival/metgenmir/gen/input_gen_metgenmir_fpkm_sur.npy"
INPUT_GEN_MET_GEN_MIR_SURVIVAL_FPKMUQ = DATASET_INPUT + "survival/metgenmir/gen/input_gen_metgenmir_fpkmuq_sur.npy"
INPUT_MIR_MET_GEN_MIR_TYPE_ER = DATASET_INPUT + "type/metgenmir/mir/input_mir_metgenmir_type_er.npy"
INPUT_MIR_MET_GEN_MIR_TYPE_PGR = DATASET_INPUT + "type/metgenmir/mir/input_mir_metgenmir_type_pgr.npy"
INPUT_MIR_MET_GEN_MIR_TYPE_HER2 = DATASET_INPUT + "type/metgenmir/mir/input_mir_metgenmir_type_her2.npy"
INPUT_MIR_MET_GEN_MIR_SURVIVAL = DATASET_INPUT + "survival/metgenmir/mir/input_mir_metgenmir_sur.npy"

INPUT_METLONG_METLONG_GEN_MIR_TYPE_ER = DATASET_INPUT + "type/metlonggenmir/metlong/input_metlong_metlonggenmir_type_er.npy"
INPUT_METLONG_METLONG_GEN_MIR_TYPE_PGR = DATASET_INPUT + "type/metlonggenmir/metlong/input_metlong_metlonggenmir_type_pgr.npy"
INPUT_METLONG_METLONG_GEN_MIR_TYPE_HER2 = DATASET_INPUT + "type/metlonggenmir/metlong/input_metlong_metlonggenmir_type_her2.npy"
INPUT_METLONG_METLONG_GEN_MIR_SURVIVAL = DATASET_INPUT + "survival/metlonggenmir/metlong/input_metlong_metlonggenmir_sur.npy"
INPUT_GEN_METLONG_GEN_MIR_TYPE_ER_COUNT = DATASET_INPUT + "type/metlonggenmir/gen/input_gen_metlonggenmir_count_type_er.npy"
INPUT_GEN_METLONG_GEN_MIR_TYPE_ER_FPKM = DATASET_INPUT + "type/metlonggenmir/gen/input_gen_metlonggenmir_fpkm_type_er.npy"
INPUT_GEN_METLONG_GEN_MIR_TYPE_ER_FPKMUQ = DATASET_INPUT + "type/metlonggenmir/gen/input_gen_metlonggenmir_fpkmuq_type_er.npy"
INPUT_GEN_METLONG_GEN_MIR_TYPE_PGR_COUNT = DATASET_INPUT + "type/metlonggenmir/gen/input_gen_metlonggenmir_count_type_pgr.npy"
INPUT_GEN_METLONG_GEN_MIR_TYPE_PGR_FPKM = DATASET_INPUT + "type/metlonggenmir/gen/input_gen_metlonggenmir_fpkm_type_pgr.npy"
INPUT_GEN_METLONG_GEN_MIR_TYPE_PGR_FPKMUQ = DATASET_INPUT + "type/metlonggenmir/gen/input_gen_metlonggenmir_fpkmuq_type_pgr.npy"
INPUT_GEN_METLONG_GEN_MIR_TYPE_HER2_COUNT = DATASET_INPUT + "type/metlonggenmir/gen/input_gen_metlonggenmir_count_type_her2.npy"
INPUT_GEN_METLONG_GEN_MIR_TYPE_HER2_FPKM = DATASET_INPUT + "type/metlonggenmir/gen/input_gen_metlonggenmir_fpkm_type_her2.npy"
INPUT_GEN_METLONG_GEN_MIR_TYPE_HER2_FPKMUQ = DATASET_INPUT + "type/metlonggenmir/gen/input_gen_metlonggenmir_fpkmuq_type_her2.npy"
INPUT_GEN_METLONG_GEN_MIR_SURVIVAL_COUNT = DATASET_INPUT + "survival/metlonggenmir/gen/input_gen_metlonggenmir_count_sur.npy"
INPUT_GEN_METLONG_GEN_MIR_SURVIVAL_FPKM = DATASET_INPUT + "survival/metlonggenmir/gen/input_gen_metlonggenmir_fpkm_sur.npy"
INPUT_GEN_METLONG_GEN_MIR_SURVIVAL_FPKMUQ = DATASET_INPUT + "survival/metlonggenmir/gen/input_gen_metlonggenmir_fpkmuq_sur.npy"
INPUT_MIR_METLONG_GEN_MIR_TYPE_ER = DATASET_INPUT + "type/metlonggenmir/mir/input_mir_metlonggenmir_type_er.npy"
INPUT_MIR_METLONG_GEN_MIR_TYPE_PGR = DATASET_INPUT + "type/metlonggenmir/mir/input_mir_metlonggenmir_type_pgr.npy"
INPUT_MIR_METLONG_GEN_MIR_TYPE_HER2 = DATASET_INPUT + "type/metlonggenmir/mir/input_mir_metlonggenmir_type_her2.npy"
INPUT_MIR_METLONG_GEN_MIR_SURVIVAL = DATASET_INPUT + "survival/metlonggenmir/mir/input_mir_metlonggenmir_sur.npy"



########################
######## LABELS ########
########################
# The main folder for all the labels of your training set
# You need at least 30GB of free space for this folder
DATASET_LABELS = MAIN_MDBN_TCGA_BRCA + "labels/"

LABELS_MET_TYPE_ER = DATASET_LABELS + "type/met/label_type_er_met.npy"
LABELS_MET_TYPE_PGR = DATASET_LABELS + "type/met/label_type_pgr_met.npy"
LABELS_MET_TYPE_HER2 = DATASET_LABELS + "type/met/label_type_her2_met.npy"
LABELS_MET_SURVIVAL = DATASET_LABELS + "survival/met/label_sur_met.npy"

LABELS_METLONG_TYPE_ER = DATASET_LABELS + "type/metlong/label_type_er_metlong.npy"
LABELS_METLONG_TYPE_PGR = DATASET_LABELS + "type/metlong/label_type_pgr_metlong.npy"
LABELS_METLONG_TYPE_HER2 = DATASET_LABELS + "type/metlong/label_type_her2_metlong.npy"
LABELS_METLONG_SURVIVAL = DATASET_LABELS + "survival/metlong/label_sur_metlong.npy"

LABELS_GEN_TYPE_ER = DATASET_LABELS + "type/gen/label_type_er_gen.npy"
LABELS_GEN_TYPE_PGR = DATASET_LABELS + "type/gen/label_type_pgr_gen.npy"
LABELS_GEN_TYPE_HER2 = DATASET_LABELS + "type/gen/label_type_her2_gen.npy"
LABELS_GEN_SURVIVAL = DATASET_LABELS + "survival/gen/label_sur_gen.npy"

LABELS_MIR_TYPE_ER = DATASET_LABELS + "type/mir/label_type_er_mir.npy"
LABELS_MIR_TYPE_PGR = DATASET_LABELS + "type/mir/label_type_pgr_mir.npy"
LABELS_MIR_TYPE_HER2 = DATASET_LABELS + "type/mir/label_type_her2_mir.npy"
LABELS_MIR_SURVIVAL = DATASET_LABELS + "survival/mir/label_sur_mir.npy"

LABELS_GEN_MIR_TYPE_ER = DATASET_LABELS + "type/genmir/label_type_er_gen_mir.npy"
LABELS_GEN_MIR_TYPE_PGR = DATASET_LABELS + "type/genmir/label_type_pgr_gen_mir.npy"
LABELS_GEN_MIR_TYPE_HER2 = DATASET_LABELS + "type/genmir/label_type_her2_gen_mir.npy"
LABELS_GEN_MIR_SURVIVAL = DATASET_LABELS + "survival/genmir/label_sur_gen_mir.npy"

LABELS_MET_GEN_MIR_TYPE_ER = DATASET_LABELS + "type/metgenmir/label_type_er_met_gen_mir.npy"
LABELS_MET_GEN_MIR_TYPE_PGR = DATASET_LABELS + "type/metgenmir/label_type_pgr_met_gen_mir.npy"
LABELS_MET_GEN_MIR_TYPE_HER2 = DATASET_LABELS + "type/metgenmir/label_type_her2_met_gen_mir.npy"
LABELS_MET_GEN_MIR_SURVIVAL = DATASET_LABELS + "survival/metgenmir/label_sur_met_gen_mir.npy"

LABELS_METLONG_GEN_MIR_TYPE_ER = DATASET_LABELS + "type/metlonggenmir/label_type_er_metlong_gen_mir.npy"
LABELS_METLONG_GEN_MIR_TYPE_PGR = DATASET_LABELS + "type/metlonggenmir/label_type_pgr_metlong_gen_mir.npy"
LABELS_METLONG_GEN_MIR_TYPE_HER2 = DATASET_LABELS + "type/metlonggenmir/label_type_her2_metlong_gen_mir.npy"
LABELS_METLONG_GEN_MIR_SURVIVAL = DATASET_LABELS + "survival/metlonggenmir/label_sur_metlong_gen_mir.npy"