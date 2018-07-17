import json
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CONF = "env/config/conf.json"
config_json = json.load(file(os.path.join(dir_path, PATH_TO_CONF)))

USE_CACHE = True
PHENOTYPE_FORMAT = "GDC"
DATASET_TYPE = "GDC-TCGA"
CANCER_TYPE = "SKCM"
BASE_PROFILE= config_json['BASE_PROFILE']
BASE_DATASET= os.path.join(BASE_PROFILE, DATASET_TYPE, CANCER_TYPE)
CACHE_DIR = os.path.join(BASE_DATASET, "cache")
DICTIONARIES_DIR = os.path.join(BASE_PROFILE, "dictionaries")
OUTPUT_DIR = os.path.join(BASE_DATASET, "output")
OUTPUT_GLOBAL_DIR = os.path.join(BASE_PROFILE, "output")
TCGA_DATA_DIR = os.path.join(BASE_DATASET, "tcga_data")
GO_DIR = os.path.join(BASE_PROFILE, "GO")
CACHE_GLOBAL_DIR = os.path.join(BASE_PROFILE, "cache_global")
LIST_DIR = os.path.join(BASE_PROFILE, "list")

SEPARATOR = "@%@"

LABEL_ID = "sample_type.samples"
PRIMARY_TUMOR = "Primary Tumor"
METASTATIC = "Metastatic"

LABELS_NORMAL = "labels_normal"
LABELS_SHUFFLE = "labels_shuffle"
LABELS_RANDOM = "labels_random"
LABELS_ALTERNATED = "labels_alternated"
LABELS_INVERTED = "labels_inverted"

ENSEMBL_TO_GENE_SYMBOLS = "ensembl2gene_symbol.txt"
ENSEMBL_TO_ENTREZ = "ensembl2entrez.txt"

GO_OBO_URL = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
GO_ASSOCIATION_GENE2GEO_URL = 'https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz'
GO_FILE_NAME = 'go-basic.obo'
GO_ASSOCIATION_FILE_NAME = "gene2go"

NUM_GTE = "gte"
NUM_GT = "gt"
NUM_EQ = "eq"
NUM_LTE = "lte"
NUM_LT = "lt"
NUM_NE = "ne"
NUM_ALL_OPS = [NUM_EQ, NUM_GTE, NUM_GT, NUM_LTE, NUM_LT, NUM_NE]


FROM_DISK = "FROM_DISK"
ON_THE_FLY = "ON_THE_FLY"
FILTER_KEYWORDS = ["_label", "_name"]
ALL_CANCER_TYPES = ["ESCA", "LAML", "ACC", "CHOL", "BLCA", "BRCA", "CESC", "COAD", "UCEC", "GBM", "HNSC", "KICH", "KIRC", "KIRP", "DLBC", "LIHC", "LGG", "LUAD", "LUSC", "SKCM", "MESO", "UVM", "PANCAN", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "STAD", "TGCT", "THYM", "THCA", "UCS"]
ALL_TUMOR_TYPES = ["Primary Tumor", "Metastatic", "Additional - New Parimary", "Additional Metatatic", "Primary Blood Derived Cancer - Peripheral Blood", "Blood Derived Cancer - Bone Marrow, Post-treatment", "Primary Blood Derived Cancer - Bone Marrow", "Recurrent Blood Derived Cancer - Peripheral Blood", "Recurrent Tumor"]
def update_dirs(BASE_DIR=config_json["BASE_PROFILE"], DATASET_DIR=None, DATASET_TYPE_u = "GDC-TCGA", CANCER_TYPE_u = "SKCM"):

    global BASE_PROFILE
    global CACHE_DIR
    global OUTPUT_DIR
    global LIST_DIR
    global TCGA_DATA_DIR
    global DICTIONARIES_DIR
    global CANCER_TYPE
    global DATASET_TYPE
    global BASE_DATASET
    global GO_DIR
    global CACHE_GLOBAL_DIR
    global OUTPUT_GLOBAL_DIR
    BASE_PROFILE=BASE_DIR
    DATASET_TYPE = DATASET_TYPE_u
    CANCER_TYPE = CANCER_TYPE_u
    if DATASET_DIR is None:
        DATASET_DIR = "{}/{}/".format(DATASET_TYPE,CANCER_TYPE)

    BASE_DATASET= os.path.join(BASE_PROFILE,DATASET_DIR)
    CACHE_DIR = os.path.join(BASE_DATASET, "cache")
    DICTIONARIES_DIR = os.path.join(BASE_PROFILE, "dictionaries")
    OUTPUT_DIR = os.path.join(BASE_DATASET, "output")
    TCGA_DATA_DIR = os.path.join(BASE_DATASET, "tcga_data")
    GO_DIR = os.path.join(BASE_PROFILE, "GO")
    LIST_DIR = os.path.join(BASE_PROFILE, "list")
    CACHE_GLOBAL_DIR = os.path.join(BASE_PROFILE, "cache_global")
    OUTPUT_GLOBAL_DIR = os.path.join(BASE_PROFILE, "output")
update_dirs()