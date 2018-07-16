
import json
import constants
from tcga import load_tcga_data
from utils.param_builder import *


############ PREDICTION_BY_MUTATION ###################
# for cur_suffix in ["gt","g
# ly","lac","tca"]:
#     for cur_dir in ["high","l ow"]:
#
for cur_tested_file in ["protein_coding_long.txt"]:
    for cur_json in ["gender"]: #

        for dataset in ["BRCA"]:
            meta_groups = None
            # meta_groups=[json.load(file("groups/{}.json".format(cur_json)))]

            constants.update_dirs(CANCER_TYPE_u=dataset)
            data_normalizaton = "fpkm"
            gene_expression_file_name, phenotype_file_name, survival_file_name, mutation_file_name, mirna_file_name, pval_preprocessing_file_name = build_gdc_params(dataset=dataset, data_normalizaton=data_normalizaton)
            tested_gene_list_file_name=  cur_tested_file # ""mir_warburg_{}_{}.txt".format(cur_dir, cur_suffix) # random_set_file_name #
            total_gene_list_file_name="protein_coding_long.txt"
            var_th_index = None
            filter_expression = None
            # filter_expression =  json.load(file("filters/{}.json".format(cur_json)))
            print "process {}".format(dataset)
            load_tcga_data.load(tested_gene_list_file_name=tested_gene_list_file_name, total_gene_list_file_name=total_gene_list_file_name, gene_expression_file_name=gene_expression_file_name, phenotype_file_name=phenotype_file_name, survival_file_name=survival_file_name, var_th_index=var_th_index, filter_expression= filter_expression, meta_groups = meta_groups)
