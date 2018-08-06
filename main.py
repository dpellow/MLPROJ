
import json
import constants
from tcga import load_tcga_data
from utils.param_builder import *
from go import get_flat_go
from nn.mesh import VAE
import numpy as np

import sys
import resource

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 *0.9, hard))
    print "max memory allowed: {}".format((get_memory() /1024.) *0.9)


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

memory_limit()



############ PREDICTION_BY_MUTATION ###################
# for cur_suffix in ["gt","g
# ly","lac","tca"]:
#     for cur_dir in ["high","l ow"]:
#
for cur_tested_file in ["protein_coding_long.txt"]:
    for cur_json in ["gender"]: #

        for dataset in ["BRCA"]:
            meta_groups = None
            meta_groups=[json.load(file("groups/{}.json".format(cur_json)))]

            constants.update_dirs(CANCER_TYPE_u=dataset)
            data_normalizaton = "fpkm_normalized_by_genes_l_inf_norm"
            gene_expression_file_name, phenotype_file_name, survival_file_name, mutation_file_name, mirna_file_name, pval_preprocessing_file_name = build_gdc_params(dataset=dataset, data_normalizaton=data_normalizaton)
            tested_gene_list_file_name=  cur_tested_file # ""mir_warburg_{}_{}.txt".format(cur_dir, cur_suffix) # random_set_file_name #
            total_gene_list_file_name="protein_coding_long.txt"
            var_th_index = 10000
            filter_expression = None
            # filter_expression =  json.load(file("filters/{}.json".format(cur_json)))
            print "fetch tcga data from {}".format(dataset)
            gene_expression_top_var, gene_expression_top_var_headers_rows, gene_expression_top_var_headers_columns, labels_assignment, survival_dataset = load_tcga_data.load(tested_gene_list_file_name=tested_gene_list_file_name, total_gene_list_file_name=total_gene_list_file_name, gene_expression_file_name=gene_expression_file_name, phenotype_file_name=phenotype_file_name, survival_file_name=survival_file_name, var_th_index=var_th_index, filter_expression= filter_expression, meta_groups = meta_groups)
            gene_expression_top_var_rotated = np.rot90(np.flip(gene_expression_top_var, 1), k=-1, axes=(1, 0))
            print "build nn:"
            vae_obj = VAE(gene_expression_top_var_rotated.shape[1])
            vae_obj.build_mesh()
            vae_obj.train_mesh(gene_expression_top_var_rotated, labels_assignment[0])


            print "fetch go"
            get_flat_go.fetch_go_hierarchy()
