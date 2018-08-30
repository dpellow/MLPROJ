
import json
import constants
from tcga import load_tcga_data
from utils.param_builder import *
from go import get_flat_go
from nn.mesh import VAEmesh
from nn.go import VAEgo
from nn.pca import *
import numpy as np
from go.go_hierarcies import build_hierarcy
from survival_comparison.patients_clustering import find_clusters_and_survival

from constants import app_config
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
# for cur_suffix in ["gt","gly","lac","tca"]:
#     for cur_dir in ["high","low"]:
#
for cur_tested_file in ["protein_coding_long.txt"]:
    for cur_json in ["gender"]: #

        for dataset in app_config["datasets"]:
            meta_groups = None
            meta_groups=[json.load(file("groups/{}.json".format(cur_json)))]

            constants.update_dirs(CANCER_TYPE_u=dataset)
            data_normalizaton = app_config["data_normalizaton"] # "fpkm_normalized_by_genes_l_inf_norm"
            gene_expression_file_name, phenotype_file_name, survival_file_name, mutation_file_name, mirna_file_name, pval_preprocessing_file_name = build_gdc_params(dataset=dataset, data_normalizaton=data_normalizaton)
            tested_gene_list_file_name = cur_tested_file # ""mir_warburg_{}_{}.txt".format(cur_dir, cur_suffix) # random_set_file_name #
            total_gene_list_file_name=app_config["total_gene_list_file_name"]
            var_th_index = app_config["var_th_index"]
            gene_list_pca_name = app_config["gene_list_pca"]
            filter_expression = None
            # filter_expression =  json.load(file("filters/{}.json".format(cur_json)))
            print "fetch tcga data from {}".format(dataset)
            gene_expression_top_var, gene_expression_top_var_headers_rows, gene_expression_top_var_headers_columns, labels_assignment, survival_dataset, gene_expression_top_var_pca, gene_expression_top_var_headers_rows_pca = load_tcga_data.load(tested_gene_list_file_name=tested_gene_list_file_name, total_gene_list_file_name=total_gene_list_file_name, gene_expression_file_name=gene_expression_file_name, gene_list_pca_name=gene_list_pca_name, phenotype_file_name=phenotype_file_name, survival_file_name=survival_file_name, var_th_index=var_th_index, filter_expression= filter_expression, meta_groups = meta_groups)
            gene_expression_top_var_rotated = np.rot90(np.flip(gene_expression_top_var, 1), k=-1, axes=(1, 0))
            gene_expression_top_var_rotated_pca = np.rot90(np.flip(gene_expression_top_var_pca, 1), k=-1, axes=(1, 0)) #pca genes

            print "build nn:"
            print "nans: {}".format(np.count_nonzero(np.isnan(gene_expression_top_var_rotated)))
            # vae_mesh_obj = VAEmesh(gene_expression_top_var_rotated.shape[1])
            # vae_mesh_obj.build_mesh()
            # vae_mesh_obj.train_mesh(gene_expression_top_var_rotated, labels_assignment[0])
            roots = app_config['root'] # ''GO:0005575']] # ['GO:0044429']
            print roots
            dict_result, go2geneids, geneids2go, get_entrez2ensembl_dict = build_hierarcy(roots)
            print "merging root dictionaries"
            vertices_dict = {}
            for r in roots:
                vertices_dict.update(dict_result[r]['vertices'])
            edges_dict = {}
            for r in roots:
                edges_dict.update(dict_result[r]['edges'])


             # VAE
            vae_go_obj = VAEgo(gene_expression_top_var_rotated.shape[1])
            vae_go_obj.build_go(gene_expression_top_var_headers_rows, go2geneids, geneids2go, vertices_dict, edges_dict)
            gene_expression_test_vae = vae_go_obj.train_go(gene_expression_top_var_headers_rows, gene_expression_top_var_rotated, gene_expression_top_var_headers_columns, survival_dataset[:, 1]) #vae_go_obj.train_go(gene_expression_top_var_headers_rows, gene_expression_top_var_rotated, labels_assignment[1])
            vae_go_obj.test_go(gene_expression_test_vae, gene_expression_top_var_headers_columns, survival_dataset[:, 1])

            # PCA
            pca_obj = PCA_obj()
            gene_expression_test_pca = pca_obj.pca_train(gene_expression_top_var_headers_rows_pca,gene_expression_top_var_rotated_pca, survival_dataset[:, 1])
            pca_obj.pca_test(gene_expression_top_var_headers_rows_pca, gene_expression_test_pca, survival_dataset[:, 1])

	        # Randomly permuted VAE
            pvals = []
            for i in range(app_config["num_randomization"]):
                gene_expression_top_var_permuted = np.random.permutation(gene_expression_top_var)
                gene_expression_top_var_permuted_rotated = np.rot90(np.flip(gene_expression_top_var_permuted, 1), k=-1, axes=(1, 0))
		        ### NEED TO ADD PARAM TO CHOOSE WHERE TO WRITE OUTPUT FILE
                vae_go_obj.train_go(gene_expression_top_var_headers_rows, gene_expression_top_var_permuted_rotated, survival_dataset[:, 1]) ### DO WE NEED TO RUN build_go AGAIN TOO??
                lr = (find_clusters_and_survival(reduced_dim_file_name=reduced_dim_file_name,
                                            total_gene_list_file_name=None,
                                            gene_expression_file_name=gene_expression_file_name,
                                            phenotype_file_name=phenotype_file_name, survival_file_name=survival_file_name,
                                            var_th_index=var_th_index, is_unsupervised=True, start_k=app_config["start_k"],
                                            end_k=app_config["end_k"], filter_expression=filter_expression, meta_groups=meta_groups,
                                            clustering_algorithm=app_config["clustering_algorithm"]))
                pvals.append(lr[0])
	        avg = sum(pvals)/float(len(pvals))


            # K-mean & survival
            # start_k = 2
            # end_k = 2
            # clustering_algorithm = "euclidean"
            # tested_gene_list_file_name = #PCA/VAE file#
            # lr_results_global = find_clusters_and_survival(reduced_dim_file_name=reduced_dim_file_name,
            #                            total_gene_list_file_name=None,
            #                            gene_expression_file_name=gene_expression_file_name,
            #                            phenotype_file_name=phenotype_file_name, survival_file_name=survival_file_name,
            #                            var_th_index=var_th_index, is_unsupervised=True, start_k=start_k,
            #                            end_k=end_k, filter_expression=filter_expression, meta_groups=meta_groups,
            #                            clustering_algorithm=clustering_algorithm)



            # print "fetch go"
            # get_flat_go.fetch_go_hierarchy()
