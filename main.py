
import json
import time
from tcga import load_tcga_data
from utils.param_builder import *
from nn.go import VAEgo
from pca.pca_im import *
import numpy as np
from go.go_hierarcies import build_hierarcy
from survival_comparison.patients_clustering import find_clusters_and_survival

from constants import app_config
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
def run(var_th_index=app_config['var_th_index'],number_of_neurons=app_config['number_of_neurons'], latent_dim=app_config['latent_dim'], num_of_epochs=app_config['num_of_epochs']):

    reduced_dim_file_name = app_config["reduced_dim_vae_file_name"]
    file(os.path.join(constants.LIST_DIR, app_config["reduced_dim_vae_file_name"]),'w+').write("\n".join([str(x) for x in range(latent_dim)]))
    tested_gene_list_file_name = app_config["possible_vae_input_genes_file_name"]

    for dataset in app_config["datasets"]:

        meta_groups = None
        constants.update_dirs(CANCER_TYPE_u=dataset)
        data_normalizaton = app_config["data_normalizaton"] # "fpkm_normalized_by_genes_l_inf_norm"
        gene_expression_file_name, phenotype_file_name, survival_file_name, mutation_file_name, mirna_file_name, pval_preprocessing_file_name = build_gdc_params(dataset=dataset, data_normalizaton=data_normalizaton)
        total_gene_list_file_name = app_config["total_gene_list_file_name"]
        if var_th_index == -1:
            var_th_index = None

        filter_expression = None
        print "fetch tcga data from {}".format(dataset)
        gene_expression_top_var, gene_expression_top_var_headers_rows, gene_expression_top_var_headers_columns, labels_assignment, survival_dataset = load_tcga_data.load(tested_gene_list_file_name=tested_gene_list_file_name, total_gene_list_file_name=total_gene_list_file_name, gene_expression_file_name=gene_expression_file_name, phenotype_file_name=phenotype_file_name, survival_file_name=survival_file_name, var_th_index=var_th_index, filter_expression= filter_expression, meta_groups = meta_groups)
        gene_expression_top_var_rotated = np.rot90(np.flip(gene_expression_top_var, 1), k=-1, axes=(1, 0))

        print "build nn:"
        print "nans: {}".format(np.count_nonzero(np.isnan(gene_expression_top_var_rotated)))

        roots = app_config['root']
        print roots
        dict_result, go2geneids, geneids2go, get_entrez2ensembl_dict = build_hierarcy(roots)
        print "merging root dictionaries"
        vertices_dict = {}
        for r in roots:
            vertices_dict.update(dict_result[r]['vertices'])
        edges_dict = {}
        for r in roots:
            edges_dict.update(dict_result[r]['edges'])

        results = []
        # VAE
        print "about prepare VAE"
        vae_go_obj = VAEgo(gene_expression_top_var_rotated.shape[1])
        vae_go_obj.build_go(gene_expression_top_var_headers_rows, go2geneids, geneids2go, vertices_dict, edges_dict, number_of_neurons, latent_dim)
        print "done prepare VAE"
        init_epochs = [0]+num_of_epochs[:len(num_of_epochs)-1]
        print "about to calc reduced dim"
        for ind, ie in enumerate(init_epochs):
            gene_expression_test_vae = vae_go_obj.train_go(gene_expression_top_var_headers_rows, gene_expression_top_var_rotated, num_of_epochs[ind],ie)
            #vae_go_obj.train_go(gene_expression_top_var_headers_rows, gene_expression_top_var_rotated, labels_assignment[1])
            vae_projections_fname = "{}_VAE_compress.tsv".format(dataset)
            print "done calc reduced dim"
            vae_go_obj.test_go(gene_expression_test_vae, gene_expression_top_var_headers_columns, survival_dataset[:, 1], latent_dim, vae_projections_fname)

            vae_lr =[]
            print "start loop over VAE. total # of loops: {}".format(app_config["num_randomization"])
            for i in range(app_config["num_randomization"]):
                print "VAE current loop: {}".format(i)
                print "VAE about to calc cluster and survival".format(i)
                vae_lr_iter = find_clusters_and_survival(reduced_dim_file_name=reduced_dim_file_name,
                                                         total_gene_list_file_name=reduced_dim_file_name,
                                                         gene_expression_file_name=vae_projections_fname,
                                                         phenotype_file_name=phenotype_file_name, survival_file_name=survival_file_name,
                                                         var_th_index=None, is_unsupervised=True, start_k=app_config["start_k"],
                                                         end_k=app_config["end_k"], filter_expression=filter_expression, meta_groups=meta_groups,
                                                         clustering_algorithm=app_config["clustering_algorithm"])
                print "VAE done calc cluster and survival".format(i)
                vae_lr.append(-10*np.log10(vae_lr_iter[0]))
                print vae_lr_iter[0]
            print "done loop over VAE with values: var_th_index={}, number_of_neurons={}, latent_dim={}, num_of_epochs={}, num_randomization={}".format(var_th_index,number_of_neurons, latent_dim, num_of_epochs, app_config["num_randomization"])
            avg_vae = np.average(vae_lr)
            var_vae = np.var(vae_lr)
            results.append({"avg" : avg_vae, "var" : var_vae, "type" : "VAE", "epochs" : num_of_epochs[ind]})
            print "current VAE results:\n" \
                  "{}".format(results[-1])
            # PCA
        pca_obj = PCA_obj()
        gene_expression_top_var_pca, gene_expression_top_var_headers_rows_pca, gene_expression_top_var_headers_columns_pca, labels_assignment_pca, survival_dataset_pca = load_tcga_data.load(tested_gene_list_file_name=app_config['actual_vae_input_genes_file_name'], total_gene_list_file_name=total_gene_list_file_name, gene_expression_file_name=gene_expression_file_name, phenotype_file_name=phenotype_file_name, survival_file_name=survival_file_name, var_th_index=None, filter_expression= filter_expression, meta_groups = meta_groups)

        tmp = gene_expression_top_var_headers_rows_pca
        gene_expression_top_var_headers_rows_pca = gene_expression_top_var_headers_columns_pca
        gene_expression_top_var_headers_columns_pca = tmp
        gene_expression_top_var_pca = np.rot90(np.flip(gene_expression_top_var_pca, 1), k=-1, axes=(1, 0))

        gene_expression_test_pca = pca_obj.pca_train(gene_expression_top_var_headers_rows_pca,gene_expression_top_var_pca, survival_dataset[:, 1], latent_dim)
        pca_projections_fname = "{}_PCA_compress.tsv".format(dataset)
        pca_obj.pca_test(gene_expression_test_pca, gene_expression_top_var_headers_rows_pca, survival_dataset[:, 1], latent_dim, pca_projections_fname)

        pca_lr =[]
        for i in range(app_config["num_randomization"]):
            print "PCA current loop: {}".format(i)
            pca_lr_iter = find_clusters_and_survival(reduced_dim_file_name=reduced_dim_file_name,
                                                     total_gene_list_file_name=reduced_dim_file_name,
                                                     gene_expression_file_name=pca_projections_fname,
                                                     phenotype_file_name=phenotype_file_name, survival_file_name=survival_file_name,
                                                     var_th_index=None, is_unsupervised=True, start_k=app_config["start_k"],
                                                     end_k=app_config["end_k"], filter_expression=filter_expression, meta_groups=meta_groups,
                                                     clustering_algorithm=app_config["clustering_algorithm"])
            pca_lr.append(-10*np.log10(pca_lr_iter[0]))
            print pca_lr_iter[0]
        avg_pca = np.average(pca_lr)
        var_pca = np.var(pca_lr)
        results.append({"avg" : avg_pca, "var" : var_pca, "type" : "PCA"})
        print "current var results:\n" \
              "{}".format(results[-1])
        ###
        ##premutated VAE HERE
        ###


        print "final VAE results:\n" \
              "{}".format(results[-2])

        print "final PCA results:\n" \
              "{}".format(results[-1])

        print "done running over: var_th_index={}, number_of_neurons={}, latent_dim={}, num_of_epochs={}".format(var_th_index,number_of_neurons, latent_dim, num_of_epochs)

	return results
if __name__ == '__main__':
    run()


        # Randomly permuted VAE
    #            pvals_random_vae = []
    #            for i in range(app_config["num_randomization"]):
    #                gene_expression_top_var_permuted = np.random.permutation(gene_expression_top_var)
    #                gene_expression_top_var_permuted_rotated = np.rot90(np.flip(gene_expression_top_var_permuted, 1), k=-1, axes=(1, 0))
    #                vae_projections_fname =dataset + "_VAE_projections_random_"+str(i)+".tsv"
    #                gene_expression_test_vae = vae_go_obj.train_go(gene_expression_top_var_headers_rows, gene_expression_top_var_permuted_rotated, gene_expression_top_var_headers_columns,  survival_dataset[:, 1], "VAE_weights_random_"+str(i)+".h5")
    #                vae_go_obj.test_go(gene_expression_test_vae, gene_expression_top_var_headers_columns, survival_dataset[:, 1],vae_projections_fname)
    #                print "current loop: {}".format(i)
    #                lr = (find_clusters_and_survival(reduced_dim_file_name=reduced_dim_file_name,
    #                                            total_gene_list_file_name=reduced_dim_file_name, gene_list_pca_name=gene_list_pca_name,
    #                                            gene_expression_file_name=vae_projections_fname,
    #                                            phenotype_file_name=phenotype_file_name, survival_file_name=survival_file_name,
    #                                            var_th_index=None, is_unsupervised=True, start_k=app_config["start_k"],
    #                                            end_k=app_config["end_k"], filter_expression=filter_expression, meta_groups=meta_groups,
    #                                            clustering_algorithm=app_config["clustering_algorithm"]))
    #                pvals_random_vae.append(lr[0])
    #            avg_random_VAE = sum(pvals_random_vae)/float(len(pvals_random_vae))
    #            var_random_VAE = np.var(pvals_random_vae)
    #            f.write("Average random VAE: " + str(avg_random_VAE) + "," + "Variance random VAE: " + str(var_random_VAE) + "," + str(time.time()) +"\r\n")
