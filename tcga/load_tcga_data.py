from matplotlib import style
style.use("ggplot")
import logging
sh = logging.StreamHandler()
logger = logging.getLogger("log")
logger.addHandler(sh)
from infra import *

sys.path.insert(0, '../')


############################ () cluster and enrichment #############################


# () main
def load(tested_gene_list_file_name, total_gene_list_file_name, gene_expression_file_name, gene_list_pca_name, phenotype_file_name, survival_file_name, gene_filter_file_name=None, tested_gene_list_path=None, total_gene_list_path=None, gene_expression_path=None, phenotype_path=None, gene_filter_file_path=None, var_th_index=None, is_unsupervised=True, start_k=2, end_k=2, meta_groups=None, filter_expression=None, clustering_algorithm="euclidean"):
    stopwatch = Stopwatch()
    stopwatch.start()
    data = load_integrated_ge_data(tested_gene_list_file_name=tested_gene_list_file_name, total_gene_list_file_name=total_gene_list_file_name, gene_expression_file_name=gene_expression_file_name, gene_list_pca_name=gene_list_pca_name, phenotype_file_name=phenotype_file_name, survival_file_name=survival_file_name, var_th_index=var_th_index, meta_groups=meta_groups, filter_expression=filter_expression)
    print stopwatch.stop("Done loading integrated data")
    if data is None:
        print "insufficient data"
        return None, None, None, None, None
    return data

