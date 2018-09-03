import os
import constants
import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
from constants import app_config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC

class PCA_obj:

    def __init__(self):
        self.pca = None

    def pca_train(self,patients_list, gene_expression_data, y_data, latent_dim):
        print "Original dim:" + str(np.shape(gene_expression_data)[1])
        print np.shape(gene_expression_data)
        print "Num of patients: " + str(len(patients_list))
        print "my dim: {}".format(latent_dim)
        if app_config["split_data"]:
            col = np.shape(gene_expression_data)[0]
            ratio = int(math.floor(col * 0.9))
            print len(gene_expression_data[:ratio][:])
            print len(gene_expression_data[ratio:][:])
            x_train = np.array(gene_expression_data[:][:ratio]).astype(np.float64)
            x_test = np.array(gene_expression_data[:][ratio:]).astype(np.float64)
            y_train = np.array(y_data[:ratio]).astype(np.int32)
            y_test = np.array(y_data[ratio:]).astype(np.int32)

            self.pca = PCA(n_components=latent_dim)
            self.pca.fit(x_train)
        else:
            x_train = np.array(gene_expression_data).astype(np.float64)
            self.pca = PCA(n_components=latent_dim)
            self.pca.fit(x_train)
            x_test = x_train

        print "New dim:" + str(len(self.pca.explained_variance_ratio_.cumsum()))
        print "variance_ratio:" + str(self.pca.explained_variance_ratio_.cumsum())

        # mse-loss
        x_projected = self.pca.inverse_transform(self.pca.transform(x_train))
        loss =((x_train-x_projected)**2).mean()
        print "MSE-loss: " + str(loss)
        print "x_shape is:" + str(np.shape(x_test))
        return x_test

        ### SVM ###
        # if app_config["use_svm"]:
        #     for kernel in ('linear', 'poly', 'rbf'):
        #         clf = SVC(kernel =kernel)
        #         clf.fit(x_train_pca, y_train)
        #         print 'score-svm-{}:'.format(kernel) + str(clf.score(x_test_pca, y_test))

    def pca_test(self,x_test,patients_list, y_data, latent_dim, pca_projections_fname = "PCA_compress.tsv"):
        print "x_shape is:" + str(np.shape(x_test))
        x_test_pca = self.pca.transform(x_test)
        print "Saving PCA data.."
        # x_projected = np.insert(x_projected,[0], np.array(app_config["latent_dim"]),axis = 0)
        # np.save(os.path.join(constants.OUTPUT_GLOBAL_DIR, "PCA_compress.txt"), x_projected)
        x_test_pca = x_test_pca.T
        pca_data = pd.DataFrame(x_test_pca, index=range(latent_dim), columns=patients_list)
        pca_data.index.name = 'PCA'
        pca_data.to_csv(os.path.join(constants.OUTPUT_GLOBAL_DIR, pca_projections_fname), sep='\t')

    ### visualize two first components
    # plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1],
    #             edgecolor='none', alpha=0.5,
    #             c= y_train,cmap='gist_rainbow')
    # plt.xlabel('component 1')
    # plt.ylabel('component 2')
    # plt.colorbar()
    # plt.savefig("test.png")
    # plt.cla()
