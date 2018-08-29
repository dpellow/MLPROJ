import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn import metrics
from constants import app_config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def pca_implement(gene_ids, gene_expression_data, y_data):
    print "Original dim:" + str(np.shape(gene_expression_data)[1])

    col = np.shape(gene_expression_data)[0]
    ratio = int(math.floor(col * 0.9))
    print len(gene_expression_data[:ratio][:])
    print len(gene_expression_data[ratio:][:])
    x_train = np.array(gene_expression_data[:][:ratio]).astype(np.float64)
    x_test = np.array(gene_expression_data[:][ratio:]).astype(np.float64)
    y_train = np.array(y_data[:ratio]).astype(np.int32)
    y_test = np.array(y_data[ratio:]).astype(np.int32)

    pca = PCA(n_components=app_config["latent_dim"])
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    print "New dim:" + str(len(pca.explained_variance_ratio_.cumsum()))
    print "variance_ratio:" + str(pca.explained_variance_ratio_.cumsum())

    # mse-loss
    x_projected = pca.inverse_transform(x_train_pca)
    loss =((x_train-x_projected)**2).mean()
    print "MSE-loss: " + str(loss)


    ### visualize two first components
    # plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1],
    #             edgecolor='none', alpha=0.5,
    #             c= y_train,cmap='gist_rainbow')
    # plt.xlabel('component 1')
    # plt.ylabel('component 2')
    # plt.colorbar()
    # plt.savefig("test.png")
    # plt.cla()

    ### SVM ###
    for kernel in ('linear', 'poly', 'rbf'):
        clf = SVC(kernel =kernel)
        clf.fit(x_train_pca, y_train)
        print 'score-svm-{}:'.format(kernel) + str(clf.score(x_test_pca, y_test))


if __name__ == '__main__':
    print pca_implement()