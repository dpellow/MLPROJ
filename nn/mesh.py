import time
import re
import numpy as np
import constants
import os
import math
import pandas as pd
import keras
from keras.layers import Dense, Input, concatenate, Lambda, BatchNormalization
from keras import backend as K
from utils.ensembl2entrez import entrez2ensembl_convertor
from keras.utils import plot_model
from keras.models import Model
from keras import metrics
from constants import app_config
from keras.callbacks import History

batch_size = app_config['batch_size']
epsilon_std = 1.0

class VAEmesh:

    def __init__(self, original_dim):
        self.vae=None
        self.original_dim = original_dim
        self.encoder = None

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon


    def build_mesh(self, num_roots, vertices_dict, latent_dim, number_of_neurons, threshold):
        self.latent_dim = latent_dim
        # complicated way to estimate how many layers and how many nodes in each layer
        # should really get this from GO vae
        num_layers = max([max(x['depth']) for x in vertices_dict.values()])

        max_genes = 1000 # fix this
        max_nodes = len(vertices_dict)
        estimate_num_nodes = math.ceil(max_nodes*threshold/float(max_genes))
        estimate_layer_delta = int(math.ceil((2*(estimate_num_nodes - num_roots))/float(num_layers*(num_layers-1))))
        layer_sizes_top_down = [num_roots]
        num_nodes = num_roots
        for i in range(num_layers-1):
            num_nodes += estimate_layer_delta
            layer_sizes_top_down.append(num_nodes)
        layer_sizes_bottom_up = list(reversed(layer_sizes_top_down))

        inputs = Input(shape=(self.original_dim,))
        x = BatchNormalization()(inputs)
        for l_size in layer_sizes_bottom_up:
            x = BatchNormalization()(Dense(l_size*number_of_neurons, activation='relu')(x))

        z_mean = BatchNormalization()(Dense(latent_dim)(x))
        z_log_var = BatchNormalization()(Dense(latent_dim)(x))

        z = BatchNormalization()(Lambda(self.sampling, output_shape=(latent_dim,))([z_mean, z_log_var]))

        h = z
        for l_size in layer_sizes_top_down:
            h = BatchNormalization()(Dense(l_size*number_of_neurons, activation='relu')(h))
        outputs = Dense(self.original_dim, activation='sigmoid')(h)

        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        # self.encoder.summary()
        plot_model(self.encoder, to_file=os.path.join(constants.OUTPUT_GLOBAL_DIR, "encoder_mesh_{}.svg".format(time.time())))
        
        # instantiate VAE model
        self.vae = Model(inputs, outputs)

        if app_config["loss_function"] == "mse":
            reconstruction_loss = self.original_dim * metrics.mse(inputs, outputs)
        if app_config["loss_function"] == "cross_ent":
            reconstruction_loss = self.original_dim * metrics.binary_crossentropy(inputs, outputs)

        if app_config["is_variational"]:
            kl = -0.5 * K.sum(1 + z_log_var - K.exp(z_log_var) - K.square(z_mean), axis=1)
            self.vae.add_loss(K.mean(reconstruction_loss + kl))  # weighting? average?
        else:
            self.vae.add_loss(reconstruction_loss)
        self.vae.compile(optimizer='rmsprop')  # , loss=loss

        plot_model(self.vae, to_file=os.path.join(constants.OUTPUT_GLOBAL_DIR, "model_mesh_{}.svg".format(time.time())))


    def train_mesh(self, gene_ids, gene_expressions, num_of_epochs, init_epoch, vae_weights_fname = "VAE_weights.h5"):
        if app_config["load_weights"]:
            self.vae = self.vae.load_weights(os.path.join(constants.OUTPUT_GLOBAL_DIR, vae_weights_fname))
        else:
            hist = self.vae.fit(gene_expressions,
                shuffle=True,
                epochs=num_of_epochs,
                batch_size=batch_size,
                validation_split = 0.1,
                initial_epoch=init_epoch
                )
            print "last loss: {}".format( hist.history['val_loss'][-1])
            print "last var_loss: {}".format(hist.history['loss'][-1])
            #self.vae.save_weights(os.path.join(constants.OUTPUT_GLOBAL_DIR, vae_weights_fname))
        return hist.history['loss'][-1],hist.history['val_loss'][-1]


    def test_mesh(self, gene_expressions, patients_list, latent_dim, mesh_projections_fname = "mesh_compress.tsv"):
        print np.shape(gene_expressions)
        latent_space = self.encoder.predict([gene_expressions], batch_size=batch_size)
        print np.shape(latent_space[2])

        print "Saving mesh data.."
        # x_projected = np.insert(x_projected,[0], np.array(latent_dim),axis = 0)
        # np.save(os.path.join(constants.OUTPUT_GLOBAL_DIR, "PCA_compress.txt"), x_projected)
        latent_data = latent_space[2].T
        pca_data = pd.DataFrame(latent_data, index=range(latent_dim), columns=patients_list)
        pca_data.index.name = 'mesh'
        pca_data.to_csv(os.path.join(constants.OUTPUT_GLOBAL_DIR, mesh_projections_fname), sep='\t')
