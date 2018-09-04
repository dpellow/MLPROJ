import numpy as np

import math
from keras.layers import Input, Dense, Lambda, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics

epsilon_std = 1.0

class VAEmesh:

    def __init__(self, original_dim):
        self.vae=None
        self.original_dim = original_dim


    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon


    def build_mesh(self, roots, vertices_dict, latent_dim, number_of_neurons, threshold):

        # complicated way to estimate how many layers and how many nodes in each layer
        # should really get this from GO vae
        num_layers = max([max(vertices_dict[r]['depth']) for r in roots])
        max_genes = 1000 # fix this
        max_nodes = len(vertices_dict)
        estimate_num_nodes = math.ceil(max_nodes*threshold/float(max_genes))
        num_roots = len(roots)
        estimate_layer_delta = math.ceil((2*(estimate_num_nodes - num_roots))/float(num_layers*(num_layers-1)))
        layer_sizes_top_down = [num_roots]
        num_nodes = num_roots
        for i in range(num_layers-1):
            num_nodes += estimate_layer_delta
            layer_sizes.append(num_nodes)
        layer_sizes_bottom_up = list(reverse(layer_sizes_top_down))

        inputs = Input(shape=(self.original_dim))
        x = BatchNormalization()(inputs)
        for l_size in layer_sizes_bottom_up:
            x = BatchNormalization()(Dense (l_size*num_neurons, activation='relu')(x))

        z_mean = BatchNormalization()(Dense(latent_dim)(x))
        z_log_var = BatchNormalization()(Dense(latent_dim)(x))

        z = BatchNormalization()(Lambda(self.sampling, output_shape=(latent_dim,))([z_mean, z_log_var]))

        h = z
        for l_size in layer_sizes_top_down:
            h = BatchNormalization()(Dense(l_size*num_neurons, activation='relu')(h))
        outputs = Dense(self.original_dim, activation='sigmoid')(h)

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
        print "number of inputs: {}".format(len(inputs))
        print "number of outputs: {}".format(len(outputs))

        plot_model(self.vae, to_file=os.path.join(constants.OUTPUT_GLOBAL_DIR, "model_mesh_{}.svg".format(time.time())))


    def train_mesh(self, gene_ids, gene_expressions, num_of_epochs, init_epoch, vae_weights_fname = "VAE_weights.h5"):
        if app_config["load_weights"]:
            self.vae = self.vae.load_weights(os.path.join(constants.OUTPUT_GLOBAL_DIR, vae_weights_fname))
        else:
            self.vae.fit(gene_expressions,
                shuffle=True,
                epochs=num_of_epochs,
                batch_size=batch_size,
                validation_split = 0.1,
                initial_epoch=init_epoch
                )


    def test_mesh(gene_expressions, patients_list, latent_dim, mesh_projections_fname = "mesh_compress.tsv"):
        print np.shape(gene_expressions)
        latent_space = self.encoder.predict([x for x in gene_expressions.T], batch_size=batch_size)
        print np.shape(latent_space[2])

        print "Saving mesh data.."
        # x_projected = np.insert(x_projected,[0], np.array(latent_dim),axis = 0)
        # np.save(os.path.join(constants.OUTPUT_GLOBAL_DIR, "PCA_compress.txt"), x_projected)
        latent_data = latent_space[2].T
        pca_data = pd.DataFrame(latent_data, index=range(latent_dim), columns=patients_list)
        pca_data.index.name = 'mesh'
        pca_data.to_csv(os.path.join(constants.OUTPUT_GLOBAL_DIR, mesh_projections_fname), sep='\t')
