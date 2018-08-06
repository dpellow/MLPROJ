import numpy as np

import math
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Lambda
from keras import backend as K
from keras import metrics
from utils.ensembl2entrez import ensembl2entrez_convertor


batch_size = 10
latent_dim = 2
intermediate_dim = 10000
epochs = 50
epsilon_std = 1.0

class VAE:

    def __init__(self, original_dim):
        self.vae=None
        self.original_dim = original_dim

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon


    def build_mesh(self, gene_list, go2genes, genes2go, vertices, edges):
        input_layer = Input(shape=(self.original_dim,))

        for cur_vertix in vertices:
            if cur_vertix["isleaf"]:
                indices = np.in1d([ensembl2entrez_convertor(x) for x in gene_list], go2genes[cur_vertix["cur_vertix"]])
                cur_neuron = Lambda(lambda x: x[indices], output_shape=((1,)))(input_layer)
                cur_vertix["neuron"] = cur_neuron

        vertices_copy = vertices
        neuron_count=0
        is_converged=False
        while not is_converged:
            is_converged = True
            for k, v  in vertices_copy.iteritems():
                is_connectable = True
                for cur_child in v["obj"].children:
                    if not cur_child.has_key["neuron"]:
                        is_connectable = False
                if is_connectable:
                    v["neuron"] = Concatenate()([cur_child["neuron"] for cur_child in v["obj"].children])
                    neuron_count+=1
                    is_converged = False
        print "converged with {} intermediate neurons".format(neuron_count)




        # h = Dense(intermediate_dim, activation='relu')(x)
        # z_mean = Dense(latent_dim)(h)
        # z_log_var = Dense(latent_dim)(h)
        #
        #
        #
        # z = Lambda(self.sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        #
        # # we instantiate these layers separately so as to reuse them later
        # decoder_h = Dense(intermediate_dim, activation='relu')
        # decoder_mean = Dense(self.original_dim, activation='sigmoid')
        # h_decoded = decoder_h(z)
        # x_decoded_mean = decoder_mean(h_decoded)
        #
        # # instantiate VAE model
        # self.vae = Model(x, x_decoded_mean)
        #
        # # Compute VAE loss
        # vae_loss = self.original_dim * metrics.mean_squared_error(x, x_decoded_mean)
        #
        # xent_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean) # self.original_dim * metrics.mean_squared_error(x, x_decoded_mean)
        # kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # vae_loss =  K.mean(xent_loss + kl_loss)
        #
        # self.vae.add_loss(vae_loss)
        # self.vae.compile(optimizer='rmsprop')
        # self.vae.summary()

    def train_mesh(self, x_data, y_data):
        # train the VAE on MNIST digits

        ratio = int(math.floor(len(x_data) * 0.9))-82
        x_train = np.array(x_data[:ratio]).astype(np.float32)
        x_test = np.array(x_data[ratio:-3]).astype(np.float32)

        self.vae.fit(x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))


        # encoder = Model(x, [z_mean, z_log_var, z], name='encoder')
        # predictions = encoder.predict(x_test)






