import numpy as np

import math
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics


batch_size = 10
latent_dim = 2
epochs = 50
epsilon_std = 1.0

class VAE:

    def __init__(self, original_dim):
        self.vae=None
        self.original_dim = original_dim
        self.intermediate_dim = original_dim/2


    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon


    def load_mesh(self):
        x = Input(shape=(self.original_dim,))
        h = Dense(self.intermediate_dim, activation='relu')(x)
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)

        z = Lambda(self.sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        # instantiate VAE model
        self.vae = Model(x, x_decoded_mean)

        # Compute VAE loss
        vae_loss = self.original_dim * metrics.mean_squared_error(x, x_decoded_mean)

        xent_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean) # self.original_dim * metrics.mean_squared_error(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = xent_loss # K.mean(xent_loss + kl_loss)

        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam') # rmsprop
        self.vae.summary()

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






