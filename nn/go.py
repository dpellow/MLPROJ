import re
import numpy as np
import constants
import os
import math
from keras.models import Model
from keras.layers import Dense, Input, concatenate, Lambda
from keras import backend as K
from keras import metrics
from utils.ensembl2entrez import ensembl2entrez_convertor
from utils.ensembl2entrez import entrez2ensembl_convertor
from keras.utils import plot_model
from keras.models import Model

batch_size = 10
latent_dim = 2
intermediate_dim = 10000
epochs = 50
epsilon_std = 1.0

class VAEgo:

    def __init__(self, original_dim):
        self.vae=None
        self.original_dim = original_dim

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon


    def build_go(self, gene_list, go2genes, genes2go, vertices, edges):
        regex = re.compile(r"[\s, \,, \+]", re.IGNORECASE)
        count=0
        genes2input = {}
        for k,v in genes2go.iteritems():
            count+=1
            print "cur count: {}".format(count)
            if count==250: break
            e2e_id = entrez2ensembl_convertor([k])
            if len(e2e_id)==0: continue
            genes2input[k]= Input(shape=(1,), name=e2e_id[0] ) # Dense(1, )(Input(shape=(1,)))

        for k, v in vertices.iteritems():
            cur_layer = []
            if v["n_children"]==0:
                for cur_entrez in go2genes[k]:
                    if genes2input.has_key(cur_entrez):
                        cur_layer.append(genes2input[cur_entrez])
                go_name = regex.sub( "_", v["name"])

                if len(cur_layer) == 1:
                    v["neuron"]=Dense(1, activation='relu', name=go_name)(cur_layer[0])
                if len(cur_layer) >1:
                    v["neuron"] = Dense(1, activation='relu', name=go_name)(concatenate(cur_layer))



        vertices_copy = vertices
        neuron_count=0
        is_converged=False
        while not is_converged:
            print "current count : {}".format(neuron_count)
            is_converged = True
            for k, v  in sorted([(k, v) for k, v in vertices_copy.iteritems()], key=lambda x: max(x[1]["depth"]), reverse=True):
                if v.has_key("neuron"):
                    continue
                is_connectable = True
                # for cur_child in v["obj"].children:
                #     if not vertices[cur_child.id].has_key("neuron"):
                #         is_connectable = False
                if is_connectable:
                    inputs = [vertices[cur_child.id]["neuron"] for cur_child in v["obj"].children if vertices[cur_child.id].has_key("neuron")]
                    go_name = regex.sub("_", v["name"])
                    if len(inputs)==1:
                        v["neuron"] = Dense(1, activation='relu', name=go_name)(inputs[0])
                        neuron_count += 1
                        is_converged = False
                    if len(inputs)>1:
                        v["neuron"] = Dense(1, activation='relu', name=go_name)(concatenate(inputs))
                        neuron_count+=1
                        is_converged = False
        print "converged with {} intermediate neurons".format(neuron_count)

        model = Model([v for k,v in genes2input.iteritems()], vertices_copy[[k for k,v in vertices_copy.iteritems() if min(v["depth"])==1][0]]['neuron'])
        plot_model(model, to_file=os.path.join(constants.OUTPUT_GLOBAL_DIR, "model.svg"))




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

    def train_go(self, x_data, y_data):
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






