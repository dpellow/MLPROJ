import re
import numpy as np
import constants
import os
import math
from keras.layers import Dense, Input, concatenate, Lambda
from keras import backend as K
from utils.ensembl2entrez import entrez2ensembl_convertor
from keras.utils import plot_model
from keras.models import Model
from keras import metrics
from constants import app_config

batch_size = 8
epochs = 50
epsilon_std = 1.0

class VAEgo:

    def __init__(self, original_dim):
        self.vae=None
        self.original_dim = original_dim

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], app_config["latent_dim"]), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

#    def vae_loss():



    def build_go(self, gene_list, go2genes, genes2go, vertices, edges):
        regex = re.compile(r"[\s, \,, \+, \:, \- ,\(,\, \), \' , \[ , \], \=, \<, \>]", re.IGNORECASE)
        count=0
        gene_list = [x[:x.index(".")] for x in gene_list]
        genes2input = {}

        # print "prepare input layer"
        # for k,v in genes2go.iteritems():
        #     count+=1
        #     print count
        #     e2e_id = entrez2ensembl_convertor([k])
        #     if len(e2e_id)==0 or e2e_id[0] not in gene_list: continue
        #     genes2input[k]= Input(shape=(1,), name="{}_{}_{}".format(e2e_id[0],str(k),"input") ) # Dense(1, )(Input(shape=(1,)))

        print "connect input layer to GO leafs"
        for k, v in vertices.iteritems():
            cur_layer = []
            if v["n_children"]==0 or True: # probably genes would be connect to upper layers
                for cur_entrez in go2genes[k]:
                    if not genes2input.has_key(cur_entrez):
                        e2e_id = entrez2ensembl_convertor([cur_entrez])
                        if len(e2e_id) != 0 and e2e_id[0] in gene_list:
                            genes2input[cur_entrez] = Input(shape=(1,), name="{}_{}_{}".format(e2e_id[0], str(cur_entrez), "input"))

                    if genes2input.has_key(cur_entrez):
                        cur_layer.append(cur_entrez)

                go_name = regex.sub( app_config["go_separator"], v["name"])+"_converged"

                v["neuron_converged"]=Dense(app_config["number_of_neurons"], activation=app_config['activation_function'], name=go_name)
                v["neuron_converged_inputs"] = cur_layer


        print "connect intermediate converged GO layers"
        neuron_count=0

        for k, v  in sorted([(k, v) for k, v in vertices.iteritems()], key=lambda x: max(x[1]["depth"]), reverse=True):
            if not v.has_key("neuron_converged"):
                go_name = regex.sub(app_config["go_separator"], v["name"] + "_converged")
                v["neuron_converged"] = Dense(app_config["number_of_neurons"], activation=app_config['activation_function'], name=go_name)
                v["neuron_converged_inputs"] = []

            inputs = [cur_child.id for cur_child in v["obj"].children if vertices[cur_child.id].has_key("neuron_converged")]

            v["neuron_converged_inputs"] = v["neuron_converged_inputs"] + inputs

        print "intermediate layers have {} neurons".format(neuron_count)

        for k, v in sorted([(k, v) for k, v in vertices.iteritems()], key=lambda x: max(x[1]["depth"]), reverse=True):
            if v.has_key("neuron_converged") and v.has_key("neuron_converged_inputs"):
                inputs = [genes2input[x] for x in v["neuron_converged_inputs"] if genes2input.has_key(x)] + \
                [vertices[x]["neuron_converged"] for x in v["neuron_converged_inputs"] if vertices.has_key(x)]
                if len(inputs) == 0:
                    del vertices[k]

                if len(inputs)==1:
                    v["neuron_converged"] = v["neuron_converged"](inputs[0])

                if len(inputs)>1:
                    v["neuron_converged"] = v["neuron_converged"](concatenate(inputs))

        roots = [vertices[x] for x in [k for k, v in vertices.iteritems() if min(v["depth"]) == 1]]
        if app_config["is_variational"]:
            print "root and sampling layers"
            inputs = [r["neuron_converged"] for r in roots]
            if len(inputs)> 1:
                z_mean = Dense(app_config["latent_dim"], name = "z_mean")(concatenate(inputs))
                z_log_var = Dense(app_config["latent_dim"], name = "z_log_var")(concatenate(inputs))
            else:
                z_mean = Dense(app_config["latent_dim"], name = "z_mean")(inputs[0])
                z_log_var = Dense(app_config["latent_dim"], name = "z_log_var")(inputs[0])
            z = Lambda(self.sampling, output_shape=(app_config["latent_dim"],), name='z')([z_mean, z_log_var])
            for r in roots:
                go_name = regex.sub(app_config["go_separator"], r["name"]+"_diverged")
                r['neuron_diverged'] = Dense(app_config["number_of_neurons"], activation=app_config['activation_function'], name=go_name)(z)
        else:
            for r in roots: r['neuron_diverged'] = r['neuron_converged']



        print "connect intermediate diverged GO layers"
        neuron_count=0
        is_converged=False
        while not is_converged:
            is_converged = True
            for k, v  in sorted([(k, v) for k, v in vertices.iteritems()], key=lambda x: max(x[1]["depth"]), reverse=False):

                if v.has_key("neuron_diverged"):
                    continue
                inputs = [vertices[cur_parent]["neuron_diverged"] for cur_parent in v["obj"]._parents if vertices.has_key(cur_parent) and vertices[cur_parent].has_key("neuron_diverged")]
                go_name = regex.sub(app_config["go_separator"], v["name"]+"_diverged")
                if len(inputs)==1:
                    v["neuron_diverged"] = Dense(app_config["number_of_neurons"], activation=app_config['activation_function'], name=go_name)(inputs[0])
                    neuron_count += 1
                    is_converged = False
                if len(inputs)>1:
                    v["neuron_diverged"] = Dense(app_config["number_of_neurons"], activation=app_config['activation_function'], name=go_name)(concatenate(inputs))
                    neuron_count+=1
                    is_converged = False

        print "intermediate layers have {} neurons".format(neuron_count)


        genes2output = {}
        print "connect input layer to output leafs"

        for k, v in genes2go.iteritems():
            count += 1
    #        print count
            e2e_id = entrez2ensembl_convertor([k])
            if len(e2e_id) == 0 or e2e_id[0] not in gene_list: continue
            neuron_parents = []
            for cur_go_term in genes2go[k]:
                if vertices.has_key(cur_go_term): # and len(vertices[cur_go_term]["obj"].children) ==0:
                    neuron_parents.append(vertices[cur_go_term]["neuron_diverged"])
            if len(neuron_parents)==1:
                genes2output[k]=Dense(1, activation=app_config["activation_function"], name="{}_{}_{}".format(e2e_id[0],str(k),"output"))(neuron_parents[0])
            if len(neuron_parents)>1:
                genes2output[k]=Dense(1, activation=app_config["activation_function"], name="{}_{}_{}".format(e2e_id[0],str(k),"output"))(concatenate(neuron_parents))


        # self.vae = Model([v for k,v in genes2input.iteritems()], root['neuron_converged'])

        model_inputs = []
        model_outputs = []
        for k in genes2output.keys():
            model_inputs.append(genes2input[k])
            model_outputs.append(genes2output[k])

        assert len(model_inputs) == len(model_outputs), "different # of inputs and outputs"

        concatenated_inputs = concatenate(model_inputs)
        concatenated_outputs = concatenate(model_outputs)

        self.vae = Model(model_inputs, model_outputs) # concatenated_outputs

        if app_config["loss_function"]=="mse":
            reconstruction_loss = len(model_inputs)*metrics.mse(concatenated_inputs,concatenated_outputs)
        if app_config["loss_function"]=="cross_ent":
            reconstruction_loss = len(model_inputs)*metrics.binary_crossentropy(concatenated_inputs,concatenated_outputs)

        if app_config["is_variational"]:
            kl = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=1)
            self.vae.add_loss(reconstruction_loss+kl) # weighting? average?

        else:
            self.vae.add_loss(reconstruction_loss)
#            for i in range(len(model_inputs)):
    #            self.vae.add_loss(metrics.binary_crossentropy(model_inputs[i], model_outputs[i]))

        # loss = metrics.binary_crossentropy(concatenated_inputs, concatenated_outputs)
        # loss = {}
        # for i in range(len(model_inputs)):
        #     loss[str(model_outputs[i].name[:model_outputs[i].name.index("/")])] = 'binary_crossentropy' # \
                # metrics.binary_crossentropy(model_inputs[i], model_outputs[i])
        # self.vae.add_loss(loss)


        self.vae.compile(optimizer='rmsprop') # , loss=loss
        plot_model(self.vae, to_file=os.path.join(constants.OUTPUT_GLOBAL_DIR, "model.svg"))



    def train_go(self, header_ensembl_ids, gene_expression_data, y_data):
        # train the VAE on MNIST digits
        header_ensembl_ids = [x[:x.index(".")] for x in header_ensembl_ids]
        input_ids = []
        for cur_input in self.vae.input:
            input_id = str(cur_input.name)[:-2]
            input_ids.append(input_id[:input_id.index("_")])

        input_data_sorted = [None for x in input_ids]

        ensembl_by_index = []
        for i, cur_input_id in enumerate(input_ids):
            try:
                input_data_sorted[i]=gene_expression_data[:,header_ensembl_ids.index(cur_input_id)]
            except:
                pass
            x=1

        for i, data in enumerate(input_data_sorted):
            if data is None:
                input_data_sorted[i] = np.array([0 for x in range(len(gene_expression_data))]).reshape(len(gene_expression_data), 1)

        concatenated_cols = input_data_sorted[0]
        for cur_col in input_data_sorted[1:]:
            concatenated_cols = np.c_[concatenated_cols, cur_col]

        trimmer_index = (len(concatenated_cols)/10)*10
        concatenated_cols=concatenated_cols[:trimmer_index]

        ratio = int(math.floor(len(concatenated_cols) * 0.9))

        x_train = np.array(concatenated_cols[:ratio]).astype(np.float32)
        x_test = np.array(concatenated_cols[ratio:]).astype(np.float32)

        y_train = np.array(y_data[:ratio]).astype(np.int32)
        y_test = np.array(y_data[ratio:]).astype(np.int32)
        # self.vae.fit([x for x in x_train.T], y_train,
        #         shuffle=True,
        #         epochs=epochs,
        #         batch_size=batch_size,
        #         validation_data=([x for x in x_test.T], y_test))

        self.vae.fit([x for x in x_train.T],
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([x for x in x_test.T], None))
