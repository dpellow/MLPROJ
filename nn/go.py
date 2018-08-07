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

batch_size = 8
latent_dim = 2
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
        regex = re.compile(r"[\s, \,, \+, \:, \- ,\(,\, '\)]", re.IGNORECASE)
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
            if v["n_children"]==0:
                for cur_entrez in go2genes[k]:
                    if not genes2input.has_key(cur_entrez):
                        e2e_id = entrez2ensembl_convertor([cur_entrez])
                        if len(e2e_id) != 0 and e2e_id[0] in gene_list:
                            genes2input[cur_entrez] = Input(shape=(1,), name="{}_{}_{}".format(e2e_id[0], str(cur_entrez), "input"))

                    if genes2input.has_key(cur_entrez):
                        cur_layer.append(genes2input[cur_entrez])

                go_name = regex.sub( "_", v["name"])

                if len(cur_layer) == 1:
                    v["neuron_converged"]=Dense(1, activation='relu', name=go_name)(cur_layer[0])
                if len(cur_layer) >1:
                    v["neuron_converged"] = Dense(1, activation='relu', name=go_name)(concatenate(cur_layer))


        print "connect intermediate converged GO layers"
        neuron_count=0
        is_converged=False
        while not is_converged:
            is_converged = True
            for k, v  in sorted([(k, v) for k, v in vertices.iteritems()], key=lambda x: max(x[1]["depth"]), reverse=True):
                if v.has_key("neuron_converged"):
                    continue
                is_connectable = True
                # for cur_child in v["obj"].children:
                #     if not vertices[cur_child.id].has_key("neuron"):
                #         is_connectable = False
                if is_connectable:
                    inputs = [vertices[cur_child.id]["neuron_converged"] for cur_child in v["obj"].children if vertices[cur_child.id].has_key("neuron_converged")]
                    go_name = regex.sub("_", v["name"]+"_converged")
                    if len(inputs)==1:
                        v["neuron_converged"] = Dense(1, activation='relu', name=go_name)(inputs[0])
                        neuron_count += 1
                        is_converged = False
                    if len(inputs)>1:
                        v["neuron_converged"] = Dense(1, activation='relu', name=go_name)(concatenate(inputs))
                        neuron_count+=1
                        is_converged = False
        print "intermediate layers have {} neurons".format(neuron_count)


        root = vertices[[k for k, v in vertices.iteritems() if min(v["depth"]) == 1][0]]
        root['neuron_diverged'] = root['neuron_converged']

        print "connect intermediate diverged GO layers"
        neuron_count=0
        is_converged=False
        while not is_converged:
            is_converged = True
            for k, v  in sorted([(k, v) for k, v in vertices.iteritems()], key=lambda x: max(x[1]["depth"]), reverse=False):

                if v.has_key("neuron_diverged"):
                    continue
                inputs = [vertices[cur_parent]["neuron_diverged"] for cur_parent in v["obj"]._parents if vertices.has_key(cur_parent) and vertices[cur_parent].has_key("neuron_diverged")]
                go_name = regex.sub("_", v["name"]+"_diverged")
                if len(inputs)==1:
                    v["neuron_diverged"] = Dense(1, activation='relu', name=go_name)(inputs[0])
                    neuron_count += 1
                    is_converged = False
                if len(inputs)>1:
                    v["neuron_diverged"] = Dense(1, activation='relu', name=go_name)(concatenate(inputs))
                    neuron_count+=1
                    is_converged = False

        print "intermediate layers have {} neurons".format(neuron_count)


        genes2output = {}
        print "connect input layer to output leafs"

        for k, v in genes2go.iteritems():
            count += 1
            print count
            e2e_id = entrez2ensembl_convertor([k])
            if len(e2e_id) == 0 or e2e_id[0] not in gene_list: continue
            neuron_parents = []
            for cur_go_term in genes2go[k]:
                if vertices.has_key(cur_go_term) and len(vertices[cur_go_term]["obj"].children) ==0:
                    neuron_parents.append(vertices[cur_go_term]["neuron_diverged"])
            if len(neuron_parents)==1:
                genes2output[k]=Dense(1, activation='relu', name="{}_{}_{}".format(e2e_id[0],str(k),"output"))(neuron_parents[0])
            if len(neuron_parents)>1:
                genes2output[k]=Dense(1, activation='relu', name="{}_{}_{}".format(e2e_id[0],str(k),"output"))(concatenate(neuron_parents))


        # self.vae = Model([v for k,v in genes2input.iteritems()], root['neuron_converged'])

        model_inputs = []
        model_outputs = []
        for k in genes2output.keys():
            model_inputs.append(genes2input[k])
            model_outputs.append(genes2output[k])

        # concatenated_inputs = concatenate(model_inputs)
        # concatenated_outputs = concatenate(model_outputs)
        
        self.vae = Model(model_inputs, model_outputs) # concatenated_outputs

        for i in range(len(model_inputs)):
            self.vae.add_loss(metrics.binary_crossentropy(model_inputs[i], model_outputs[i]))

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








