import re
import numpy as np
import constants
import os
import math
import pandas as pd
import keras
from keras.layers import Dense, Input, concatenate, Lambda
from keras import backend as K
from utils.ensembl2entrez import entrez2ensembl_convertor
from keras.utils import plot_model
from keras.models import Model
from keras import metrics
from constants import app_config

batch_size = 10
epsilon_std = 1.0


class VAEgo:

    def __init__(self, original_dim):
        self.vae = None
        self.encoder = None
        self.original_dim = original_dim

    def sampling(self, args, latent_dim):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    #    def vae_loss():

    def build_go(self, gene_list, go2genes, genes2go, vertices, edges, number_of_neurons, latent_dim):
        regex = re.compile(r"[\s, \,, \+, \:, \- ,\(,\, \), \' , \[ , \], \=, \<, \>]", re.IGNORECASE)

        count = 0
        gene_list = [x[:x.index(".")] for x in gene_list]
        genes2input = {}

        # print "prepare input layer"
        # for k,v in genes2go.iteritems():
        #     count+=1
        #     print count
        #     e2e_id = entrez2ensembl_convertor([k])
        #     if len(e2e_id)==0 or e2e_id[0] not in gene_list: continue
        #     genes2input[k]= Input(shape=(1,), name="{}_{}_{}".format(e2e_id[0],str(k),"input") ) # Dense(1, )(Input(shape=(1,)))
        input_ensembl_ids = []
        print "connect input layer to GO leafs"
        for k, v in vertices.iteritems():
            cur_layer = []
            if v["n_children"] == 0 or True:  # probably genes would be connect to upper layers
                for cur_entrez in go2genes[k]:
                    if not genes2input.has_key(cur_entrez):
                        e2e_id = entrez2ensembl_convertor([cur_entrez])
                        if len(e2e_id) != 0 and e2e_id[0] in gene_list:
                            input_ensembl_ids.append(e2e_id[0])
                            genes2input[cur_entrez] = Input(shape=(1,),
                                                            name="{}_{}_{}".format(e2e_id[0], str(cur_entrez), "input"))

                    if genes2input.has_key(cur_entrez):
                        cur_layer.append(cur_entrez)

                go_name = regex.sub(app_config["go_separator"], v["name"]) + "_converged"

                v["neuron_converged"] = Dense(number_of_neurons,
                                              activation=app_config['activation_function'], name=go_name)
                v["neuron_converged_inputs"] = cur_layer

        
        print "save input layer as list"
        file(os.path.join(constants.LIST_DIR, app_config["actual_vae_input_genes_file_name"]),'w+').write("\n".join(input_ensembl_ids))
        print "connect intermediate converged GO layers"
        neuron_count = 0

        for k, v in sorted([(k, v) for k, v in vertices.iteritems()], key=lambda x: max(x[1]["depth"]), reverse=True):
            if not v.has_key("neuron_converged"):
                go_name = regex.sub(app_config["go_separator"], v["name"] + "_converged")
                v["neuron_converged"] = Dense(number_of_neurons,
                                              activation=app_config['activation_function'], name=go_name)
                v["neuron_converged_inputs"] = []

            inputs = [cur_child.id for cur_child in v["obj"].children if
                      vertices[cur_child.id].has_key("neuron_converged")]

            v["neuron_converged_inputs"] = v["neuron_converged_inputs"] + inputs

        print "intermediate layers have {} neurons".format(neuron_count)

        for k, v in sorted([(k, v) for k, v in vertices.iteritems()], key=lambda x: max(x[1]["depth"]), reverse=True):
            if v.has_key("neuron_converged") and v.has_key("neuron_converged_inputs"):
                inputs = [genes2input[x] for x in v["neuron_converged_inputs"] if genes2input.has_key(x)] + \
                         [vertices[x]["neuron_converged"] for x in v["neuron_converged_inputs"] if vertices.has_key(x)]
                if len(inputs) == 0:
                    del vertices[k]

                if len(inputs) == 1:
                    v["neuron_converged"] = v["neuron_converged"](inputs[0])

                if len(inputs) > 1:
                    v["neuron_converged"] = v["neuron_converged"](concatenate(inputs))

        print [min(x["depth"]) for x in [v for k, v in vertices.iteritems()]]
        roots = [vertices[x] for x in [k for k, v in vertices.iteritems() if min(v["depth"]) == 1]]
        print "num of roots: {}".format(len(roots))
        if app_config["is_variational"]:
            print "root and sampling layers"
            inputs = [r["neuron_converged"] for r in roots]
            if len(inputs) > 1:
                z_mean = Dense(latent_dim, name="z_mean")(concatenate(inputs))
                z_log_var = Dense(latent_dim, name="z_log_var")(concatenate(inputs))
            else:
                z_mean = Dense(latent_dim, name="z_mean")(inputs[0])
                z_log_var = Dense(latent_dim, name="z_log_var")(inputs[0])
            z = Lambda(self.sampling, output_shape=(latent_dim,), name='z', arguments={"latent_dim":latent_dim})([z_mean, z_log_var])
            for r in roots:
                go_name = regex.sub(app_config["go_separator"], r["name"] + "_diverged")
                r['neuron_diverged'] = Dense(number_of_neurons,
                                             activation=app_config['activation_function'], name=go_name)(z)
        else:
            for r in roots: r['neuron_diverged'] = r['neuron_converged']

        print "connect intermediate diverged GO layers"
        neuron_count = 0
        is_converged = False
        while not is_converged:
            is_converged = True
            for k, v in sorted([(k, v) for k, v in vertices.iteritems()], key=lambda x: max(x[1]["depth"]),
                               reverse=False):

                if v.has_key("neuron_diverged"):
                    continue
                inputs = [vertices[cur_parent]["neuron_diverged"] for cur_parent in v["obj"]._parents if
                          vertices.has_key(cur_parent) and vertices[cur_parent].has_key("neuron_diverged")]
                go_name = regex.sub(app_config["go_separator"], v["name"] + "_diverged")
                if len(inputs) == 1:
                    v["neuron_diverged"] = Dense(number_of_neurons,
                                                 activation=app_config['activation_function'], name=go_name)(inputs[0])
                    neuron_count += 1
                    is_converged = False
                if len(inputs) > 1:
                    v["neuron_diverged"] = Dense(number_of_neurons,
                                                 activation=app_config['activation_function'], name=go_name)(
                        concatenate(inputs))
                    neuron_count += 1
                    is_converged = False
            print neuron_count
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
                if vertices.has_key(cur_go_term):  # and len(vertices[cur_go_term]["obj"].children) ==0:
                    neuron_parents.append(vertices[cur_go_term]["neuron_diverged"])
            if len(neuron_parents) == 1:
                genes2output[k] = Dense(1, activation='sigmoid', name="{}_{}_{}".format(e2e_id[0], str(k), "output"))(
                    neuron_parents[0])
            if len(neuron_parents) > 1:
                genes2output[k] = Dense(1, activation='sigmoid', name="{}_{}_{}".format(e2e_id[0], str(k), "output"))(
                    concatenate(neuron_parents))

        # self.vae = Model([v for k,v in genes2input.iteritems()], root['neuron_converged'])

        model_inputs = []
        model_outputs = []
        for k in genes2output.keys():
            model_inputs.append(genes2input[k])
            model_outputs.append(genes2output[k])

        assert len(model_inputs) == len(model_outputs), "different # of inputs and outputs"

        concatenated_inputs = concatenate(model_inputs)
        concatenated_outputs = concatenate(model_outputs)

        # instantiate encoder model
        self.encoder = Model(model_inputs, [z_mean, z_log_var, z], name='encoder')
        # self.encoder.summary()
        plot_model(self.encoder, to_file=os.path.join(constants.OUTPUT_GLOBAL_DIR, "encoder.svg"))

        # self.decoder = Model()

        # Overall VAE model, for reconstruction and training
        self.vae = Model(model_inputs, model_outputs)  # concatenated_outputs

        if app_config["loss_function"] == "mse":
            reconstruction_loss = len(model_inputs) * metrics.mse(concatenated_inputs, concatenated_outputs)
        if app_config["loss_function"] == "cross_ent":
            reconstruction_loss = len(model_inputs) * metrics.binary_crossentropy(concatenated_inputs,
                                                                                  concatenated_outputs)

        if app_config["is_variational"]:
            kl = -0.5 * K.sum(1 + z_log_var - K.exp(z_log_var) - K.square(z_mean), axis=1)
            self.vae.add_loss(K.mean(reconstruction_loss + kl))  # weighting? average?

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

        self.vae.compile(optimizer='rmsprop')  # , loss=loss
        print "number of inputs: {}".format(len(model_inputs))
        print "number of outputs: {}".format(len(model_outputs))

        plot_model(self.vae, to_file=os.path.join(constants.OUTPUT_GLOBAL_DIR, "model.svg"))

    def train_go(self, header_ensembl_ids,gene_expression_data, num_of_epochs, vae_weights_fname = "VAE_weights.h5"):
        print np.shape(gene_expression_data)
        print "about to start training"
        header_ensembl_ids = [x[:x.index(".")] for x in header_ensembl_ids]
        input_ids = []

        print "about prepare samples"
        for cur_input in self.vae.input:
            input_id = str(cur_input.name)[:-2]
            input_ids.append(input_id[:input_id.index("_")])

        input_data_sorted = [None for x in input_ids]

        ensembl_by_index = []
        for i, cur_input_id in enumerate(input_ids):
            try:
                input_data_sorted[i] = gene_expression_data[:, header_ensembl_ids.index(cur_input_id)]
            except:
                pass
            x = 1


        for i, data in enumerate(input_data_sorted):
            if data is None:
                input_data_sorted[i] = np.array([0 for x in range(len(gene_expression_data))]).reshape(
                    len(gene_expression_data), 1)



        concatenated_cols = input_data_sorted[0]
        for cur_col in input_data_sorted[1:]:
            concatenated_cols = np.c_[concatenated_cols, cur_col]

        print "done prepare samples"

        # trimmer_index = (len(concatenated_cols) / 10) * 10
        # # if app_config["split_data"]:
        # print "trimmer_index " + str(trimmer_index)
        # concatenated_cols = concatenated_cols[:trimmer_index]
        # print "concatenated_cols " + str(len(concatenated_cols))

        ratio = 0.9
        # print "split according ratio {}".format()
        # ratio = int(math.floor(len(concatenated_cols) * ratio))
        # print len(concatenated_cols[:ratio])
        # print len(concatenated_cols[ratio:])
        # batch_size = len(concatenated_cols[ratio:])
        x_total = np.array(concatenated_cols).astype(np.float64)
        x_train = x_total # np.array(concatenated_cols[:ratio]).astype(np.float64)
        # x_test = np.array(concatenated_cols[ratio:]).astype(np.float64)
        # # y_train = np.array(y_data[:ratio]).astype(np.int32)
        # # y_test = np.array(y_data[ratio:]).astype(np.int32)

        # else:
        #     x_train = np.array(concatenated_cols).astype(np.float64)
        #     y_train = np.array(y_data).astype(np.int32)

        # self.vae.fit([x for x in x_train.T], y_train,
        #         shuffle=True,
        #         epochs=epochs,
        #         batch_size=batch_size,
        #         validation_data=([x for x in x_test.T], y_test))

        # json_log = open('loss_log.json', mode='wt', buffering=1)
        # json_logging_callback = LambdaCallback(
        #     on_epoch_end=lambda epoch, logs: json_log.write(
        #         json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
        #     on_train_end=lambda logs: json_log.close()
        # )
        # inp = self.vae.input  # input placeholder
        # outputs = [layer.output for layer in self.vae.layers]  # all layer outputs
        # functors = [K.function([inp], [out]) for out in outputs]

        if app_config["load_weights"]:
            self.vae = self.vae.load_weights(os.path.join(constants.OUTPUT_GLOBAL_DIR, vae_weights_fname))
        else:
            hist = self.vae.fit([x for x in x_train.T],
                                shuffle=True,
                                epochs=num_of_epochs,
                                batch_size=batch_size,
                                validation_split=0.1,
                                # validation_data=([x for x in x_test.T], None)
                                                  )
            self.vae.save_weights(os.path.join(constants.OUTPUT_GLOBAL_DIR, vae_weights_fname))
        return x_total

        # print(history.losses)
        # print(hist.history)


        #### Test #####
        # score = self.vae.evaluate(x_test, y_test, verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])

    def test_go(self, test_input_data, patients_list, y_data, latent_dim, vae_projections_fname = "VAE_compress.tsv"):
        print np.shape(test_input_data)
        latent_space = self.encoder.predict([x for x in test_input_data.T], batch_size=batch_size)
        print np.shape(latent_space[2])

        print "Saving VAE data.."
        # x_projected = np.insert(x_projected,[0], np.array(latent_dim),axis = 0)
        # np.save(os.path.join(constants.OUTPUT_GLOBAL_DIR, "PCA_compress.txt"), x_projected)
        latent_data = latent_space[2].T
        pca_data = pd.DataFrame(latent_data, index=range(latent_dim), columns=patients_list)
        pca_data.index.name = 'VAE'
        pca_data.to_csv(os.path.join(constants.OUTPUT_GLOBAL_DIR, vae_projections_fname), sep='\t')
    # print x_test.T[:3]
    # pred = [x for x in x_test.T]
    # pred = [np.array([y[0] for y in pred]), np.array([y[1] for y in pred]), np.array([y[2] for y in pred])]
    # print pred
    # print self.vae.predict(pred)
