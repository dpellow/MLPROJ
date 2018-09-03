import os
import constants

def aggregate_all(latent_dims, thresholds, num_neurons, ts):
    has_header =False
    with open(os.path.join(constants.OUTPUT_GLOBAL_DIR, "results_total_{}.txt".format(ts)), 'a+') as f:
        for d in latent_dims:
            for t in thresholds:
                for n in num_neurons:
                    cur_result = open(os.path.join(constants.OUTPUT_GLOBAL_DIR, "results_{}_{}_{}.txt".format(d, t, n))).readlines()
                    if not has_header:
                        f.writelines(cur_result)
                        has_header = True
                    else:
                        f.writelines(cur_result[1:])


def clear_all(latent_dims, thresholds, num_neurons):
    for d in latent_dims:
        for t in thresholds:
            for n in num_neurons:
                try:
                    os.remove(os.path.join(constants.OUTPUT_GLOBAL_DIR, "results_{}_{}_{}.txt".format(d, t, n)))
                except:
                    print "could not remove {} ".format(constants.OUTPUT_GLOBAL_DIR, "results_{}_{}_{}.txt".format(d, t, n))
                    pass
