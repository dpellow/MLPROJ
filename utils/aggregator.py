import os


def aggregate_all(latent_dims, epochs, thresholds, num_neurons, ts):
    for d in latent_dims:
        for e in epochs:
            for t in thresholds:
                for n in num_neurons:
                    cur_result = open(os.path.join("results_{}_{}_{}_{}.txt".format(d, e, t, n))).read()
                    open(os.path.join("results_total_{}.txt".format(ts)), 'w+').write(cur_result)
