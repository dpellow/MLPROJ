# Run main in a loop

import constants
import os
import main
import json
import time
from utils.aggregator import aggregate_all
from utils.aggregator import clear_all

epochs = [1,20,50,100,200]

#David's run 1
thresholds = [100,500]
num_neurons = [5,10]
latent_dims = [2,5]

#David's run 2
#thresholds = [100,500]
#num_neurons = [50]
#latent_dims = [2,5,10,50]

#David's run 3 
#thresholds = [100,500]
#num_neurons = [100]
#latent_dims = [2,5,10,50]


#Asia run 1
#thresholds=[1500]
#num_neurons=[5,10]
#latent_dims=[2,5]

#asia run 2
#thresholds=[1500]
#num_neurons = [50]
#latent_dims = [2,5,10,50]

#asia run 3
#thresholds=[1500]
#num_neurons = [100]
#latent_dims = [2,5,10,50]



clear_all(latent_dims, thresholds, num_neurons)


rows_output=[]
for d in latent_dims:
    for t in thresholds:
        for n in num_neurons:
	  if n < d: continue
            ##result proc
              results = main.run(var_th_index=t,number_of_neurons=n, latent_dim=d, num_of_epochs=epochs)
 
              with open(os.path.join(constants.OUTPUT_GLOBAL_DIR, "results_{}_{}_{}.txt".format(d,t,n)),'a+') as f:
                f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format("threshold", "number_of_neurons", "latent_dims", "epochs", "type", "average", "variance"))
                for cur_line in results:
                    cur_row = str(t)+"\t"+str(n)+"\t"+str(d)+"\t"
                    method = cur_line['type']
                    avg = cur_line['avg']
                    var = cur_line['var']
                    e = cur_line.get('epochs'," " )
                    cur_row += (str(e)+'\t'+method +'\t'+str(avg)+'\t'+str(var)+'\n')
                    f.write(cur_row)



aggregate_all(latent_dims, thresholds, num_neurons,time.time())
