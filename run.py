# Run main in a loop

import constants
import os
import main
import json
import time
from utils.aggregator import aggregate_all
from utils.aggregator import clear_all
latent_dims = [5]# ,5,10,20,30,50,75,100]
epochs = [100]


#thresholds = [2000,5000,10000,15000,20000]
#num_neurons = [100,75,50,30,20,10,5]


#David's run
#thresholds = [2000,5000,10000]#,15000,20000]
#num_neurons = [100,75,50]#,30,20,10,5]


#Hagai's run
thresholds = [100,] # , 2000,5000,10000]#,15000,20000]
num_neurons = [30]# ,20,10,5]


#Asia's run
#thresholds = [15000,20000]
#num_neurons = [100,75,50,30,20,10,5]

clear_all(latent_dims, thresholds, num_neurons)


rows_output=[]
for d in latent_dims:
    for t in thresholds:
        for n in num_neurons:
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
